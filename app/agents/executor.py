from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from app.agents.base_agent import BaseAgent
from app.db.mongodb import get_db
from app.services.alpaca_trader import (
    place_bracket_order, close_position, get_orders, cancel_order,
    get_positions, update_stop_loss,
    # 🆕 nuove funzioni
    place_notional_buy, wait_for_fill, place_brackets_after_fill,
)
from app.services.telegram_bot import send_telegram


class Executor(BaseAgent):
    """
    ⚡ AGENTE 4: Executor v3.0
    Esegue trade, trailing stop, notifiche Telegram, cancella ordini stale.
    
    v3.0 — Supporto fractional/notional:
    - BUY: place_notional_buy → wait_for_fill → place_brackets_after_fill
    - SELL: close_position (gestisce fractional nativamente)
    - SL/TP separati invece di bracket (Alpaca non supporta bracket+notional)
    
    Trade Sync v4 mantenuto, ora con qty float.
    Timezone dinamico con zoneinfo.
    """

    def __init__(self):
        super().__init__(name="executor", version="3.0")

    def default_params(self) -> dict:
        return {
            "limit_price_buffer_pct": 0.5,
            "stale_order_hours": 2,
            "send_telegram": True,
            "allow_premarket": False,
            "trailing_level_1_pct": 5.0,
            "trailing_level_2_pct": 8.0,
            "trailing_level_3_pct": 12.0,
            # 🆕 Sizing mode (sincronizzato da settings)
            "position_sizing_mode": "notional",
            # 🆕 Timeout per fill polling
            "fill_timeout_sec": 15,
        }

    # ==========================================
    # Timezone dinamico con zoneinfo
    # ==========================================
    @staticmethod
    def is_market_open() -> dict:
        et_now = datetime.now(ZoneInfo("America/New_York"))
        is_weekday = et_now.weekday() < 5
        extended_open = et_now.replace(hour=4, minute=0, second=0, microsecond=0)
        extended_close = et_now.replace(hour=20, minute=0, second=0, microsecond=0)
        regular_open = et_now.replace(hour=9, minute=30, second=0, microsecond=0)
        regular_close = et_now.replace(hour=16, minute=0, second=0, microsecond=0)
        is_regular = regular_open <= et_now <= regular_close
        is_extended = extended_open <= et_now <= extended_close
        return {
            "is_open": is_weekday and is_extended,
            "is_regular": is_weekday and is_regular,
            "is_extended": is_weekday and is_extended and not is_regular,
            "eastern_time": et_now.strftime("%Y-%m-%d %H:%M:%S ET"),
            "is_weekday": is_weekday,
            "session": "regular" if is_regular else ("extended" if is_extended else "closed"),
        }

    async def _cancel_stale_orders(self, params: dict) -> int:
        """
        🔧 v3.1 — Cancella SOLO ordini BUY stale (entry non eseguiti).
        Non cancella mai SL/TP di posizioni aperte (che sono ordini sell con type=stop/limit).
        
        Logica:
        - side=buy + status non filled da >24h → ENTRY MAI PARTITO → cancella
        - side=sell + type=stop/limit → SL o TP di posizione aperta → MAI cancellare
        - Tutto il resto → log e skip
        """
        stale_hours = params.get("stale_order_hours", 2)
        cutoff = datetime.utcnow() - timedelta(hours=stale_hours)
        cancelled = 0
        skipped_sl_tp = 0
        
        orders = await get_orders(status="open", limit=50)
        if not orders:
            return 0
        
        for order in orders:
            side = order.get("side", "")
            order_type = order.get("type", "")
            symbol = order.get("symbol", "?")
            order_id = order.get("id", "")
            
            # 🛡️ PROTEZIONE: mai cancellare SL/TP (sell stop/limit)
            if side == "sell" and order_type in ("stop", "limit", "stop_limit", "trailing_stop"):
                skipped_sl_tp += 1
                continue
            
            # 🔧 Cancella solo BUY entry stale
            if side != "buy":
                continue
            
            created = order.get("created_at", "")
            if not created:
                continue
            
            try:
                order_time = datetime.fromisoformat(created.replace("Z", "+00:00"))
                if order_time.replace(tzinfo=None) < cutoff:
                    result = await cancel_order(order_id)
                    if result is not None:
                        cancelled += 1
                        print(f"  ⏰ Cancelled stale BUY: {symbol} ({order_id[:8]}...)")
            except (ValueError, TypeError) as e:
                print(f"  ⚠️ Parse date error for {symbol}: {e}")
                continue
        
        if skipped_sl_tp > 0:
            print(f"  🛡️ Protected {skipped_sl_tp} SL/TP orders from stale cancellation")
        
        return cancelled

    async def _send_notification(self, message: str, params: dict):
        if params.get("send_telegram", True):
            await send_telegram(message)

    async def _calc_days_held(self, db, ticker: str) -> int:
        buy_trade = await db.trade_history.find_one(
            {"ticker": ticker, "side": "buy", "sell_linked": {"$ne": True}},
            sort=[("date", -1)]
        )
        if buy_trade and buy_trade.get("date"):
            days = (datetime.utcnow() - buy_trade["date"]).days
            return max(days, 1)
        return 0

# ==========================================
    # 🆕 v3.2 — SOFTWARE SL/TP (per fractional shares)
    # ==========================================
    async def _check_software_sl_tp(self, positions: list, params: dict) -> dict:
        """
        🆕 v3.2 — Controlla software SL/TP per posizioni con fractional shares.
        
        Alpaca NON supporta SL/TP nativi con fractional shares (limit/stop + GTC).
        Quindi l'Executor gestisce SL/TP "software":
        - Legge SL/TP salvati in db.trade_history quando il buy è stato piazzato
        - Ad ogni run confronta prezzo corrente con SL/TP
        - Se prezzo <= SL → chiude posizione (STOP_LOSS_HIT)
        - Se prezzo >= TP → chiude posizione (TAKE_PROFIT_HIT)
        
        Ritorna dict con:
        - triggered: lista posizioni chiuse
        - checked: totale posizioni controllate
        - errors: lista errori
        """
        db = get_db()
        result = {"triggered": [], "checked": 0, "errors": []}
        
        if not positions:
            return result
        
        for pos in positions:
            symbol = pos.get("symbol")
            current_price = float(pos.get("current_price", 0))
            entry_price = float(pos.get("avg_entry_price", 0))
            shares = float(pos.get("qty", 0))
            
            if not symbol or current_price <= 0 or entry_price <= 0:
                continue
            
            # Trova il BUY corrispondente (ultimo, non ancora chiuso)
            buy_trade = await db.trade_history.find_one(
                {
                    "ticker": symbol,
                    "side": "buy",
                    "sell_linked": {"$ne": True}
                },
                sort=[("date", -1)]
            )
            
            if not buy_trade:
                # Nessun buy trovato in trade_history → skip
                continue
            
            result["checked"] += 1
            
            stop_loss = buy_trade.get("stop_loss", 0)
            target = buy_trade.get("target", 0)
            
            # Trailing stop dinamico (se esiste, sovrascrive SL iniziale)
            trailing = await db.trailing_stops.find_one({"ticker": symbol})
            if trailing:
                trailing_stop = trailing.get("stop_price", 0)
                if trailing_stop > stop_loss:
                    stop_loss = trailing_stop
            
           # ============================================
            # 🛡️ v3.3 — SANITY CHECKS su SL/TP prima del trigger
            # 🔧 v4.2 — Distingui break-even (SL = entry) da invalid (SL > entry)
            # ============================================
            if stop_loss > 0:
                tolerance = entry_price * 0.001  # 0.1% tolerance
                if stop_loss > entry_price + tolerance:
                    # SL sopra entry di più dello 0.1% = bug reale
                    print(f"  ⚠️ INVALID SL for {symbol}: stop_loss ${stop_loss:.2f} > entry ${entry_price:.2f} (skipping SL check)")
                    stop_loss = 0
                elif abs(stop_loss - entry_price) <= tolerance:
                    # SL ~= entry = break-even valido (post APM scale-out)
                    print(f"  🛡️ BREAK-EVEN SL for {symbol}: ${stop_loss:.2f} (post scale-out)")
            
            # Se target <= entry_price → è un bug upstream (Alpha/Risk)
            if target > 0 and target <= entry_price:
                print(f"  ⚠️ INVALID TP for {symbol}: target ${target:.2f} <= entry ${entry_price:.2f} (skipping TP check)")
                target = 0
            
            # ============================================
            # CHECK STOP LOSS
            # ============================================
            if stop_loss > 0 and current_price <= stop_loss:
                try:
                    close_result = await close_position(symbol)
                    if close_result is not None:
                        pnl_pct = round(((current_price - entry_price) / entry_price) * 100, 2)
                        pnl_dollar = round((current_price - entry_price) * shares, 2)
                        days_held = await self._calc_days_held(db, symbol)
                        
                        sell_order_id = f"sw_sl_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                        
                        # Log in trade_history
                        await db.trade_history.insert_one({
                            "ticker": symbol,
                            "side": "sell",
                            "entry_price": entry_price,
                            "exit_price": current_price,
                            "shares": float(shares),
                            "pnl_pct": pnl_pct,
                            "pnl_dollar": pnl_dollar,
                            "days_held": days_held,
                            "reason": "SOFTWARE_STOP_LOSS",
                            "setup_type": buy_trade.get("setup_type", "unknown"),
                            "sector": buy_trade.get("sector", "unknown"),
                            "rsi_at_entry": buy_trade.get("rsi_at_entry", 50),
                            "market_regime": buy_trade.get("market_regime", "UNKNOWN"),
                            "order_id": sell_order_id,
                            "buy_order_id": buy_trade.get("order_id", ""),
                            "agent": "executor_software_sl",
                            "date": datetime.utcnow(),
                            "source": "software_sl_tp",
                            "trigger_price": stop_loss,
                        })
                        
                        # Link buy → sell
                        await db.trade_history.update_one(
                            {"_id": buy_trade["_id"]},
                            {"$set": {"sell_linked": True, "sell_order_id": sell_order_id}}
                        )
                        
                        # Cleanup trailing stop
                        await db.trailing_stops.delete_one({"ticker": symbol})
                        
                        result["triggered"].append({
                            "ticker": symbol,
                            "reason": "SOFTWARE_STOP_LOSS",
                            "trigger_price": stop_loss,
                            "current_price": current_price,
                            "pnl_pct": pnl_pct,
                            "pnl_dollar": pnl_dollar,
                        })
                        
                        print(f"  🛑 SOFTWARE SL HIT {symbol}: ${current_price:.2f} <= ${stop_loss:.2f} "
                              f"(P&L {pnl_pct:+.2f}%, ${pnl_dollar:+.0f})")
                        continue  # Skip TP check per questo ticker
                except Exception as e:
                    result["errors"].append({"ticker": symbol, "action": "sl", "error": str(e)})
                    print(f"  ⚠️ SW SL error {symbol}: {e}")
            
            # ============================================
            # CHECK TAKE PROFIT
            # ============================================
            if target > 0 and current_price >= target:
                try:
                    close_result = await close_position(symbol)
                    if close_result is not None:
                        pnl_pct = round(((current_price - entry_price) / entry_price) * 100, 2)
                        pnl_dollar = round((current_price - entry_price) * shares, 2)
                        days_held = await self._calc_days_held(db, symbol)
                        
                        sell_order_id = f"sw_tp_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                        
                        await db.trade_history.insert_one({
                            "ticker": symbol,
                            "side": "sell",
                            "entry_price": entry_price,
                            "exit_price": current_price,
                            "shares": float(shares),
                            "pnl_pct": pnl_pct,
                            "pnl_dollar": pnl_dollar,
                            "days_held": days_held,
                            "reason": "SOFTWARE_TAKE_PROFIT",
                            "setup_type": buy_trade.get("setup_type", "unknown"),
                            "sector": buy_trade.get("sector", "unknown"),
                            "rsi_at_entry": buy_trade.get("rsi_at_entry", 50),
                            "market_regime": buy_trade.get("market_regime", "UNKNOWN"),
                            "order_id": sell_order_id,
                            "buy_order_id": buy_trade.get("order_id", ""),
                            "agent": "executor_software_tp",
                            "date": datetime.utcnow(),
                            "source": "software_sl_tp",
                            "trigger_price": target,
                        })
                        
                        await db.trade_history.update_one(
                            {"_id": buy_trade["_id"]},
                            {"$set": {"sell_linked": True, "sell_order_id": sell_order_id}}
                        )
                        
                        await db.trailing_stops.delete_one({"ticker": symbol})
                        
                        result["triggered"].append({
                            "ticker": symbol,
                            "reason": "SOFTWARE_TAKE_PROFIT",
                            "trigger_price": target,
                            "current_price": current_price,
                            "pnl_pct": pnl_pct,
                            "pnl_dollar": pnl_dollar,
                        })
                        
                        print(f"  🎯 SOFTWARE TP HIT {symbol}: ${current_price:.2f} >= ${target:.2f} "
                              f"(P&L {pnl_pct:+.2f}%, ${pnl_dollar:+.0f})")
                except Exception as e:
                    result["errors"].append({"ticker": symbol, "action": "tp", "error": str(e)})
                    print(f"  ⚠️ SW TP error {symbol}: {e}")
        
        if result["triggered"]:
            print(f"  💥 Software SL/TP: {len(result['triggered'])} triggered, "
                  f"{result['checked']} checked")
        
        return result
    
    async def _manage_trailing_stops(self, positions: list, params: dict) -> list:
        db = get_db()
        adjustments = []
        level1 = params.get("trailing_level_1_pct", 5.0)
        level2 = params.get("trailing_level_2_pct", 8.0)
        level3 = params.get("trailing_level_3_pct", 12.0)

        for p in positions:
            symbol = p.get("symbol")
            entry_price = float(p.get("avg_entry_price", 0))
            current_price = float(p.get("current_price", 0))
            pnl_pct = float(p.get("unrealized_plpc", 0)) * 100

            if entry_price <= 0:
                continue

            new_stop = None
            reason = None
            if pnl_pct >= level3:
                new_stop = round(entry_price * 1.08, 2)
                reason = f"Trailing L3: P&L {pnl_pct:.1f}% > {level3}%, stop -> entry+8%"
            elif pnl_pct >= level2:
                new_stop = round(entry_price * 1.04, 2)
                reason = f"Trailing L2: P&L {pnl_pct:.1f}% > {level2}%, stop -> entry+4%"
            elif pnl_pct >= level1:
                new_stop = round(entry_price, 2)
                reason = f"Trailing L1: P&L {pnl_pct:.1f}% > {level1}%, stop -> break-even"

            if new_stop and new_stop < current_price:
                existing = await db.trailing_stops.find_one({"ticker": symbol})
                if existing and existing.get("stop_price", 0) >= new_stop:
                    continue
                result = await update_stop_loss(symbol, new_stop)
                if result:
                    await db.trailing_stops.update_one(
                        {"ticker": symbol},
                        {"$set": {"ticker": symbol, "stop_price": new_stop,
                                  "reason": reason, "updated_at": datetime.utcnow()}},
                        upsert=True
                    )
                    adjustments.append({"ticker": symbol, "new_stop": new_stop, "reason": reason})
                    print(f"  📈 {reason} -> ${new_stop}")
        return adjustments

    # ==========================================
    # Trade Sync v5 — Anti-mismatch fractional/integer
    # ==========================================
    async def _sync_closed_trades(self):
        """
        Trade Sync v5 — Robust sync di sell con anti-mismatch bug.
        
        🆕 v5 checks (previene bug del RESET):
        1. QTY mismatch: se sell qty è intera (92) e buy qty è frazionale (45.68) → skip
        2. TIMESTAMP check: sell PRIMA del buy → skip
        3. TIME GAP check: sell più di 30 giorni dopo buy → skip
        4. SIZE tolerance: qty diff > 10% → skip (safety)
        """
        db = get_db()
        synced = 0
        skipped_mismatch = 0
        
        try:
            positions = await get_positions() or []
            open_tickers = {p.get("symbol") for p in positions}

            all_orders = await get_orders(status="all", limit=200, nested=False)
            if not all_orders:
                return synced

            for order in all_orders:
                if order.get("side") != "sell" or order.get("status") != "filled":
                    continue

                ticker = order.get("symbol", "")
                filled_price = float(order.get("filled_avg_price") or 0)
                filled_qty = float(order.get("filled_qty") or order.get("qty") or 0)
                order_id = order.get("id", "")
                
                # 🆕 v5 — Extract sell timestamp
                sell_created_str = order.get("created_at", "")
                sell_date = None
                if sell_created_str:
                    try:
                        sell_date = datetime.fromisoformat(sell_created_str.replace("Z", "+00:00")).replace(tzinfo=None)
                    except:
                        pass

                if not ticker or not order_id or filled_price <= 0:
                    continue

                if ticker in open_tickers:
                    continue

                existing_oid = await db.trade_history.find_one({"order_id": order_id})
                if existing_oid:
                    continue

                buy_trade = await db.trade_history.find_one(
                    {
                        "ticker": ticker,
                        "side": "buy",
                        "sell_linked": {"$ne": True}
                    },
                    sort=[("date", -1)]
                )
                if not buy_trade:
                    continue

                entry_price = buy_trade.get("entry_price", 0)
                buy_shares = float(buy_trade.get("shares", 0))
                buy_date = buy_trade.get("date")
                
                # 🆕 v5 CHECK 1 — QTY MISMATCH (fractional vs integer)
                # Se sell è quantità intera (es. 92.0) e buy è frazionale (45.68)
                # → è una sell VECCHIA del reset, NON collegare
                sell_is_integer = (filled_qty == int(filled_qty)) and filled_qty >= 1
                buy_is_fractional = (buy_shares != int(buy_shares))
                
                if sell_is_integer and buy_is_fractional:
                    print(f"  ⏭️ SKIP {ticker}: sell qty {filled_qty} (integer) != buy qty {buy_shares:.4f} (fractional) — likely reset artifact")
                    skipped_mismatch += 1
                    continue
                
                # 🆕 v5 CHECK 2 — SIZE MISMATCH (> 10% diff)
                if buy_shares > 0:
                    qty_diff_pct = abs(filled_qty - buy_shares) / buy_shares
                    if qty_diff_pct > 0.10:  # > 10% difference
                        print(f"  ⏭️ SKIP {ticker}: qty mismatch {filled_qty} vs buy {buy_shares:.4f} ({qty_diff_pct*100:.1f}% diff)")
                        skipped_mismatch += 1
                        continue
                
                # 🆕 v5 CHECK 3 — TIMESTAMP (sell BEFORE buy)
                if sell_date and buy_date and sell_date < buy_date:
                    print(f"  ⏭️ SKIP {ticker}: sell {sell_date.date()} BEFORE buy {buy_date.date()}")
                    skipped_mismatch += 1
                    continue
                
                # 🆕 v5 CHECK 4 — TIME GAP (> 30 days)
                if sell_date and buy_date:
                    days_gap = (sell_date - buy_date).days
                    if days_gap > 30:
                        print(f"  ⏭️ SKIP {ticker}: sell {days_gap} days after buy (too old)")
                        skipped_mismatch += 1
                        continue

                # Price sanity check (esistente)
                if entry_price > 0 and abs(filled_price - entry_price) / entry_price > 0.30:
                    print(f"  ⏭️ SKIP {ticker}: price diff > 30% ({filled_price} vs {entry_price})")
                    skipped_mismatch += 1
                    continue

                # ✅ Tutti i check passati — procedi con sync
                shares = filled_qty if filled_qty > 0 else buy_shares
                buy_date = buy_date or datetime.utcnow()
                pnl_pct = round(((filled_price - entry_price) / entry_price) * 100, 2) if entry_price > 0 else 0
                pnl_dollar = round((filled_price - entry_price) * shares, 2) if entry_price > 0 else 0

                order_type = order.get("type", "")
                if order_type == "stop":
                    reason = "STOP_LOSS"
                elif order_type == "limit":
                    reason = "TAKE_PROFIT"
                elif order_type == "market":
                    reason = "MARKET_SELL"
                else:
                    reason = "TP_OR_SL"

                days_held = max(1, (datetime.utcnow() - buy_date).days) if buy_date else 1

                await db.trade_history.insert_one({
                    "ticker": ticker, "side": "sell",
                    "entry_price": entry_price,
                    "exit_price": round(filled_price, 2),
                    "shares": float(shares),
                    "pnl_pct": pnl_pct,
                    "pnl_dollar": pnl_dollar,
                    "days_held": days_held,
                    "reason": reason,
                    "setup_type": buy_trade.get("setup_type", "unknown"),
                    "sector": buy_trade.get("sector", "unknown"),
                    "rsi_at_entry": buy_trade.get("rsi_at_entry", 50),
                    "market_regime": buy_trade.get("market_regime", "UNKNOWN"),
                    "order_id": order_id,
                    "buy_order_id": buy_trade.get("order_id", ""),
                    "agent": "executor_sync",
                    "date": datetime.utcnow(),
                    "synced": True,
                    "source": "trade_sync_v5",
                })

                await db.trade_history.update_one(
                    {"_id": buy_trade["_id"]},
                    {"$set": {"sell_linked": True, "sell_order_id": order_id}}
                )

                emoji = "🟢" if pnl_pct > 0 else "🔴"
                print(f"  {emoji} SYNCED {reason} {ticker}: {pnl_pct:+.2f}% (${pnl_dollar:+.0f}) {days_held}d [oid:{order_id[:8]}]")
                synced += 1

        except Exception as e:
            print(f"  ⚠️ Trade sync error: {e}")

        if synced > 0:
            print(f"  📥 Synced {synced} closed trades from Alpaca")
        if skipped_mismatch > 0:
            print(f"  🛡️ Skipped {skipped_mismatch} sell orders (mismatch prevention)")
        return synced

    # ==========================================
    # 🆕 BUY notional flow (3-step)
    # ==========================================
    async def _execute_notional_buy(self, trade: dict, regime: str, params: dict, db) -> dict:
        """
        🆕 Esegue un BUY notional in 3 step:
        1. place_notional_buy
        2. wait_for_fill
        3. place_brackets_after_fill (SL + TP)
        
        Ritorna dict con: success, ticker, notional, filled_qty, avg_price, errors
        """
        ticker = trade["ticker"]
        notional_usd = trade.get("notional_usd", 0)
        target = trade["target_price"]
        stop = trade["stop_loss"]
        timeout = params.get("fill_timeout_sec", 15)
        
        result = {
            "success": False,
            "ticker": ticker,
            "notional_usd": notional_usd,
            "filled_qty": 0,
            "filled_avg_price": 0,
            "buy_order_id": None,
            "sl_order_id": None,
            "tp_order_id": None,
            "errors": [],
        }
        
        # ===== Step 1: BUY notional =====
        buy_result = await place_notional_buy(ticker, notional_usd)
        if not buy_result:
            result["errors"].append("Notional buy order failed")
            return result
        
        buy_order_id = buy_result.get("id", "")
        result["buy_order_id"] = buy_order_id
        
        # ===== Step 2: Wait for fill =====
        fill_info = await wait_for_fill(buy_order_id, timeout_sec=timeout)
        
        if not fill_info.get("filled"):
            result["errors"].append(f"Buy not filled: status={fill_info.get('status')}")
            print(f"  ⚠️ {ticker} BUY not filled in {timeout}s (status: {fill_info.get('status')})")
            return result
        
        filled_qty = fill_info["filled_qty"]
        filled_avg_price = fill_info["filled_avg_price"]
        result["filled_qty"] = filled_qty
        result["filled_avg_price"] = filled_avg_price
        
        print(f"  ✅ {ticker} FILLED: {filled_qty:.4f} shares @ ${filled_avg_price:.2f}")
        
        # 🆕 v3.4 — RECALC target/SL post-fill se slippage > 3%
        # Prezzo di fill può essere molto diverso da quello di Alpha (mercato mosso)
        # Se target <= filled_price O stop_loss >= filled_price → ricalcola da fill_price
        original_target = target
        original_stop = stop
        recalc_triggered = False
        
        # Recalc SL se stop >= filled_price (SL sopra entry = bug)
        if stop >= filled_avg_price:
            new_stop = round(filled_avg_price * 0.96, 2)  # -4% da fill reale
            print(f"  🔧 RECALC SL {ticker}: ${stop:.2f} -> ${new_stop:.2f} (fill was ${filled_avg_price:.2f})")
            stop = new_stop
            recalc_triggered = True
        
        # Recalc TP se target <= filled_price (TP sotto entry = bug)
        if target <= filled_avg_price:
            new_target = round(filled_avg_price * 1.08, 2)  # +8% da fill reale
            print(f"  🔧 RECALC TP {ticker}: ${target:.2f} -> ${new_target:.2f} (fill was ${filled_avg_price:.2f})")
            target = new_target
            recalc_triggered = True
        
        # Recalc soft se slippage entry > 3% (target/SL potrebbero non riflettere fill reale)
        # Anche se non è invalid, meglio ricalibrare
        if not recalc_triggered:
            slippage_pct = abs(filled_avg_price - target + target) / filled_avg_price * 100
            # Verifica: SL è a distanza ragionevole (2-10%)?
            sl_distance_pct = abs(filled_avg_price - stop) / filled_avg_price * 100
            tp_distance_pct = abs(target - filled_avg_price) / filled_avg_price * 100
            
            # Se SL è molto vicino (<1%) o troppo lontano (>15%) → ricalibra
            if sl_distance_pct < 1.0 or sl_distance_pct > 15.0:
                new_stop = round(filled_avg_price * 0.96, 2)  # -4%
                print(f"  🔧 RECALIBRATE SL {ticker}: ${stop:.2f} -> ${new_stop:.2f} (was {sl_distance_pct:.1f}% from fill)")
                stop = new_stop
                recalc_triggered = True
            
            # Se TP è molto vicino (<2%) → ricalibra
            if tp_distance_pct < 2.0:
                new_target = round(filled_avg_price * 1.08, 2)  # +8%
                print(f"  🔧 RECALIBRATE TP {ticker}: ${target:.2f} -> ${new_target:.2f} (was {tp_distance_pct:.1f}% from fill)")
                target = new_target
                recalc_triggered = True
        
        # Salva valori aggiornati nel result per il caller
        result["target"] = target
        result["stop_loss"] = stop
        result["recalc_triggered"] = recalc_triggered
        if recalc_triggered:
            result["original_target"] = original_target
            result["original_stop"] = original_stop
        
        # ===== Step 3: Piazza SL + TP =====
        brackets = await place_brackets_after_fill(
            symbol=ticker,
            qty=filled_qty,
            take_profit=target,
            stop_loss=stop,
        )
        
        if brackets.get("stop_loss_order"):
            result["sl_order_id"] = brackets["stop_loss_order"].get("id")
        if brackets.get("take_profit_order"):
            result["tp_order_id"] = brackets["take_profit_order"].get("id")
        if brackets.get("errors"):
            result["errors"].extend(brackets["errors"])
        
        # Success se almeno il buy è filled (SL/TP possono fallire e gestiamo dopo)
        result["success"] = True
        
        return result

    # ==========================================
    # ANALYZE — Esecuzione principale
    # ==========================================
    async def analyze(self, context: dict) -> dict:
        db = get_db()
        params = await self.get_params()
        market_ctx = context.get("market_context", {})
        approved_trades = context.get("approved_trades", [])
        approved_sells = context.get("approved_sells", [])

        sizing_mode = params.get("position_sizing_mode", "notional")  # 🆕

        # 0. CHECK MARKET STATUS
        market_status = self.is_market_open()
        allow_premarket = params.get("allow_premarket", False)

        # 0.5 SYNC CLOSED TRADES (always runs)
        synced = await self._sync_closed_trades()

        if not market_status["is_open"] and not allow_premarket:
            msg = f"Market closed ({market_status['eastern_time']}). {len(approved_trades)} buys and {len(approved_sells)} sells queued."
            print(f"⚡ Executor: {msg}")
            return {
                "executed_buys": [], "executed_sells": [], "failed_orders": [],
                "cancelled_stale": 0, "trailing_adjustments": [],
                "market_status": market_status, "message": msg,
                "synced_trades": synced,
            }

        # 1. CANCEL STALE ORDERS
        cancelled = await self._cancel_stale_orders(params)

        # 1.5 TRAILING STOP MANAGEMENT
        positions = await get_positions() or []
        
        # 🆕 v3.2 — Software SL/TP check PRIMA del trailing (posizioni potrebbero chiudersi)
        sl_tp_result = await self._check_software_sl_tp(positions, params)
        
        # Se abbiamo chiuso posizioni via SL/TP, ricarichiamo la lista aggiornata
        if sl_tp_result["triggered"]:
            positions = await get_positions() or []
        
        trailing_adjustments = await self._manage_trailing_stops(positions, params)

        # ============================================
        # 2. EXECUTE SELLS (invariato)
        # ============================================
        executed_sells = []
        failed_sells = []
        regime = market_ctx.get("market_regime", "UNKNOWN")

        for s in approved_sells:
            ticker = s["ticker"]
            try:
                position_info = None
                try:
                    pos_list = await get_positions() or []
                    for pos in pos_list:
                        if pos.get("symbol") == ticker:
                            position_info = pos
                            break
                except Exception:
                    pass

                entry_price = float(position_info.get("avg_entry_price", 0)) if position_info else s.get("entry_price", 0)
                current_price = float(position_info.get("current_price", 0)) if position_info else s.get("current_price", 0)
                shares = float(position_info.get("qty", 0)) if position_info else s.get("shares", 0)  # 🔧 float

                pnl_pct = round(((current_price - entry_price) / entry_price) * 100, 2) if entry_price > 0 else round(s.get("pnl_pct", 0), 2)
                pnl_dollar = round((current_price - entry_price) * shares, 2) if entry_price > 0 and shares > 0 else 0

                result = await close_position(ticker)
                if result is not None:
                    days_held = await self._calc_days_held(db, ticker)

                    buy_trade = await db.trade_history.find_one(
                        {"ticker": ticker, "side": "buy", "sell_linked": {"$ne": True}},
                        sort=[("date", -1)]
                    )

                    executed_sells.append({
                        "ticker": ticker, "reason": s.get("reason", ""),
                        "pnl_pct": pnl_pct, "pnl_dollar": pnl_dollar,
                        "shares": shares, "days_held": days_held,
                        "executed_at": datetime.utcnow().isoformat(),
                    })

                    sell_order_id = f"executor_direct_{ticker}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

                    await db.trade_history.insert_one({
                        "ticker": ticker, "side": "sell",
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "shares": float(shares),  # 🔧 float
                        "pnl_pct": pnl_pct,
                        "pnl_dollar": pnl_dollar,
                        "days_held": days_held,
                        "reason": s.get("reason", ""),
                        "rsi_at_entry": s.get("rsi", buy_trade.get("rsi_at_entry", 50) if buy_trade else 50),
                        "setup_type": buy_trade.get("setup_type", "unknown") if buy_trade else s.get("setup_type", "unknown"),
                        "sector": buy_trade.get("sector", "unknown") if buy_trade else s.get("sector", "unknown"),
                        "market_regime": regime,
                        "agent": "executor",
                        "order_id": sell_order_id,
                        "buy_order_id": buy_trade.get("order_id", "") if buy_trade else "",
                        "date": datetime.utcnow(),
                        "source": "executor_direct",
                    })

                    if buy_trade:
                        await db.trade_history.update_one(
                            {"_id": buy_trade["_id"]},
                            {"$set": {"sell_linked": True, "sell_order_id": sell_order_id}}
                        )

                    await db.trailing_stops.delete_one({"ticker": ticker})

                    emoji = "🟢" if pnl_pct > 0 else "🔴"
                    msg = (f"{emoji} <b>SELL {ticker}</b>\n"
                           f"Reason: {s.get('reason', '')}\n"
                           f"P&L: {pnl_pct:+.1f}% (${pnl_dollar:+.0f}) | {shares:.4f} shares\n"
                           f"Days: {days_held} | Regime: {regime}")
                    await self._send_notification(msg, params)

                    print(f"  ✅ SOLD {ticker}: {s.get('reason')} (P&L {pnl_pct:+.1f}%, ${pnl_dollar:+.0f}, {days_held}d)")
                else:
                    failed_sells.append({"ticker": ticker, "reason": "Close failed"})
            except Exception as e:
                failed_sells.append({"ticker": ticker, "reason": str(e)})

        # ============================================
        # 3. EXECUTE BUYS
        # ============================================
        executed_buys = []
        failed_buys = []

        # Open orders check (per evitare doppi buy)
        open_orders = await get_orders(status="open", limit=100)
        open_buy_tickers = set()
        if open_orders:
            for o in open_orders:
                if o.get("side") == "buy" and o.get("status") in ("new", "accepted", "pending_new"):
                    open_buy_tickers.add(o.get("symbol"))

        for t in approved_trades:
            ticker = t["ticker"]

            if ticker in open_buy_tickers:
                print(f"  ⏭ Skip {ticker}: already has open buy order")
                continue

            trade_sizing_mode = t.get("sizing_mode", sizing_mode)

            try:
                # 🆕 ============================================
                # FLUSSO NOTIONAL/FRACTIONAL
                # 🆕 ============================================
                if trade_sizing_mode == "notional":
                    notional_result = await self._execute_notional_buy(t, regime, params, db)
                    
                    if notional_result["success"]:
                        filled_qty = notional_result["filled_qty"]
                        avg_price = notional_result["filled_avg_price"]
                        # 🆕 v3.4 — Usa target/stop RICALCOLATI se sono stati aggiornati
                        target = notional_result.get("target", t["target_price"])
                        stop = notional_result.get("stop_loss", t["stop_loss"])
                        notional_usd = notional_result["notional_usd"]
                        buy_order_id = notional_result["buy_order_id"]
                        
                        executed_buys.append({
                            "ticker": ticker,
                            "sizing_mode": "notional",
                            "notional_usd": notional_usd,
                            "filled_qty": filled_qty,
                            "filled_avg_price": avg_price,
                            "target": target,
                            "stop_loss": stop,
                            "confluence": t.get("confluence", 0),
                            "setup_type": t.get("setup_type", ""),
                            "buy_order_id": buy_order_id,
                            "sl_order_id": notional_result.get("sl_order_id"),
                            "tp_order_id": notional_result.get("tp_order_id"),
                            "executed_at": datetime.utcnow().isoformat(),
                        })
                        
                        await db.trade_history.insert_one({
                            "ticker": ticker, "side": "buy",
                            "sizing_mode": "notional",
                            "notional_usd": notional_usd,
                            "entry_price": avg_price,
                            "shares": filled_qty,  # float
                            "target": target, "stop_loss": stop,
                            "confluence": t.get("confluence", 0),
                            "setup_type": t.get("setup_type", ""),
                            "sector": t.get("sector", ""),
                            "rsi_at_entry": t.get("rsi", 50),
                            "market_regime": regime,
                            "agent": "executor",
                            "order_id": buy_order_id,
                            "sl_order_id": notional_result.get("sl_order_id"),
                            "tp_order_id": notional_result.get("tp_order_id"),
                            "date": datetime.utcnow(),
                            "sell_linked": False,
                        })
                        
                        from app.services.stock_names import get_stock_name
                        stock_name = get_stock_name(ticker)
                        msg = (f"🟡 <b>BUY {ticker}</b> ({stock_name})\n"
                               f"Notional: ${notional_usd:.0f} | {filled_qty:.4f} shares @ ${avg_price:.2f}\n"
                               f"Target: ${target} | Stop: ${stop}\n"
                               f"Confluence: {t.get('confluence', 0)} | {t.get('setup_type', '')}\n"
                               f"Regime: {regime}")
                        await self._send_notification(msg, params)
                        print(f"  ✅ BUY {ticker} ({stock_name}): ${notional_usd:.0f} ({filled_qty:.4f} sh @ ${avg_price:.2f})")
                    else:
                        failed_buys.append({
                            "ticker": ticker,
                            "reason": "; ".join(notional_result.get("errors", ["Notional buy failed"]))
                        })
                
                # ============================================
                # FLUSSO LEGACY (shares intere, bracket)
                # ============================================
                else:
                    shares = t.get("shares", 0)
                    price = t["price"]
                    target = t["target_price"]
                    stop = t["stop_loss"]
                    buffer_pct = params.get("limit_price_buffer_pct", 0.5) / 100
                    limit_price = round(price * (1 + buffer_pct), 2)
                    
                    result = await place_bracket_order(
                        symbol=ticker, qty=shares,
                        limit_price=limit_price, take_profit=target, stop_loss=stop,
                    )
                    if result:
                        order_id = result.get("id", "")
                        executed_buys.append({
                            "ticker": ticker, "shares": shares,
                            "sizing_mode": "shares",
                            "limit_price": limit_price, "target": target,
                            "stop_loss": stop, "confluence": t.get("confluence", 0),
                            "setup_type": t.get("setup_type", ""),
                            "order_id": order_id,
                            "executed_at": datetime.utcnow().isoformat(),
                        })
                        await db.trade_history.insert_one({
                            "ticker": ticker, "side": "buy",
                            "sizing_mode": "shares",
                            "entry_price": price, "shares": shares,
                            "target": target, "stop_loss": stop,
                            "confluence": t.get("confluence", 0),
                            "setup_type": t.get("setup_type", ""),
                            "sector": t.get("sector", ""),
                            "rsi_at_entry": t.get("rsi", 50),
                            "market_regime": regime,
                            "agent": "executor", "order_id": order_id,
                            "date": datetime.utcnow(),
                            "sell_linked": False,
                        })
                        from app.services.stock_names import get_stock_name
                        stock_name = get_stock_name(ticker)
                        msg = (f"🟡 <b>BUY {ticker}</b> ({stock_name})\n"
                               f"Shares: {shares} @ ${limit_price}\n"
                               f"Target: ${target} | Stop: ${stop}\n"
                               f"Confluence: {t.get('confluence', 0)} | {t.get('setup_type', '')}\n"
                               f"Regime: {regime}")
                        await self._send_notification(msg, params)
                        print(f"  ✅ BUY {ticker} ({stock_name}): {shares} shares @ ${limit_price}")
                    else:
                        failed_buys.append({"ticker": ticker, "reason": "Order returned None"})
            
            except Exception as e:
                failed_buys.append({"ticker": ticker, "reason": str(e)})

        # ============================================
        # 4. SUMMARY NOTIFICATION
        # ============================================
        failed_orders = failed_sells + failed_buys
        if executed_buys or executed_sells:
            from app.services.stock_names import get_stock_name
            summary = f"<b>🤖 SwingLab Report</b>\nRegime: {regime}\n"
            if executed_buys:
                summary += f"\n<b>Buys ({len(executed_buys)}):</b>\n"
                for b in executed_buys:
                    if b.get("sizing_mode") == "notional":
                        summary += f"  {b['ticker']} ({get_stock_name(b['ticker'])}) ${b.get('notional_usd', 0):.0f}\n"
                    else:
                        summary += f"  {b['ticker']} ({get_stock_name(b['ticker'])}) x{b.get('shares', 0)}\n"
            if executed_sells:
                summary += f"\n<b>Sells ({len(executed_sells)}):</b>\n"
                for s in executed_sells:
                    e = "🟢" if s.get('pnl_pct', 0) > 0 else "🔴"
                    summary += f"  {e} {s['ticker']} ({s.get('pnl_pct',0):+.1f}%, ${s.get('pnl_dollar',0):+.0f})\n"
            if trailing_adjustments:
                summary += f"\n<b>Trailing Stops ({len(trailing_adjustments)}):</b>\n"
                for t in trailing_adjustments:
                    summary += f"  📈 {t['ticker']} stop -> ${t['new_stop']}\n"
            await self._send_notification(summary, params)

        # ============================================
        # LLM REASONING
        # ============================================
        from app.services.llm_service import llm_ask, llm_available
        executor_reasoning = None
        if llm_available():
            try:
                agents_context = ""
                try:
                    from app.agents.shared_brain import brain
                    brain_data = await brain.get_full_state()
                    macro_r = brain_data.get("market", {}).get("llm_reasoning", "")
                    risk_r = brain_data.get("approved", {}).get("risk_report", {}).get("llm_reasoning", "")
                    if macro_r:
                        agents_context += f"\nMacro: {macro_r[:100]}"
                    if risk_r:
                        agents_context += f"\nRisk: {risk_r[:100]}"
                except Exception:
                    pass

                exec_summary = (
                    f"Market: {market_status['session']} ({market_status['eastern_time']})\n"
                    f"Sizing mode: {sizing_mode}\n"
                    f"Buys executed: {len(executed_buys)} ({', '.join(b['ticker'] for b in executed_buys)})\n"
                    f"Sells executed: {len(executed_sells)} ({', '.join(s['ticker'] for s in executed_sells)})\n"
                    f"Failed: {len(failed_orders)} ({', '.join(f['ticker']+': '+f['reason'] for f in failed_orders)})\n"
                    f"Trailing stops adjusted: {len(trailing_adjustments)}\n"
                    f"Stale orders cancelled: {cancelled}\n"
                    f"Trades synced: {synced}\n"
                    f"Regime: {regime}"
                )
                executor_reasoning = llm_ask(
                    system_prompt=(
                        "Sei un execution specialist di swing trading. "
                        "Valuta le esecuzioni appena fatte in max 2 frasi in italiano. "
                        "Indica se le esecuzioni sono state ottimali e cosa migliorare. "
                        "Sii diretto, concreto, no disclaimers."
                    ),
                    user_prompt=f"Execution report:\n{exec_summary}{agents_context}",
                    max_tokens=150,
                    temperature=0.3,
                    agent_name="executor",
                )
                if executor_reasoning:
                    print(f"  🧠 Executor LLM: {executor_reasoning[:80]}...")
            except Exception as e:
                print(f"  Executor LLM error: {e}")

        # Log decision
        sw_sl_tp_count = len(sl_tp_result.get("triggered", []))
        total_sells = len(executed_sells) + sw_sl_tp_count
        
        await self.log_decision(
            decision_type="execution_complete",
            data={
                "buys": len(executed_buys),
                "sells": total_sells,
                "direct_sells": len(executed_sells),
                "software_sl_tp": sw_sl_tp_count,
                "failed": len(failed_orders),
                "cancelled_stale": cancelled,
                "trailing_adjustments": len(trailing_adjustments),
                "synced_trades": synced,
                "sizing_mode": sizing_mode,
                "market_open": market_status["is_open"],
                "regime": regime,
            },
            reasoning=(
                f"Executed {len(executed_buys)} buys, {total_sells} sells "
                f"({sw_sl_tp_count} software SL/TP), "
                f"{len(trailing_adjustments)} trailing stops, synced {synced}"
            ),
            confidence=80,
        )

        print(f"\n⚡ Executor: {len(executed_buys)} buys, {total_sells} sells "
              f"({sw_sl_tp_count} software SL/TP), "
              f"{len(trailing_adjustments)} trailing stops, {cancelled} stale cancelled, {synced} synced "
              f"[mode={sizing_mode}]")

        return {
            "executed_buys": executed_buys, "executed_sells": executed_sells,
            "failed_orders": failed_orders, "cancelled_stale": cancelled,
            "trailing_adjustments": trailing_adjustments,
            "synced_trades": synced,
            "software_sl_tp": sl_tp_result,  # 🆕 v3.2
            "market_status": market_status,
            "llm_reasoning": executor_reasoning,
        }

    # ==========================================
    # LEARN (invariato)
    # ==========================================
    async def learn(self) -> dict:
        db = get_db()
        params = await self.get_params()

        recent_buys = await db.trade_history.find({
            "side": "buy", "agent": "executor",
            "date": {"$gte": datetime.utcnow() - timedelta(days=30)},
        }).to_list(100)

        if len(recent_buys) < 3:
            return {"message": "Not enough data", "orders": len(recent_buys)}

        failed_decisions = await self._col_decisions().find({
            "type": "execution_complete",
            "created_at": {"$gte": datetime.utcnow() - timedelta(days=30)},
        }).to_list(100)

        total_failed = sum(d.get("data", {}).get("failed", 0) for d in failed_decisions)
        total_executed = sum(d.get("data", {}).get("buys", 0) + d.get("data", {}).get("sells", 0) for d in failed_decisions)
        fill_rate = (total_executed / (total_executed + total_failed) * 100) if (total_executed + total_failed) > 0 else 100

        buffer = params.get("limit_price_buffer_pct", 0.5)
        if fill_rate < 70:
            buffer = min(1.5, buffer + 0.2)
        elif fill_rate > 95 and buffer > 0.3:
            buffer = max(0.2, buffer - 0.1)

        params["limit_price_buffer_pct"] = round(buffer, 2)
        await self.save_params(params)
        await self.save_performance({"fill_rate": round(fill_rate, 1), "buffer": buffer})

        print(f"⚡ Executor LEARN: fill_rate={fill_rate:.1f}%, buffer={buffer}%")
        return {"fill_rate": round(fill_rate, 1), "buffer": buffer}
