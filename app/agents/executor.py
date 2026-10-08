from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from app.agents.base_agent import BaseAgent
from app.db.mongodb import get_db
from app.services.alpaca_trader import (
    place_bracket_order, close_position, get_orders, cancel_order,
    get_positions, update_stop_loss,
    place_notional_buy, wait_for_fill, place_brackets_after_fill,
)
from app.services.telegram_bot import send_telegram


NON_ADOPTABLE_TICKERS = {"SPY"}
TERMINAL_ORDER_STATUSES = ("canceled", "cancelled", "expired", "rejected", "done_for_day")


class Executor(BaseAgent):
    """
    ⚡ AGENTE 4: Executor v3.1
    Esegue trade, trailing stop, notifiche Telegram, cancella ordini stale.
    v3.0 — Supporto fractional/notional:
    - BUY: place_notional_buy → wait_for_fill → place_brackets_after_fill
    - SELL: close_position (gestisce fractional nativamente)
    - SL/TP separati invece di bracket (Alpaca non supporta bracket+notional)
    v3.1 — Ordini accettati ma non ancora eseguiti registrati come pending
    e riconciliati alla pipeline successiva; posizioni orfane adottate.
    Trade Sync v5 mantenuto, ora con qty float.
    Timezone dinamico con zoneinfo.
    """

    def __init__(self):
        super().__init__(name="executor", version="3.1")

    def default_params(self) -> dict:
        return {
            "limit_price_buffer_pct": 0.5,
            "stale_order_hours": 2,
            "send_telegram": True,
            "allow_premarket": False,
            "trailing_level_1_pct": 5.0,
            "trailing_level_2_pct": 8.0,
            "trailing_level_3_pct": 12.0,
            "position_sizing_mode": "notional",
            "fill_timeout_sec": 15,
            "orphan_default_stop_pct": 8.0,
            "orphan_default_target_pct": 10.0,
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
        regular_time = regular_open <= et_now <= regular_close
        extended_time = extended_open <= et_now <= extended_close
        is_regular = is_weekday and regular_time
        is_extended = is_weekday and extended_time and not regular_time
        return {
            "is_open": is_regular or is_extended,
            "is_regular": is_regular,
            "is_extended": is_extended,
            "eastern_time": et_now.strftime("%Y-%m-%d %H:%M:%S ET"),
            "is_weekday": is_weekday,
            "session": "regular" if is_regular else ("extended" if is_extended else "closed"),
        }

    async def _cancel_stale_orders(self, params: dict) -> int:
        """
        🔧 v3.1 — Cancella SOLO ordini BUY stale (entry non eseguiti).
        Non cancella mai SL/TP di posizioni aperte (che sono ordini sell con type=stop/limit).
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

            if side == "sell" and order_type in ("stop", "limit", "stop_limit", "trailing_stop"):
                skipped_sl_tp += 1
                continue

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
    # 🆕 v3.1 — Livelli e target adattivi condivisi
    # ==========================================
    @staticmethod
    def _sanitize_levels(fill_price: float, target: float, stop: float):
        recalc = False
        if stop >= fill_price:
            stop = round(fill_price * 0.96, 2)
            recalc = True
        if target <= fill_price:
            target = round(fill_price * 1.08, 2)
            recalc = True
        if not recalc:
            sl_distance_pct = abs(fill_price - stop) / fill_price * 100
            tp_distance_pct = abs(target - fill_price) / fill_price * 100
            if sl_distance_pct < 1.0 or sl_distance_pct > 15.0:
                stop = round(fill_price * 0.96, 2)
                recalc = True
            if tp_distance_pct < 2.0:
                target = round(fill_price * 1.08, 2)
                recalc = True
        return target, stop, recalc

    @staticmethod
    def _adaptive_fields(entry_price: float, target: float, stop: float) -> dict:
        sl_distance_pct = ((entry_price - stop) / entry_price * 100) if entry_price > 0 else 4.0
        target_distance_pct = ((target - entry_price) / entry_price * 100) if entry_price > 0 else 8.0
        target_distance_pct = max(2.0, min(40.0, target_distance_pct))
        sl_distance_pct = max(1.0, min(15.0, sl_distance_pct))
        return {
            "target_distance_pct": round(target_distance_pct, 2),
            "sl_distance_pct": round(sl_distance_pct, 2),
            "adaptive_t1_pct": round(target_distance_pct * 0.40, 2),
            "adaptive_t2_pct": round(target_distance_pct * 0.70, 2),
            "adaptive_t3_pct": round(target_distance_pct * 1.00, 2),
        }

    # ==========================================
    # 🆕 v3.1 — Riconciliazione pending + adozione orfani
    # ==========================================
    async def _find_last_approval(self, db, ticker: str) -> dict:
        since = datetime.utcnow() - timedelta(days=3)

        def scan(node):
            if isinstance(node, dict):
                if node.get("ticker") == ticker and (node.get("stop_loss") or node.get("target_price")):
                    return node
                for value in node.values():
                    found = scan(value)
                    if found:
                        return found
            elif isinstance(node, list):
                for value in node:
                    found = scan(value)
                    if found:
                        return found
            return None

        approval = {}
        for collection in ("agent_decisions_risk_manager", "agent_decisions_alpha_strategist"):
            try:
                cursor = db[collection].find({"created_at": {"$gte": since}}).sort("created_at", -1).limit(30)
                async for doc in cursor:
                    found = scan(doc.get("data", doc))
                    if found:
                        for key, value in found.items():
                            if key not in approval or approval.get(key) in (None, "", 0):
                                approval[key] = value
                        break
            except Exception as e:
                print(f"  ⚠️ Approval lookup {collection} {ticker}: {e}")
            if approval.get("channel") and approval.get("stop_loss"):
                break
        return approval

    async def _reconcile_pending_and_orphans(self, positions: list, params: dict) -> dict:
        db = get_db()
        report = {"confirmed": [], "cancelled": [], "still_pending": [], "adopted": []}

        pending = await db.trade_history.find(
            {"side": "buy", "fill_status": "pending", "sell_linked": {"$ne": True}}
        ).to_list(50)
        for record in pending:
            ticker = record.get("ticker")
            order_id = record.get("order_id")
            if not order_id:
                continue
            try:
                info = await wait_for_fill(order_id, timeout_sec=1)
            except Exception as e:
                print(f"  ⚠️ Pending check {ticker}: {e}")
                continue
            status = str(info.get("status") or "").lower()
            if info.get("filled"):
                fill_price = float(info.get("filled_avg_price") or 0)
                fill_qty = float(info.get("filled_qty") or 0)
                if fill_price <= 0 or fill_qty <= 0:
                    report["still_pending"].append(ticker)
                    continue
                target, stop, _ = self._sanitize_levels(
                    fill_price, float(record.get("target") or 0), float(record.get("stop_loss") or 0)
                )
                await db.trade_history.update_one(
                    {"_id": record["_id"]},
                    {"$set": {
                        "fill_status": "filled",
                        "entry_price": fill_price,
                        "shares": fill_qty,
                        "target": target,
                        "stop_loss": stop,
                        "filled_confirmed_at": datetime.utcnow(),
                        **self._adaptive_fields(fill_price, target, stop),
                    }},
                )
                report["confirmed"].append(ticker)
                print(f"  ✅ PENDING CONFIRMED {ticker}: {fill_qty:.4f} sh @ ${fill_price:.2f} | SL ${stop} TP ${target}")
            elif status in TERMINAL_ORDER_STATUSES:
                await db.trade_history.delete_one({"_id": record["_id"]})
                report["cancelled"].append(ticker)
                print(f"  🗑️ PENDING REMOVED {ticker}: ordine {status}")
            else:
                report["still_pending"].append(ticker)

        if not positions:
            return report

        filled_buys = None
        for pos in positions:
            symbol = pos.get("symbol")
            if not symbol or symbol in NON_ADOPTABLE_TICKERS:
                continue
            existing = await db.trade_history.find_one(
                {"ticker": symbol, "side": "buy", "sell_linked": {"$ne": True}}
            )
            if existing:
                continue
            entry_price = float(pos.get("avg_entry_price", 0) or 0)
            qty = float(pos.get("qty", 0) or 0)
            if entry_price <= 0 or qty <= 0:
                continue
            if filled_buys is None:
                filled_buys = {}
                try:
                    orders = await get_orders(status="all", limit=200, nested=False) or []
                    for order in orders:
                        if order.get("side") == "buy" and order.get("status") == "filled":
                            sym = order.get("symbol")
                            if sym and sym not in filled_buys:
                                filled_buys[sym] = order
                except Exception as e:
                    print(f"  ⚠️ Orphan orders lookup: {e}")
            order = filled_buys.get(symbol, {})
            order_id = order.get("id") or f"adopted_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            if order.get("id") and await db.trade_history.find_one({"order_id": order_id}):
                continue
            approval = await self._find_last_approval(db, symbol)
            stop = float(approval.get("stop_loss") or 0) or entry_price * (1 - params.get("orphan_default_stop_pct", 8.0) / 100)
            target = float(approval.get("target_price") or approval.get("target") or 0) or entry_price * (1 + params.get("orphan_default_target_pct", 10.0) / 100)
            target, stop, _ = self._sanitize_levels(entry_price, round(target, 2), round(stop, 2))
            buy_date = datetime.utcnow()
            created = order.get("filled_at") or order.get("created_at")
            if created:
                try:
                    buy_date = datetime.fromisoformat(str(created).replace("Z", "+00:00")).replace(tzinfo=None)
                except (ValueError, TypeError):
                    pass
            channel = approval.get("channel") or "ALPHA"
            await db.trade_history.insert_one({
                "ticker": symbol, "side": "buy",
                "sizing_mode": "notional",
                "notional_usd": round(entry_price * qty, 2),
                "entry_price": entry_price,
                "shares": qty,
                "target": target, "stop_loss": stop,
                **self._adaptive_fields(entry_price, target, stop),
                "confluence": approval.get("confluence", 0),
                "setup_type": approval.get("setup_type", "adopted"),
                "channel": channel,
                "max_plan_status": approval.get("max_plan_status"),
                "sector": approval.get("sector", ""),
                "rsi_at_entry": approval.get("rsi", 50),
                "market_regime": approval.get("market_regime", "UNKNOWN"),
                "agent": "executor_adopt",
                "order_id": order_id,
                "fill_status": "filled",
                "adopted": True,
                "adopted_from_approval": bool(approval),
                "date": buy_date,
                "sell_linked": False,
            })
            report["adopted"].append(symbol)
            print(f"  🧷 ADOPTED ORPHAN {symbol} [{channel}]: {qty:.4f} sh @ ${entry_price:.2f} | SL ${stop} TP ${target}"
                  f"{'' if approval else ' (livelli di default)'}")

        return report

    # ==========================================
    # 🆕 v3.2 — SOFTWARE SL/TP (per fractional shares)
    # ==========================================
    async def _check_software_sl_tp(self, positions: list, params: dict) -> dict:
        """
        🆕 v3.2 — Controlla software SL/TP per posizioni con fractional shares.
        Alpaca NON supporta SL/TP nativi con fractional shares (limit/stop + GTC).
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

            buy_trade = await db.trade_history.find_one(
                {
                    "ticker": symbol,
                    "side": "buy",
                    "sell_linked": {"$ne": True}
                },
                sort=[("date", -1)]
            )

            if not buy_trade:
                continue

            result["checked"] += 1

            stop_loss = buy_trade.get("stop_loss", 0)
            target = buy_trade.get("target", 0)

            trailing = await db.trailing_stops.find_one({"ticker": symbol})
            if trailing:
                trailing_stop = trailing.get("stop_price", 0)
                if trailing_stop > stop_loss:
                    stop_loss = trailing_stop

            if stop_loss > 0:
                tolerance = entry_price * 0.001
                is_profit_lock = bool(trailing) and (
                    trailing.get("apm_managed")
                    or float(trailing.get("floor_price", 0) or 0) > 0
                    or buy_trade.get("partial_scaled_out")
                )
                if stop_loss > entry_price + tolerance:
                    if is_profit_lock:
                        lock_pct = (stop_loss / entry_price - 1) * 100
                        print(f"  🔒 PROFIT-LOCK SL {symbol}: ${stop_loss:.2f} (+{lock_pct:.1f}% sopra entry, post scale-out)")
                    else:
                        safe_stop = round(entry_price * 0.96, 2)
                        print(f"  🔧 FIXED INVALID SL {symbol}: ${stop_loss:.2f} > entry ${entry_price:.2f} → ${safe_stop:.2f}")
                        stop_loss = safe_stop
                        await db.trade_history.update_one(
                            {"_id": buy_trade["_id"]},
                            {"$set": {"stop_loss": safe_stop}},
                        )
                elif abs(stop_loss - entry_price) <= tolerance:
                    print(f"  🛡️ BREAK-EVEN SL for {symbol}: ${stop_loss:.2f} (post scale-out)")

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
                            "channel": buy_trade.get("channel", "ALPHA"),
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

                        await db.trade_history.update_one(
                            {"_id": buy_trade["_id"]},
                            {"$set": {"sell_linked": True, "sell_order_id": sell_order_id}}
                        )

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
                        continue
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
                            "channel": buy_trade.get("channel", "ALPHA"),
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
            apm_trailing = await db.trailing_stops.find_one({"ticker": symbol})
            if apm_trailing and apm_trailing.get("apm_managed"):
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

                sell_created_str = order.get("created_at", "")
                sell_date = None
                if sell_created_str:
                    try:
                        sell_date = datetime.fromisoformat(sell_created_str.replace("Z", "+00:00")).replace(tzinfo=None)
                    except Exception:
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

                sell_is_integer = (filled_qty == int(filled_qty)) and filled_qty >= 1
                buy_is_fractional = (buy_shares != int(buy_shares))

                if sell_is_integer and buy_is_fractional:
                    print(f"  ⏭️ SKIP {ticker}: sell qty {filled_qty} (integer) != buy qty {buy_shares:.4f} (fractional) — likely reset artifact")
                    skipped_mismatch += 1
                    continue

                if buy_shares > 0:
                    qty_diff_pct = abs(filled_qty - buy_shares) / buy_shares
                    if qty_diff_pct > 0.10:
                        print(f"  ⏭️ SKIP {ticker}: qty mismatch {filled_qty} vs buy {buy_shares:.4f} ({qty_diff_pct*100:.1f}% diff)")
                        skipped_mismatch += 1
                        continue

                if sell_date and buy_date and sell_date < buy_date:
                    print(f"  ⏭️ SKIP {ticker}: sell {sell_date.date()} BEFORE buy {buy_date.date()}")
                    skipped_mismatch += 1
                    continue

                if sell_date and buy_date:
                    days_gap = (sell_date - buy_date).days
                    if days_gap > 30:
                        print(f"  ⏭️ SKIP {ticker}: sell {days_gap} days after buy (too old)")
                        skipped_mismatch += 1
                        continue
                if entry_price > 0 and abs(filled_price - entry_price) / entry_price > 0.30:
                    print(f"  ⏭️ SKIP {ticker}: price diff > 30% ({filled_price} vs {entry_price})")
                    skipped_mismatch += 1
                    continue
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
                    "channel": buy_trade.get("channel", "ALPHA"),
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
    # BUY notional flow (3-step)
    # ==========================================
    async def _execute_notional_buy(self, trade: dict, regime: str, params: dict, db) -> dict:
        """
        Esegue un BUY notional in 3 step:
        1. place_notional_buy
        2. wait_for_fill (se non eseguito entro il timeout → pending, non fallito)
        3. place_brackets_after_fill (SL + TP)
        """
        ticker = trade["ticker"]
        notional_usd = trade.get("notional_usd", 0)
        target = trade["target_price"]
        stop = trade["stop_loss"]
        timeout = params.get("fill_timeout_sec", 15)

        result = {
            "success": False,
            "pending": False,
            "ticker": ticker,
            "notional_usd": notional_usd,
            "filled_qty": 0,
            "filled_avg_price": 0,
            "buy_order_id": None,
            "sl_order_id": None,
            "tp_order_id": None,
            "errors": [],
        }

        buy_result = await place_notional_buy(ticker, notional_usd)
        if not buy_result:
            result["errors"].append("Notional buy order failed")
            return result

        buy_order_id = buy_result.get("id", "")
        result["buy_order_id"] = buy_order_id

        fill_info = await wait_for_fill(buy_order_id, timeout_sec=timeout)

        if not fill_info.get("filled"):
            status = str(fill_info.get("status") or "").lower()
            if status in TERMINAL_ORDER_STATUSES:
                result["errors"].append(f"Buy not filled: status={status}")
                print(f"  ⚠️ {ticker} BUY {status}")
                return result
            result["pending"] = True
            result["target"] = target
            result["stop_loss"] = stop
            print(f"  ⏳ {ticker} BUY accettato, non ancora eseguito in {timeout}s (status: {status}) → registrato come pending")
            return result

        filled_qty = fill_info["filled_qty"]
        filled_avg_price = fill_info["filled_avg_price"]
        result["filled_qty"] = filled_qty
        result["filled_avg_price"] = filled_avg_price

        print(f"  ✅ {ticker} FILLED: {filled_qty:.4f} shares @ ${filled_avg_price:.2f}")

        original_target = target
        original_stop = stop
        target, stop, recalc_triggered = self._sanitize_levels(filled_avg_price, target, stop)
        if recalc_triggered:
            print(f"  🔧 RECALC {ticker}: SL ${original_stop:.2f} -> ${stop:.2f}, TP ${original_target:.2f} -> ${target:.2f} (fill ${filled_avg_price:.2f})")

        result["target"] = target
        result["stop_loss"] = stop
        result["recalc_triggered"] = recalc_triggered
        if recalc_triggered:
            result["original_target"] = original_target
            result["original_stop"] = original_stop

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
        sizing_mode = params.get("position_sizing_mode", "notional")
        market_status = self.is_market_open()
        allow_premarket = params.get("allow_premarket", False)
        synced = await self._sync_closed_trades()
        can_trade = market_status["is_regular"] or (allow_premarket and market_status["is_open"])
        if not can_trade:
            msg = (
                f"Market {market_status['session']} ({market_status['eastern_time']}). "
                f"{len(approved_trades)} buy e {len(approved_sells)} sell approvati ma non eseguiti; "
                "saranno rivalutati alla prossima pipeline."
            )
            print(f"⚡ Executor: {msg}")
            return {
                "executed_buys": [], "executed_sells": [], "failed_orders": [],
                "cancelled_stale": 0, "trailing_adjustments": [],
                "market_status": market_status, "message": msg,
                "synced_trades": synced,
                "deferred_buys": [trade.get("ticker") for trade in approved_trades],
                "deferred_sells": [trade.get("ticker") for trade in approved_sells],
                "requires_revalidation": True,
            }
        cancelled = await self._cancel_stale_orders(params)
        positions = await get_positions() or []

        reconcile_report = await self._reconcile_pending_and_orphans(positions, params)

        sl_tp_result = await self._check_software_sl_tp(positions, params)

        if sl_tp_result["triggered"]:
            positions = await get_positions() or []

        trailing_adjustments = await self._manage_trailing_stops(positions, params)

        # ============================================
        # 2. EXECUTE SELLS
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
                shares = float(position_info.get("qty", 0)) if position_info else s.get("shares", 0)
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
                        "shares": float(shares),
                        "pnl_pct": pnl_pct,
                        "pnl_dollar": pnl_dollar,
                        "days_held": days_held,
                        "reason": s.get("reason", ""),
                        "rsi_at_entry": s.get("rsi", buy_trade.get("rsi_at_entry", 50) if buy_trade else 50),
                        "setup_type": buy_trade.get("setup_type", "unknown") if buy_trade else s.get("setup_type", "unknown"),
                        "channel": buy_trade.get("channel", "ALPHA") if buy_trade else s.get("channel", "ALPHA"),
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
        pending_buys = []
        failed_buys = []
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
                if trade_sizing_mode == "notional":
                    notional_result = await self._execute_notional_buy(t, regime, params, db)

                    if notional_result["success"] or notional_result.get("pending"):
                        is_pending = bool(notional_result.get("pending"))
                        filled_qty = notional_result["filled_qty"]
                        avg_price = notional_result["filled_avg_price"] or float(t.get("price") or 0)
                        target = notional_result.get("target", t["target_price"])
                        stop = notional_result.get("stop_loss", t["stop_loss"])
                        notional_usd = notional_result["notional_usd"]
                        buy_order_id = notional_result["buy_order_id"]

                        record = {
                            "ticker": ticker, "side": "buy",
                            "sizing_mode": "notional",
                            "notional_usd": notional_usd,
                            "entry_price": avg_price,
                            "shares": filled_qty,
                            "target": target, "stop_loss": stop,
                            **self._adaptive_fields(avg_price, target, stop),
                            "confluence": t.get("confluence", 0),
                            "setup_type": t.get("setup_type", ""),
                            "channel": t.get("channel", "ALPHA"),
                            "max_plan_status": t.get("max_plan_status"),
                            "sector": t.get("sector", ""),
                            "rsi_at_entry": t.get("rsi", 50),
                            "market_regime": regime,
                            "agent": "executor",
                            "order_id": buy_order_id,
                            "sl_order_id": notional_result.get("sl_order_id"),
                            "tp_order_id": notional_result.get("tp_order_id"),
                            "fill_status": "pending" if is_pending else "filled",
                            "date": datetime.utcnow(),
                            "sell_linked": False,
                        }
                        await db.trade_history.insert_one(record)

                        if is_pending:
                            pending_buys.append({
                                "ticker": ticker, "notional_usd": notional_usd,
                                "buy_order_id": buy_order_id, "channel": t.get("channel", "ALPHA"),
                            })
                            continue

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
                            "channel": t.get("channel", "ALPHA"),
                            "buy_order_id": buy_order_id,
                            "sl_order_id": notional_result.get("sl_order_id"),
                            "tp_order_id": notional_result.get("tp_order_id"),
                            "executed_at": datetime.utcnow().isoformat(),
                        })

                        print(f"  🎯 {ticker} adaptive: T1=+{record['adaptive_t1_pct']}% T2=+{record['adaptive_t2_pct']}% "
                              f"T3=+{record['adaptive_t3_pct']}% (target=+{record['target_distance_pct']:.1f}%)")

                        from app.services.stock_names import get_stock_name
                        stock_name = get_stock_name(ticker)
                        msg = (f"🟡 <b>BUY {ticker}</b> ({stock_name})\n"
                               f"Notional: ${notional_usd:.0f} | {filled_qty:.4f} shares @ ${avg_price:.2f}\n"
                               f"Target: ${target} | Stop: ${stop}\n"
                               f"Canale: {t.get('channel', 'ALPHA')} | {t.get('setup_type', '')}\n"
                               f"Regime: {regime}")
                        await self._send_notification(msg, params)
                        print(f"  ✅ BUY {ticker} ({stock_name}): ${notional_usd:.0f} ({filled_qty:.4f} sh @ ${avg_price:.2f})")
                    else:
                        failed_buys.append({
                            "ticker": ticker,
                            "reason": "; ".join(notional_result.get("errors", ["Notional buy failed"]))
                        })

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
                            "channel": t.get("channel", "ALPHA"),
                            "max_plan_status": t.get("max_plan_status"),
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
        if executed_buys or executed_sells or pending_buys:
            from app.services.stock_names import get_stock_name
            summary = f"<b>🤖 SwingLab Report</b>\nRegime: {regime}\n"
            if executed_buys:
                summary += f"\n<b>Buys ({len(executed_buys)}):</b>\n"
                for b in executed_buys:
                    if b.get("sizing_mode") == "notional":
                        summary += f"  {b['ticker']} ({get_stock_name(b['ticker'])}) ${b.get('notional_usd', 0):.0f}\n"
                    else:
                        summary += f"  {b['ticker']} ({get_stock_name(b['ticker'])}) x{b.get('shares', 0)}\n"
            if pending_buys:
                summary += f"\n<b>Buy in attesa di esecuzione ({len(pending_buys)}):</b>\n"
                for b in pending_buys:
                    summary += f"  ⏳ {b['ticker']} ${b.get('notional_usd', 0):.0f}\n"
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
                    f"Buys pending fill: {len(pending_buys)} ({', '.join(b['ticker'] for b in pending_buys)})\n"
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

        sw_sl_tp_count = len(sl_tp_result.get("triggered", []))
        total_sells = len(executed_sells) + sw_sl_tp_count

        await self.log_decision(
            decision_type="execution_complete",
            data={
                "buys": len(executed_buys),
                "pending_buys": len(pending_buys),
                "sells": total_sells,
                "direct_sells": len(executed_sells),
                "software_sl_tp": sw_sl_tp_count,
                "failed": len(failed_orders),
                "cancelled_stale": cancelled,
                "trailing_adjustments": len(trailing_adjustments),
                "synced_trades": synced,
                "reconcile": reconcile_report,
                "sizing_mode": sizing_mode,
                "market_open": market_status["is_open"],
                "regime": regime,
            },
            reasoning=(
                f"Executed {len(executed_buys)} buys ({len(pending_buys)} pending), {total_sells} sells "
                f"({sw_sl_tp_count} software SL/TP), "
                f"{len(trailing_adjustments)} trailing stops, synced {synced}"
            ),
            confidence=80,
        )
        print(f"\n⚡ Executor: {len(executed_buys)} buys, {len(pending_buys)} pending, {total_sells} sells "
              f"({sw_sl_tp_count} software SL/TP), "
              f"{len(trailing_adjustments)} trailing stops, {cancelled} stale cancelled, {synced} synced, "
              f"{len(reconcile_report['confirmed'])} confermati, {len(reconcile_report['adopted'])} adottati "
              f"[mode={sizing_mode}]")
        return {
            "executed_buys": executed_buys, "executed_sells": executed_sells,
            "pending_buys": pending_buys,
            "failed_orders": failed_orders, "cancelled_stale": cancelled,
            "trailing_adjustments": trailing_adjustments,
            "synced_trades": synced,
            "reconcile": reconcile_report,
            "software_sl_tp": sl_tp_result,
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
