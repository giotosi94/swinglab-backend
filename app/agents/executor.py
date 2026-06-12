from datetime import datetime, timedelta
from app.agents.base_agent import BaseAgent
from app.db.mongodb import get_db
from app.services.alpaca_trader import (
    place_bracket_order, close_position, get_orders, cancel_order,
    get_positions, update_stop_loss
)
from app.services.telegram_bot import send_telegram


class Executor(BaseAgent):
    """
    ⚡ AGENTE 4: Executor
    Esegue trade, trailing stop, notifiche Telegram, cancella ordini stale.
    """

    def __init__(self):
        super().__init__(name="executor", version="1.1")

    def default_params(self) -> dict:
        return {
            "limit_price_buffer_pct": 0.5,
            "stale_order_hours": 24,
            "send_telegram": True,
            "allow_premarket": False,
            # Trailing stop levels
            "trailing_level_1_pct": 5.0,   # P&L > 5% -> stop at break-even
            "trailing_level_2_pct": 8.0,   # P&L > 8% -> stop at entry + 4%
            "trailing_level_3_pct": 12.0,  # P&L > 12% -> stop at entry + 8%
        }

    @staticmethod
    def is_market_open() -> dict:
        utc_now = datetime.utcnow()
        et_offset = timedelta(hours=-4)
        et_now = utc_now + et_offset
        is_weekday = et_now.weekday() < 5

        # Extended hours: 4:00 AM - 8:00 PM ET
        extended_open = et_now.replace(hour=4, minute=0, second=0, microsecond=0)
        extended_close = et_now.replace(hour=20, minute=0, second=0, microsecond=0)

        # Regular hours: 9:30 AM - 4:00 PM ET
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
        stale_hours = params.get("stale_order_hours", 24)
        cutoff = datetime.utcnow() - timedelta(hours=stale_hours)
        cancelled = 0
        orders = await get_orders(status="open", limit=50)
        if not orders:
            return 0
        for order in orders:
            created = order.get("created_at", "")
            if created:
                try:
                    order_time = datetime.fromisoformat(created.replace("Z", "+00:00"))
                    if order_time.replace(tzinfo=None) < cutoff:
                        result = await cancel_order(order["id"])
                        if result is not None:
                            cancelled += 1
                            print(f"  Cancelled stale order: {order.get('symbol')} ({order.get('id')[:8]}...)")
                except (ValueError, TypeError):
                    pass
        return cancelled

    async def _send_notification(self, message: str, params: dict):
        if params.get("send_telegram", True):
            await send_telegram(message)

    async def _calc_days_held(self, db, ticker: str) -> int:
        """Calcola i giorni di holding cercando il BUY corrispondente."""
        buy_trade = await db.trade_history.find_one(
            {"ticker": ticker, "side": "buy"},
            sort=[("date", -1)]
        )
        if buy_trade and buy_trade.get("date"):
            days = (datetime.utcnow() - buy_trade["date"]).days
            return max(days, 1)
        return 0

    async def _manage_trailing_stops(self, positions: list, params: dict) -> list:
        """Gestisce i trailing stop per le posizioni aperte."""
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

            # Determine new stop level
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
                # Check if we already adjusted this stop
                existing = await db.trailing_stops.find_one({"ticker": symbol})
                if existing and existing.get("stop_price", 0) >= new_stop:
                    continue  # Already at a higher stop, skip

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

    async def _sync_closed_trades(self):
        """
        Sincronizza i trade chiusi confrontando BUY in trade_history
        con le posizioni attuali su Alpaca.
        Se un BUY esiste ma la posizione no → è stato chiuso (TP/SL).
        """
        db = get_db()
        synced = 0

        try:
            positions = await get_positions() or []
            open_tickers = {p.get("symbol") for p in positions}

            buy_trades = await db.trade_history.find({"side": "buy"}).to_list(500)

            for buy in buy_trades:
                ticker = buy.get("ticker", "")
                if not ticker:
                    continue

                if ticker in open_tickers:
                    continue

                existing_sell = await db.trade_history.find_one({
                    "ticker": ticker, "side": "sell",
                })
                if existing_sell:
                    continue

                orders = await get_orders(status="all", limit=100, nested=False)
                exit_price = 0
                reason = "TP_OR_SL"

                if orders:
                    for o in orders:
                        if (o.get("symbol") == ticker and
                            o.get("side") == "sell" and
                            o.get("status") == "filled" and
                            o.get("filled_avg_price")):
                            exit_price = float(o["filled_avg_price"])
                            if o.get("type") == "stop":
                                reason = "STOP_LOSS"
                            elif o.get("type") == "limit":
                                reason = "TAKE_PROFIT"
                            break

                if exit_price <= 0:
                    from app.services.alpaca_trader import get_latest_price
                    exit_price = await get_latest_price(ticker) or 0

                entry_price = buy.get("entry_price", 0)
                shares = buy.get("shares", 0)
                buy_date = buy.get("date", datetime.utcnow())

                if entry_price > 0 and exit_price > 0:
                    pnl_pct = round(((exit_price - entry_price) / entry_price) * 100, 2)
                    pnl_dollar = round(pnl_pct / 100 * entry_price * shares, 2)
                else:
                    pnl_pct = 0
                    pnl_dollar = 0

                days_held = max(1, (datetime.utcnow() - buy_date).days) if buy_date else 1

                trade_doc = {
                    "ticker": ticker, "side": "sell",
                    "entry_price": entry_price, "exit_price": round(exit_price, 2),
                    "shares": shares, "pnl_pct": pnl_pct, "pnl_dollar": pnl_dollar,
                    "days_held": days_held, "reason": reason,
                    "setup_type": buy.get("setup_type", "unknown"),
                    "sector": buy.get("sector", "unknown"),
                    "rsi_at_entry": buy.get("rsi_at_entry", 50),
                    "market_regime": buy.get("market_regime", "UNKNOWN"),
                    "agent": "executor_sync",
                    "date": datetime.utcnow(), "synced": True,
                }

                await db.trade_history.insert_one(trade_doc)
                await db.trailing_stops.delete_one({"ticker": ticker})

                emoji = "🟢" if pnl_pct > 0 else "🔴"
                print(f"  {emoji} SYNCED {reason} {ticker}: {pnl_pct:+.2f}% (${pnl_dollar:+.0f}) {days_held}d")

                msg = (f"{emoji} <b>SYNCED {reason} {ticker}</b>\n"
                       f"Entry: ${entry_price} -> Exit: ${exit_price}\n"
                       f"P&L: {pnl_pct:+.2f}% (${pnl_dollar:+.0f})\n"
                       f"Days: {days_held} | Setup: {buy.get('setup_type', '')}")
                await self._send_notification(msg, await self.get_params())

                synced += 1

        except Exception as e:
            print(f"  ⚠️ Trade sync error: {e}")

        if synced > 0:
            print(f"  📥 Synced {synced} closed trades from Alpaca")

        return synced
    
    async def analyze(self, context: dict) -> dict:
        db = get_db()
        params = await self.get_params()
        market_ctx = context.get("market_context", {})
        approved_trades = context.get("approved_trades", [])
        approved_sells = context.get("approved_sells", [])

        # 0. CHECK MARKET STATUS
        market_status = self.is_market_open()
        allow_premarket = params.get("allow_premarket", False)

        # 0.5 SYNC CLOSED TRADES (always runs, even when market closed)
        synced = await self._sync_closed_trades()

        if not market_status["is_open"] and not allow_premarket:
            msg = f"Market closed ({market_status['eastern_time']}). {len(approved_trades)} buys and {len(approved_sells)} sells queued."
            print(f"⚡ Executor: {msg}")
            return {
                "executed_buys": [], "executed_sells": [], "failed_orders": [],
                "cancelled_stale": 0, "trailing_adjustments": [],
                "market_status": market_status, "message": msg,
            }
             
        # 1. CANCEL STALE ORDERS
        cancelled = await self._cancel_stale_orders(params)

        # 1.5 TRAILING STOP MANAGEMENT
        positions = await get_positions() or []
        trailing_adjustments = await self._manage_trailing_stops(positions, params)

        # 2. EXECUTE SELLS
        executed_sells = []
        failed_sells = []
        regime = market_ctx.get("market_regime", "UNKNOWN")

        for s in approved_sells:
            ticker = s["ticker"]
            try:
                result = await close_position(ticker)
                if result is not None:
                    days_held = await self._calc_days_held(db, ticker)

                    executed_sells.append({
                        "ticker": ticker, "reason": s.get("reason", ""),
                        "pnl_pct": s.get("pnl_pct", 0), "days_held": days_held,
                        "executed_at": datetime.utcnow().isoformat(),
                    })

                    await db.trade_history.insert_one({
                        "ticker": ticker, "side": "sell",
                        "entry_price": s.get("entry_price", 0),
                        "exit_price": s.get("current_price", 0),
                        "pnl_pct": round(s.get("pnl_pct", 0), 2),
                        "days_held": days_held,
                        "reason": s.get("reason", ""),
                        "rsi_at_entry": s.get("rsi", 50),
                        "setup_type": s.get("setup_type", "unknown"),
                        "sector": s.get("sector", "unknown"),
                        "market_regime": regime,
                        "agent": "executor",
                        "date": datetime.utcnow(),
                    })

                    # Clean up trailing stop record
                    await db.trailing_stops.delete_one({"ticker": ticker})

                    emoji = "🟢" if s.get("pnl_pct", 0) > 0 else "🔴"
                    msg = (f"{emoji} <b>SELL {ticker}</b>\n"
                           f"Reason: {s.get('reason', '')}\n"
                           f"P&L: {s.get('pnl_pct', 0):.1f}% | Days: {days_held}\n"
                           f"Regime: {regime}")
                    await self._send_notification(msg, params)
                    print(f"  ✅ SOLD {ticker}: {s.get('reason')} (P&L {s.get('pnl_pct', 0):.1f}%, {days_held}d)")
                else:
                    failed_sells.append({"ticker": ticker, "reason": "Close failed"})
            except Exception as e:
                failed_sells.append({"ticker": ticker, "reason": str(e)})

        # 3. EXECUTE BUYS
        executed_buys = []
        failed_buys = []
        buffer_pct = params.get("limit_price_buffer_pct", 0.5) / 100

        # Check existing open orders to avoid duplicates
        open_orders = await get_orders(status="open", limit=100)
        open_buy_tickers = set()
        if open_orders:
            for o in open_orders:
                if o.get("side") == "buy" and o.get("status") in ("new", "accepted", "pending_new"):
                    open_buy_tickers.add(o.get("symbol"))

        for t in approved_trades:
            ticker = t["ticker"]

            # Skip if already has open buy order
            if ticker in open_buy_tickers:
                print(f"  ⏭ Skip {ticker}: already has open buy order")
                continue

            shares = t["shares"]
            price = t["price"]
            target = t["target_price"]
            stop = t["stop_loss"]
            limit_price = round(price * (1 + buffer_pct), 2)

            try:
                result = await place_bracket_order(
                    symbol=ticker, qty=shares,
                    limit_price=limit_price, take_profit=target, stop_loss=stop,
                )
                if result:
                    order_id = result.get("id", "")
                    executed_buys.append({
                        "ticker": ticker, "shares": shares,
                        "limit_price": limit_price, "target": target,
                        "stop_loss": stop, "confluence": t.get("confluence", 0),
                        "setup_type": t.get("setup_type", ""),
                        "order_id": order_id,
                        "executed_at": datetime.utcnow().isoformat(),
                    })

                    await db.trade_history.insert_one({
                        "ticker": ticker, "side": "buy",
                        "entry_price": price, "shares": shares,
                        "target": target, "stop_loss": stop,
                        "confluence": t.get("confluence", 0),
                        "setup_type": t.get("setup_type", ""),
                        "sector": t.get("sector", ""),
                        "rsi_at_entry": t.get("rsi", 50),
                        "market_regime": regime,
                        "agent": "executor", "order_id": order_id,
                        "date": datetime.utcnow(),
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

        # 4. SUMMARY NOTIFICATION
        failed_orders = failed_sells + failed_buys
        if executed_buys or executed_sells:
            from app.services.stock_names import get_stock_name
            summary = f"<b>🤖 SwingLab Report</b>\nRegime: {regime}\n"
            if executed_buys:
                summary += f"\n<b>Buys ({len(executed_buys)}):</b>\n"
                for b in executed_buys:
                    summary += f"  {b['ticker']} ({get_stock_name(b['ticker'])}) x{b['shares']}\n"
            if executed_sells:
                summary += f"\n<b>Sells ({len(executed_sells)}):</b>\n"
                for s in executed_sells:
                    e = "🟢" if s.get('pnl_pct', 0) > 0 else "🔴"
                    summary += f"  {e} {s['ticker']} ({s.get('pnl_pct',0):+.1f}%, {s.get('days_held',0)}d)\n"
            if trailing_adjustments:
                summary += f"\n<b>Trailing Stops ({len(trailing_adjustments)}):</b>\n"
                for t in trailing_adjustments:
                    summary += f"  📈 {t['ticker']} stop -> ${t['new_stop']}\n"
            await self._send_notification(summary, params)

        # Log decision
        await self.log_decision(
            decision_type="execution_complete",
            data={
                "buys": len(executed_buys), "sells": len(executed_sells),
                "failed": len(failed_orders), "cancelled_stale": cancelled,
                "trailing_adjustments": len(trailing_adjustments),
                "market_open": market_status["is_open"], "regime": regime,
            },
            reasoning=f"Executed {len(executed_buys)} buys, {len(executed_sells)} sells, "
                      f"{len(trailing_adjustments)} trailing stops adjusted",
            confidence=80,
        )

        print(f"\n⚡ Executor: {len(executed_buys)} buys, {len(executed_sells)} sells, "
              f"{len(trailing_adjustments)} trailing stops, {cancelled} stale cancelled")

        return {
            "executed_buys": executed_buys, "executed_sells": executed_sells,
            "failed_orders": failed_orders, "cancelled_stale": cancelled,
            "trailing_adjustments": trailing_adjustments,
            "market_status": market_status,
        }

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
