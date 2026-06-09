from datetime import datetime, timedelta
from app.agents.base_agent import BaseAgent
from app.db.mongodb import get_db
from app.services.alpaca_trader import (
    place_bracket_order, close_position, get_orders, cancel_order
)
from app.services.telegram_bot import send_telegram


class Executor(BaseAgent):
    """
    ⚡ AGENTE 4: Executor — "Il Trader"
    Esegue materialmente i trade approvati dal RiskManager.
    Si occupa di:
    - Verificare che il mercato sia aperto
    - Piazzare bracket orders (entry + TP + SL)
    - Chiudere posizioni su segnali di vendita
    - Inviare notifiche Telegram
    - Cancellare ordini stale
    - Tracciare slippage

    Input: approved_trades[], approved_sells[] (da RiskManager), market_context
    Output: execution_report
    """

    def __init__(self):
        super().__init__(name="executor", version="1.0")

    def default_params(self) -> dict:
        return {
            "limit_price_buffer_pct": 0.5,  # Paga fino a 0.5% in piu' del prezzo attuale
            "stale_order_hours": 24,         # Cancella ordini pending da piu' di 24h
            "send_telegram": True,           # Abilita notifiche Telegram
            "allow_premarket": False,        # Non tradare in pre-market
        }

    @staticmethod
    def is_market_open() -> dict:
        """
        Verifica se il mercato USA (NYSE) e' aperto.
        NYSE: Lunedi-Venerdi, 9:30-16:00 Eastern Time (UTC-4 in estate, UTC-5 in inverno).
        Approssimazione: usiamo UTC-4 (EDT, valido da Marzo a Novembre).
        """
        from datetime import timezone
        utc_now = datetime.utcnow()
        # Eastern Time (approssimazione EDT = UTC-4)
        et_offset = timedelta(hours=-4)
        et_now = utc_now + et_offset

        is_weekday = et_now.weekday() < 5  # 0=Mon, 4=Fri
        market_open = et_now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = et_now.replace(hour=16, minute=0, second=0, microsecond=0)
        is_trading_hours = market_open <= et_now <= market_close

        return {
            "is_open": is_weekday and is_trading_hours,
            "eastern_time": et_now.strftime("%Y-%m-%d %H:%M:%S ET"),
            "is_weekday": is_weekday,
            "is_trading_hours": is_trading_hours,
            "next_open": market_open.strftime("%H:%M ET") if not is_trading_hours else None,
        }

    async def _cancel_stale_orders(self, params: dict) -> int:
        """Cancella ordini pending da troppo tempo."""
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
                            print(f"  Cancelled stale order: {order.get('symbol')} "
                                  f"({order.get('id')[:8]}...)")
                except (ValueError, TypeError):
                    pass

        return cancelled

    async def _send_notification(self, message: str, params: dict):
        """Invia notifica Telegram se abilitato."""
        if params.get("send_telegram", True):
            await send_telegram(message)

    async def analyze(self, context: dict) -> dict:
        """
        Esecuzione dei trade approvati.
        context deve contenere:
        - approved_trades: trade da eseguire (da RiskManager)
        - approved_sells: posizioni da chiudere (da RiskManager)
        - market_context: contesto macro (da MacroAnalyst)
        """
        db = get_db()
        params = await self.get_params()
        market_ctx = context.get("market_context", {})
        approved_trades = context.get("approved_trades", [])
        approved_sells = context.get("approved_sells", [])

        # ============================================
        # 0. CHECK MARKET STATUS
        # ============================================
        market_status = self.is_market_open()
        allow_premarket = params.get("allow_premarket", False)

        if not market_status["is_open"] and not allow_premarket:
            msg = (f"⏰ Market closed ({market_status['eastern_time']}). "
                   f"{len(approved_trades)} buys and {len(approved_sells)} sells queued.")
            print(f"⚡ Executor: {msg}")

            return {
                "executed_buys": [],
                "executed_sells": [],
                "failed_orders": [],
                "cancelled_stale": 0,
                "market_status": market_status,
                "message": msg,
            }

        # ============================================
        # 1. CANCEL STALE ORDERS
        # ============================================
        cancelled = await self._cancel_stale_orders(params)

        # ============================================
        # 2. EXECUTE SELLS (priorita' alta)
        # ============================================
        executed_sells = []
        failed_sells = []
        regime = market_ctx.get("market_regime", "UNKNOWN")

        for s in approved_sells:
            ticker = s["ticker"]
            try:
                result = await close_position(ticker)
                if result is not None:
                    executed_sells.append({
                        "ticker": ticker,
                        "reason": s.get("reason", ""),
                        "pnl_pct": s.get("pnl_pct", 0),
                        "executed_at": datetime.utcnow().isoformat(),
                    })

                    # Log nel trade_history
                    await db.trade_history.insert_one({
                        "ticker": ticker,
                        "side": "sell",
                        "entry_price": s.get("entry_price", 0),
                        "exit_price": s.get("current_price", 0),
                        "pnl_pct": round(s.get("pnl_pct", 0), 2),
                        "reason": s.get("reason", ""),
                        "rsi_at_entry": s.get("rsi", 50),
                        "setup_type": s.get("setup_type", "unknown"),
                        "sector": s.get("sector", "unknown"),
                        "market_regime": regime,
                        "agent": "executor",
                        "date": datetime.utcnow(),
                    })

                    # Notifica Telegram
                    emoji = "🟢" if s.get("pnl_pct", 0) > 0 else "🔴"
                    msg = (f"{emoji} <b>SELL {ticker}</b>\n"
                           f"Reason: {s.get('reason', '')}\n"
                           f"P&L: {s.get('pnl_pct', 0):.1f}%\n"
                           f"Regime: {regime}")
                    await self._send_notification(msg, params)

                    print(f"  ✅ SOLD {ticker}: {s.get('reason')} (P&L {s.get('pnl_pct', 0):.1f}%)")
                else:
                    failed_sells.append({"ticker": ticker, "reason": "Close failed"})
                    print(f"  ❌ SELL FAILED {ticker}")
            except Exception as e:
                failed_sells.append({"ticker": ticker, "reason": str(e)})
                print(f"  ❌ SELL ERROR {ticker}: {e}")

        # ============================================
        # 3. EXECUTE BUYS
        # ============================================
        executed_buys = []
        failed_buys = []
        buffer_pct = params.get("limit_price_buffer_pct", 0.5) / 100

        for t in approved_trades:
            ticker = t["ticker"]
            shares = t["shares"]
            price = t["price"]
            target = t["target_price"]
            stop = t["stop_loss"]
            limit_price = round(price * (1 + buffer_pct), 2)

            try:
                result = await place_bracket_order(
                    symbol=ticker,
                    qty=shares,
                    limit_price=limit_price,
                    take_profit=target,
                    stop_loss=stop,
                )

                if result:
                    order_id = result.get("id", "")
                    executed_buys.append({
                        "ticker": ticker,
                        "shares": shares,
                        "limit_price": limit_price,
                        "target": target,
                        "stop_loss": stop,
                        "confluence": t.get("confluence", 0),
                        "setup_type": t.get("setup_type", ""),
                        "order_id": order_id,
                        "executed_at": datetime.utcnow().isoformat(),
                    })

                    # Log nel trade_history
                    await db.trade_history.insert_one({
                        "ticker": ticker,
                        "side": "buy",
                        "entry_price": price,
                        "shares": shares,
                        "target": target,
                        "stop_loss": stop,
                        "confluence": t.get("confluence", 0),
                        "setup_type": t.get("setup_type", ""),
                        "sector": t.get("sector", ""),
                        "rsi_at_entry": t.get("rsi", 50),
                        "market_regime": regime,
                        "agent": "executor",
                        "order_id": order_id,
                        "date": datetime.utcnow(),
                    })

                    # Notifica Telegram
                    msg = (f"🟡 <b>BUY {ticker}</b>\n"
                           f"Shares: {shares} @ ${limit_price}\n"
                           f"Target: ${target} | Stop: ${stop}\n"
                           f"Confluence: {t.get('confluence', 0)} | "
                           f"Setup: {t.get('setup_type', '')}\n"
                           f"Regime: {regime}")
                    await self._send_notification(msg, params)

                    print(f"  ✅ BUY {ticker}: {shares} shares @ ${limit_price}, "
                          f"TP=${target}, SL=${stop}")
                else:
                    failed_buys.append({"ticker": ticker, "reason": "Order returned None"})
                    print(f"  ❌ BUY FAILED {ticker}")
            except Exception as e:
                failed_buys.append({"ticker": ticker, "reason": str(e)})
                print(f"  ❌ BUY ERROR {ticker}: {e}")

        # ============================================
        # 4. BUILD REPORT
        # ============================================
        failed_orders = failed_sells + failed_buys

        # Notifica riassuntiva
        if executed_buys or executed_sells:
            summary_msg = f"<b>🤖 SwingLab Agent Report</b>\n\n"
            summary_msg += f"Regime: {regime}\n"
            if executed_buys:
                summary_msg += f"\n<b>Buys ({len(executed_buys)}):</b>\n"
                for b in executed_buys:
                    summary_msg += f"  {b['ticker']} x{b['shares']}\n"
            if executed_sells:
                summary_msg += f"\n<b>Sells ({len(executed_sells)}):</b>\n"
                for s in executed_sells:
                    pnl = s.get('pnl_pct', 0)
                    emoji = "🟢" if pnl > 0 else "🔴"
                    summary_msg += f"  {emoji} {s['ticker']} ({pnl:+.1f}%)\n"
            if failed_orders:
                summary_msg += f"\n⚠️ Failed: {len(failed_orders)}\n"
            await self._send_notification(summary_msg, params)

        # Log decision
        await self.log_decision(
            decision_type="execution_complete",
            data={
                "buys": len(executed_buys),
                "sells": len(executed_sells),
                "failed": len(failed_orders),
                "cancelled_stale": cancelled,
                "market_open": market_status["is_open"],
                "regime": regime,
            },
            reasoning=f"Executed {len(executed_buys)} buys, {len(executed_sells)} sells, "
                      f"{len(failed_orders)} failed, {cancelled} stale cancelled",
            confidence=80,
        )

        print(f"\n⚡ Executor: {len(executed_buys)} buys, {len(executed_sells)} sells, "
              f"{len(failed_orders)} failed, {cancelled} stale cancelled")

        return {
            "executed_buys": executed_buys,
            "executed_sells": executed_sells,
            "failed_orders": failed_orders,
            "cancelled_stale": cancelled,
            "market_status": market_status,
        }

    async def learn(self) -> dict:
        """
        Learning loop dell'Executor.
        Analizza:
        - Slippage (prezzo eseguito vs prezzo atteso)
        - Tempo degli ordini (quali orari funzionano meglio)
        - Tasso di riempimento degli ordini
        """
        db = get_db()
        params = await self.get_params()

        # Analizza ordini recenti
        recent_buys = await db.trade_history.find({
            "side": "buy",
            "agent": "executor",
            "date": {"$gte": datetime.utcnow() - timedelta(days=30)},
        }).to_list(100)

        if len(recent_buys) < 3:
            return {"message": "Not enough data to learn", "orders": len(recent_buys)}

        # Analizza ordini falliti
        failed_decisions = await self._col_decisions().find({
            "type": "execution_complete",
            "created_at": {"$gte": datetime.utcnow() - timedelta(days=30)},
        }).to_list(100)

        total_failed = sum(d.get("data", {}).get("failed", 0) for d in failed_decisions)
        total_executed = sum(
            d.get("data", {}).get("buys", 0) + d.get("data", {}).get("sells", 0)
            for d in failed_decisions
        )

        fill_rate = (total_executed / (total_executed + total_failed) * 100
                     ) if (total_executed + total_failed) > 0 else 100

        # Se fill rate basso, aumenta il buffer sul limit price
        buffer = params.get("limit_price_buffer_pct", 0.5)
        if fill_rate < 70:
            buffer = min(1.5, buffer + 0.2)
        elif fill_rate > 95 and buffer > 0.3:
            buffer = max(0.2, buffer - 0.1)

        params["limit_price_buffer_pct"] = round(buffer, 2)
        await self.save_params(params)

        learn_result = {
            "fill_rate": round(fill_rate, 1),
            "total_executed": total_executed,
            "total_failed": total_failed,
            "limit_price_buffer": buffer,
        }

        await self.save_performance({
            "fill_rate": round(fill_rate, 1),
            "buffer": buffer,
        })

        print(f"⚡ Executor LEARN: fill_rate={fill_rate:.1f}%, buffer={buffer}%")
        return learn_result
