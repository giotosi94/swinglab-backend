from datetime import datetime, timedelta
from app.agents.base_agent import BaseAgent
from app.db.mongodb import get_db


class RiskManager(BaseAgent):
    """
    🛡️ AGENTE 3: Risk Manager — "Il Controllore del Rischio"
    Approva o rifiuta i trade proposti dall'AlphaStrategist.
    Calcola il position sizing, controlla i limiti di rischio,
    e protegge il portafoglio da drawdown eccessivi.

    Input: buy_candidates, sell_signals (da AlphaStrategist),
           market_context (da MacroAnalyst), account Alpaca
    Output: approved_trades[], rejected_trades[], risk_report
    """

    def __init__(self):
        super().__init__(name="risk_manager", version="1.0")

    def default_params(self) -> dict:
        return {
            # Position sizing
            "risk_pct_per_trade": 2.0,      # % equity rischiato per trade
            "max_position_pct": 20.0,       # % equity max per singola posizione
            "max_positions": 5,
            "max_per_sector": 2,
            # Risk/Reward
            "min_risk_reward": 1.5,         # Minimo R/R accettabile
            "ideal_risk_reward": 2.0,       # R/R ideale (bonus)
            # Loss limits
            "daily_loss_limit_pct": -3.0,   # Max perdita giornaliera (%)
            "weekly_loss_limit_pct": -5.0,  # Max perdita settimanale (%)
            "drawdown_reduce_pct": -3.0,    # Sotto questo, dimezza esposizione
            "drawdown_stop_pct": -5.0,      # Sotto questo, stop trading
            # Minimum buying power reserved
            "min_cash_reserve_pct": 10.0,   # Mantieni almeno 10% in cash
        }

    async def _check_loss_limits(self, account: dict, params: dict) -> dict:
        """
        Verifica se i limiti di perdita giornaliera/settimanale sono stati superati.
        Ritorna lo stato del trading: allowed, reduced, stopped.
        """
        db = get_db()
        equity = float(account.get("equity", 0))
        last_equity = float(account.get("last_equity", equity))

        # Daily P&L
        daily_pnl = equity - last_equity
        daily_pnl_pct = (daily_pnl / last_equity * 100) if last_equity > 0 else 0

        # Weekly P&L (dalle performance history se disponibile)
        week_ago = datetime.utcnow() - timedelta(days=7)
        weekly_trades = await db.trade_history.find({
            "date": {"$gte": week_ago},
            "side": "sell",
        }).to_list(100)
        weekly_pnl_pct = sum(t.get("pnl_pct", 0) for t in weekly_trades)

        daily_limit = params.get("daily_loss_limit_pct", -3.0)
        weekly_limit = params.get("weekly_loss_limit_pct", -5.0)
        drawdown_reduce = params.get("drawdown_reduce_pct", -3.0)
        drawdown_stop = params.get("drawdown_stop_pct", -5.0)

        trading_status = "allowed"
        status_reason = "Normal"
        exposure_modifier = 1.0

        if daily_pnl_pct <= drawdown_stop or weekly_pnl_pct <= drawdown_stop:
            trading_status = "stopped"
            status_reason = f"Loss limit hit: daily={daily_pnl_pct:.1f}%, weekly={weekly_pnl_pct:.1f}%"
            exposure_modifier = 0.0
        elif daily_pnl_pct <= drawdown_reduce or weekly_pnl_pct <= drawdown_reduce:
            trading_status = "reduced"
            status_reason = f"Reducing exposure: daily={daily_pnl_pct:.1f}%, weekly={weekly_pnl_pct:.1f}%"
            exposure_modifier = 0.5
        elif daily_pnl_pct <= daily_limit * 0.7:
            trading_status = "caution"
            status_reason = f"Approaching limits: daily={daily_pnl_pct:.1f}%"
            exposure_modifier = 0.75

        return {
            "status": trading_status,
            "reason": status_reason,
            "exposure_modifier": exposure_modifier,
            "daily_pnl_pct": round(daily_pnl_pct, 2),
            "weekly_pnl_pct": round(weekly_pnl_pct, 2),
        }

    def _calc_position_size(self, price: float, stop_loss: float, equity: float,
                            buying_power: float, risk_per_trade: float,
                            max_position_value: float) -> dict:
        """Calcola il numero di shares basato su risk management."""
        risk_per_share = abs(price - stop_loss)
        if risk_per_share <= 0.01 or price <= 0:
            return {"shares": 0, "reason": "Invalid stop loss"}

        # Shares basato su rischio per trade
        shares_by_risk = int(risk_per_trade / risk_per_share)
        # Shares basato su max position value
        shares_by_value = int(max_position_value / price) if max_position_value > price else 0
        # Shares basato su buying power disponibile (lascia margine)
        shares_by_bp = int((buying_power * 0.9) / price) if buying_power > price else 0

        shares = min(shares_by_risk, shares_by_value, shares_by_bp)
        shares = max(shares, 0)

        if shares <= 0:
            return {"shares": 0, "reason": "Insufficient capital"}

        return {
            "shares": shares,
            "total_risk": round(risk_per_share * shares, 2),
            "total_value": round(price * shares, 2),
            "risk_per_share": round(risk_per_share, 2),
            "pct_of_equity": round((price * shares / equity) * 100, 2) if equity > 0 else 0,
        }

    async def analyze(self, context: dict) -> dict:
        """
        Analisi del rischio: approva, modifica o rifiuta i trade proposti.
        context deve contenere:
        - market_context: output del MacroAnalyst
        - buy_candidates: output dell'AlphaStrategist
        - sell_signals: output dell'AlphaStrategist
        - account: dati account Alpaca
        - positions: posizioni aperte Alpaca
        """
        params = await self.get_params()
        market_ctx = context.get("market_context", {})
        candidates = context.get("buy_candidates", [])
        sell_signals = context.get("sell_signals", [])
        account = context.get("account", {})
        positions = context.get("positions", [])

        equity = float(account.get("equity", 0))
        cash = float(account.get("cash", 0))
        buying_power = float(account.get("buying_power", 0))

        if equity <= 0:
            return {"error": "No equity data", "approved_trades": [], "rejected_trades": []}

        # ============================================
        # 1. CHECK LOSS LIMITS
        # ============================================
        loss_check = await self._check_loss_limits(account, params)
        exposure_from_losses = loss_check["exposure_modifier"]

        # Regime multiplier dal MacroAnalyst
        regime_multiplier = market_ctx.get("exposure_multiplier", 0.5)

        # Multiplier combinato
        final_multiplier = regime_multiplier * exposure_from_losses

        # ============================================
        # 2. CALCULATE RISK BUDGET
        # ============================================
        risk_pct = params.get("risk_pct_per_trade", 2.0) / 100
        max_pos_pct = params.get("max_position_pct", 20.0) / 100
        max_positions = params.get("max_positions", 5)
        max_per_sector = params.get("max_per_sector", 2)
        min_rr = params.get("min_risk_reward", 1.5)
        min_cash_reserve = params.get("min_cash_reserve_pct", 10.0) / 100

        risk_per_trade = equity * risk_pct * final_multiplier
        max_position_value = equity * max_pos_pct * final_multiplier
        cash_reserve = equity * min_cash_reserve

        num_positions = len(positions)

        # Conta esposizione per settore
        db = get_db()
        assets_all = await db.assets.find({}, {"ticker": 1, "sector_code": 1}).to_list(300)
        ticker_to_sector = {a["ticker"]: a.get("sector_code", "UNKNOWN") for a in assets_all}
        sector_exposure = {}
        for p in positions:
            sec = ticker_to_sector.get(p.get("symbol"), "UNKNOWN")
            sector_exposure[sec] = sector_exposure.get(sec, 0) + 1

        # ============================================
        # 3. PROCESS SELL SIGNALS (sempre approvati — urgency li prioritizza)
        # ============================================
        approved_sells = []
        for s in sell_signals:
            approved_sells.append({
                **s,
                "approved": True,
                "priority": {"critical": 1, "high": 2, "normal": 3}.get(s.get("urgency"), 3),
            })
        approved_sells.sort(key=lambda x: x["priority"])

        # ============================================
        # 4. PROCESS BUY CANDIDATES
        # ============================================
        approved_trades = []
        rejected_trades = []

        # Se trading e' stopped, rifiuta tutto
        if loss_check["status"] == "stopped":
            for c in candidates:
                rejected_trades.append({
                    **c, "reason": f"Trading stopped: {loss_check['reason']}"
                })
            print(f"🛡️ RiskManager: TRADING STOPPED — {loss_check['reason']}")
        else:
            available_buying_power = buying_power - cash_reserve

            for c in candidates:
                ticker = c["ticker"]
                price = c["price"]
                stop_loss = c["stop_loss"]
                target = c["target_price"]
                sector = c.get("sector", "UNKNOWN")
                rr = c.get("risk_reward", 0)

                # Check max positions
                if num_positions + len(approved_trades) >= max_positions:
                    rejected_trades.append({**c, "reason": "Max positions reached"})
                    continue

                # Check sector limit
                current_sector_count = sector_exposure.get(sector, 0)
                approved_sector_count = sum(1 for t in approved_trades if t.get("sector") == sector)
                if current_sector_count + approved_sector_count >= max_per_sector:
                    rejected_trades.append({**c, "reason": f"Sector {sector} full ({max_per_sector})"})
                    continue

                # Check Risk/Reward
                if rr < min_rr:
                    rejected_trades.append({**c, "reason": f"R/R too low: {rr} < {min_rr}"})
                    continue

                # Calculate position size
                sizing = self._calc_position_size(
                    price, stop_loss, equity, available_buying_power,
                    risk_per_trade, max_position_value
                )

                if sizing["shares"] <= 0:
                    rejected_trades.append({**c, "reason": sizing.get("reason", "Zero shares")})
                    continue

                # Check buying power
                if sizing["total_value"] > available_buying_power:
                    rejected_trades.append({**c, "reason": "Insufficient buying power"})
                    continue

                # APPROVED! ✅
                approved_trade = {
                    **c,
                    "shares": sizing["shares"],
                    "total_value": sizing["total_value"],
                    "total_risk": sizing["total_risk"],
                    "risk_per_share": sizing["risk_per_share"],
                    "pct_of_equity": sizing["pct_of_equity"],
                    "approved": True,
                }
                approved_trades.append(approved_trade)
                available_buying_power -= sizing["total_value"]

        # ============================================
        # 5. RISK REPORT
        # ============================================
        total_market_value = sum(
            float(p.get("market_value", 0)) for p in positions
        )
        total_unrealized_pnl = sum(
            float(p.get("unrealized_pl", 0)) for p in positions
        )
        new_exposure = sum(t["total_value"] for t in approved_trades)

        risk_report = {
            "equity": round(equity, 2),
            "cash": round(cash, 2),
            "buying_power": round(buying_power, 2),
            "total_market_value": round(total_market_value, 2),
            "total_unrealized_pnl": round(total_unrealized_pnl, 2),
            "new_exposure": round(new_exposure, 2),
            "total_exposure_pct": round(
                ((total_market_value + new_exposure) / equity) * 100, 1
            ) if equity > 0 else 0,
            "sector_exposure": sector_exposure,
            "current_positions": num_positions,
            "max_positions": max_positions,
            "regime_multiplier": round(regime_multiplier, 2),
            "loss_check": loss_check,
            "final_multiplier": round(final_multiplier, 2),
            "risk_per_trade": round(risk_per_trade, 2),
            "cash_reserve": round(cash_reserve, 2),
        }

# ============================================
        # 6. LLM REASONING (optional)
        # ============================================
        from app.services.llm_service import llm_ask, llm_available
        llm_reasoning = None
        if llm_available():
            try:
# Read other agents' reasoning
                agents_context = ""
                try:
                    from app.agents.shared_brain import brain
                    brain_data = await brain.get_full_state()
                    macro_r = brain_data.get("market", {}).get("llm_reasoning", "")
                    if macro_r:
                        agents_context += f"\nMacro says: {macro_r[:150]}"
                    candidates = brain_data.get("candidates", {}).get("buy", [])
                    if candidates:
                        top_tickers = ", ".join(c.get("ticker", "") for c in candidates[:3])
                        agents_context += f"\nAlpha top picks: {top_tickers}"
                except:
                    pass
                
                risk_summary = (
                    f"Equity: ${equity:.0f} | Cash: ${cash:.0f}\n"
                    f"Positions: {num_positions}/{max_positions}\n"
                    f"Exposure: {risk_report['total_exposure_pct']:.1f}%\n"
                    f"Regime: {market_ctx.get('market_regime', 'NEUTRAL')} (multiplier {final_multiplier:.1f}x)\n"
                    f"Daily P&L: {loss_check['daily_pnl_pct']:.2f}% | Weekly: {loss_check['weekly_pnl_pct']:.2f}%\n"
                    f"Sector exposure: {sector_exposure}\n"
                    f"Approved: {len(approved_trades)} trades ({', '.join(t['ticker'] for t in approved_trades)})\n"
                    f"Rejected: {len(rejected_trades)} ({', '.join(t['ticker']+': '+t['reason'] for t in rejected_trades[:3])})\n"
                    f"Sells: {len(approved_sells)} ({', '.join(s['ticker']+': '+s['reason'] for s in approved_sells)})\n"
                    f"Risk per trade: ${risk_per_trade:.0f}\n"
                    f"New exposure: ${new_exposure:.0f}"
                )
                llm_reasoning = llm_ask(
                    system_prompt=(
                        "Sei un risk manager esperto di swing trading. "
                        "Valuta il portafoglio e le decisioni di rischio in max 3 frasi in italiano. "
                        "Indica: 1) Se l'esposizione è adeguata, 2) Il rischio principale, 3) Suggerimento. "
                        "Sii diretto, concreto, no disclaimers."
                    ),
                    user_prompt=f"Risk report:\n{risk_summary}{agents_context}",
                    max_tokens=200,
                    temperature=0.3,
                )
                if llm_reasoning:
                    print(f"  🧠 Risk LLM: {llm_reasoning[:80]}...")
            except Exception as e:
                print(f"  Risk LLM error: {e}")

        risk_report["llm_reasoning"] = llm_reasoning
        
        # Log decisions
        for t in approved_trades:
            await self.log_decision(
                decision_type="trade_approved",
                data={
                    "ticker": t["ticker"],
                    "shares": t["shares"],
                    "confluence": t.get("confluence", 0),
                    "risk_reward": t.get("risk_reward", 0),
                    "total_risk": t["total_risk"],
                    "sector": t.get("sector"),
                    "regime": market_ctx.get("market_regime"),
                },
                reasoning=f"Approved {t['ticker']}: {t['shares']} shares, "
                          f"R/R={t.get('risk_reward', 0)}, "
                          f"risk=${t['total_risk']}",
                confidence=t.get("confluence", 50),
            )

        for t in rejected_trades:
            await self.log_decision(
                decision_type="trade_rejected",
                data={"ticker": t["ticker"], "reason": t["reason"]},
                reasoning=f"Rejected {t['ticker']}: {t['reason']}",
                confidence=20,
            )

        print(f"🛡️ RiskManager: {len(approved_trades)} approved, "
              f"{len(rejected_trades)} rejected, "
              f"{len(approved_sells)} sells | "
              f"Exposure: {risk_report['total_exposure_pct']:.1f}%")

        return {
            "approved_trades": approved_trades,
            "rejected_trades": rejected_trades,
            "approved_sells": approved_sells,
            "risk_report": risk_report,
        }

    async def learn(self) -> dict:
        """
        Learning loop del RiskManager.
        Analizza se il position sizing e i limiti di rischio hanno protetto
        il portafoglio in modo ottimale.
        """
        db = get_db()
        params = await self.get_params()

        trades = await db.trade_history.find({"side": "sell"}).to_list(500)
        if len(trades) < self.min_decisions_to_learn:
            return {"message": "Not enough data to learn", "trades": len(trades)}

        wins = [t for t in trades if t.get("pnl_pct", 0) > 0]
        losses = [t for t in trades if t.get("pnl_pct", 0) <= 0]
        total = len(trades)
        win_rate = len(wins) / total * 100 if total > 0 else 50

        # Analisi distribuzione perdite
        loss_pcts = [abs(t.get("pnl_pct", 0)) for t in losses]
        avg_loss = sum(loss_pcts) / len(loss_pcts) if loss_pcts else 2.0
        max_loss = max(loss_pcts) if loss_pcts else 5.0

        # Analisi distribuzione vincite
        win_pcts = [t.get("pnl_pct", 0) for t in wins]
        avg_win = sum(win_pcts) / len(win_pcts) if win_pcts else 3.0

        # Profit factor
        total_wins = sum(win_pcts) if win_pcts else 0
        total_losses = sum(loss_pcts) if loss_pcts else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 1.0

        # Aggiusta parametri basandosi sui risultati
        risk_pct = params.get("risk_pct_per_trade", 2.0)
        min_rr = params.get("min_risk_reward", 1.5)

        # Se le perdite sono troppo grandi, riduci rischio per trade
        if avg_loss > 3.0 and risk_pct >= 2.0:
            risk_pct = max(1.0, risk_pct - 0.25)
        elif avg_loss < 1.5 and win_rate > 55:
            risk_pct = min(3.0, risk_pct + 0.25)

        # Se profit factor basso, aumenta R/R minimo
        if profit_factor < 1.2:
            min_rr = min(2.5, min_rr + 0.2)
        elif profit_factor > 2.0 and min_rr > 1.5:
            min_rr = max(1.3, min_rr - 0.1)

        # Max positions: se drawdown frequenti, riduci
        max_positions = params.get("max_positions", 5)
        recent_losses = [t for t in losses if
                         t.get("date") and
                         (datetime.utcnow() - t["date"]).days < 14]
        if len(recent_losses) >= 3 and max_positions > 3:
            max_positions = max(3, max_positions - 1)
        elif len(recent_losses) == 0 and win_rate > 60 and max_positions < 7:
            max_positions = min(7, max_positions + 1)

        params["risk_pct_per_trade"] = round(risk_pct, 2)
        params["min_risk_reward"] = round(min_rr, 2)
        params["max_positions"] = max_positions

        await self.save_params(params)

        learn_result = {
            "win_rate": round(win_rate, 1),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "max_loss": round(max_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "risk_pct_per_trade": risk_pct,
            "min_risk_reward": min_rr,
            "max_positions": max_positions,
        }

        await self.save_performance({
            "profit_factor": round(profit_factor, 2),
            "risk_pct": risk_pct,
            "max_positions": max_positions,
        })

        print(f"🛡️ RiskManager LEARN: PF={profit_factor:.2f}, "
              f"risk={risk_pct}%, R/R>={min_rr}, max_pos={max_positions}")

        return learn_result
