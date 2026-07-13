from datetime import datetime, timedelta
from app.agents.base_agent import BaseAgent
from app.db.mongodb import get_db


class RiskManager(BaseAgent):
    """
    🛡️ AGENTE 3: Risk Manager v4.2 — DPS + Kelly Criterion
    """

    def __init__(self):
        super().__init__(name="risk_manager", version="2.0")

    def default_params(self) -> dict:
        return {
            "risk_pct_per_trade": 2.0,
            "max_position_pct": 20.0,
            "max_positions": 5,
            "max_per_sector": 2,
            "position_sizing_mode": "notional",
            "position_size_pct": 20.0,
            "fractionable_only": True,
            "min_notional_per_trade": 100.0,
            "min_risk_reward": 1.5,
            "ideal_risk_reward": 2.0,
            "daily_loss_limit_pct": -3.0,
            "weekly_loss_limit_pct": -5.0,
            "drawdown_reduce_pct": -3.0,
            "drawdown_stop_pct": -5.0,
            "min_cash_reserve_pct": 10.0,
            # DPS v4.1
            "dps_enabled": True,
            "dps_rr_ideal": 2.5,
            "dps_ml_ideal": 75.0,
            "dps_conf_ideal": 55.0,
            "dps_max_multiplier": 1.6,
            "dps_min_multiplier": 0.5,
            "dps_aggressiveness": 1.0,
            # 🆕 Kelly v4.2
            "kelly_enabled": True,
            "kelly_min_trades": 20,
            "kelly_fractional_factor": 0.25,
        }

    async def _check_loss_limits(self, account: dict, params: dict) -> dict:
        db = get_db()
        equity = float(account.get("equity", 0))
        last_equity = float(account.get("last_equity", equity))
        daily_pnl = equity - last_equity
        daily_pnl_pct = (daily_pnl / last_equity * 100) if last_equity > 0 else 0
        week_ago = datetime.utcnow() - timedelta(days=7)
        weekly_trades = await db.trade_history.find({
            "date": {"$gte": week_ago},
            "side": "sell",
        }).to_list(100)
        weekly_pnl_pct = sum(t.get("pnl_pct", 0) for t in weekly_trades)
        daily_limit = params.get("daily_loss_limit_pct", -3.0)
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

    # ============================================
    # DPS v4.1 — Smart Multiplier
    # ============================================
    def _calc_smart_multiplier(self, candidate: dict, params: dict) -> dict:
        if not params.get("dps_enabled", True):
            return {"multiplier": 1.0, "breakdown": "DPS disabled"}
        rr_ideal = params.get("dps_rr_ideal", 2.5)
        ml_ideal = params.get("dps_ml_ideal", 75.0)
        conf_ideal = params.get("dps_conf_ideal", 55.0)
        max_mult = params.get("dps_max_multiplier", 1.6)
        min_mult = params.get("dps_min_multiplier", 0.5)
        aggressiveness = params.get("dps_aggressiveness", 1.0)
        rr = candidate.get("risk_reward", 1.0)
        ml_score = candidate.get("ml_score", 50)
        if ml_score > 0 and ml_score <= 1.0:
            ml_score = ml_score * 100
        confluence = candidate.get("confluence", 40)
        rr_ratio = rr / rr_ideal if rr_ideal > 0 else 1.0
        rr_mult = max(0.5, min(1.4, 0.4 + rr_ratio * 0.6))
        if ml_score > 0:
            ml_ratio = ml_score / ml_ideal if ml_ideal > 0 else 1.0
            ml_mult = max(0.7, min(1.3, 0.5 + ml_ratio * 0.5))
        else:
            ml_mult = 1.0
        conf_ratio = confluence / conf_ideal if conf_ideal > 0 else 1.0
        conf_mult = max(0.7, min(1.3, 0.5 + conf_ratio * 0.5))
        combined = rr_mult * ml_mult * conf_mult
        combined = 1.0 + (combined - 1.0) * aggressiveness
        final_mult = max(min_mult, min(max_mult, combined))
        breakdown = (
            f"R/R {rr:.2f} -> {rr_mult:.2f}x | "
            f"ML {ml_score:.0f}% -> {ml_mult:.2f}x | "
            f"Conf {confluence:.0f} -> {conf_mult:.2f}x | "
            f"Final: {final_mult:.2f}x"
        )
        return {
            "multiplier": round(final_mult, 3),
            "rr_mult": round(rr_mult, 3),
            "ml_mult": round(ml_mult, 3),
            "conf_mult": round(conf_mult, 3),
            "breakdown": breakdown,
        }

    # ============================================
    # 🆕 v4.2 — KELLY CRITERION + VOLATILITY
    # ============================================
    async def _calc_kelly_multiplier(self, params: dict, market_ctx: dict) -> dict:
        db = get_db()
        if not params.get("kelly_enabled", True):
            return {"combined": 1.0, "reason": "Kelly disabled"}
        min_trades = params.get("kelly_min_trades", 20)
        trades = await db.trade_history.find({
            "side": "sell",
            "pnl_pct": {"$exists": True}
        }).sort("date", -1).limit(100).to_list(100)
        if len(trades) < min_trades:
            return {"combined": 1.0, "reason": f"Kelly skipped: {len(trades)} trades < {min_trades} min"}
        wins = [t for t in trades if t.get("pnl_pct", 0) > 0]
        losses = [t for t in trades if t.get("pnl_pct", 0) <= 0]
        if not wins or not losses:
            return {"combined": 1.0, "reason": "Kelly skipped: no wins or losses yet"}
        win_rate = len(wins) / len(trades)
        loss_rate = 1 - win_rate
        avg_win = sum(t.get("pnl_pct", 0) for t in wins) / len(wins)
        avg_loss = abs(sum(t.get("pnl_pct", 0) for t in losses) / len(losses))
        if avg_win <= 0 or avg_loss <= 0:
            return {"combined": 1.0, "reason": "Kelly skipped: invalid win/loss avg"}
        kelly_pct = (win_rate * avg_win - loss_rate * avg_loss) / avg_win
        fractional_factor = params.get("kelly_fractional_factor", 0.25)
        fractional_kelly = max(0, kelly_pct * fractional_factor)
        if fractional_kelly <= 0:
            kelly_mult = 0.5
        elif fractional_kelly < 0.05:
            kelly_mult = 0.5 + (fractional_kelly / 0.05) * 0.5
        elif fractional_kelly < 0.15:
            kelly_mult = 1.0 + ((fractional_kelly - 0.05) / 0.10) * 0.5
        else:
            kelly_mult = 1.5
        volatility_regime = market_ctx.get("volatility_regime", "NORMAL")
        vix_mult_map = {"LOW": 1.0, "NORMAL": 1.0, "HIGH": 0.85, "EXTREME": 0.7}
        volatility_mult = vix_mult_map.get(volatility_regime, 1.0)
        combined = kelly_mult * volatility_mult
        reason = (
            f"Kelly: WR {win_rate*100:.0f}% x win {avg_win:.1f}% - LR {loss_rate*100:.0f}% x loss {avg_loss:.1f}% "
            f"= {kelly_pct*100:.1f}% (fractional {fractional_kelly*100:.1f}%) -> {kelly_mult:.2f}x | "
            f"Vol {volatility_regime} -> {volatility_mult:.2f}x | Combined: {combined:.2f}x"
        )
        return {
            "kelly_pct": round(kelly_pct * 100, 2),
            "fractional_kelly": round(fractional_kelly * 100, 2),
            "kelly_mult": round(kelly_mult, 3),
            "volatility_mult": round(volatility_mult, 3),
            "combined": round(combined, 3),
            "reason": reason,
            "n_trades": len(trades),
            "win_rate": round(win_rate * 100, 1),
        }

    # ============================================
    # Position Sizing Notional
    # ============================================
    def _calc_position_size_notional(self, price: float, stop_loss: float, equity: float,
                                      available_capital: float, risk_per_trade_usd: float,
                                      position_size_pct: float, min_notional: float) -> dict:
        if price <= 0 or stop_loss <= 0:
            return {"notional_usd": 0, "reason": "Invalid price/stop"}
        notional_by_pct = equity * (position_size_pct / 100)
        stop_loss_pct = abs(price - stop_loss) / price * 100
        if stop_loss_pct <= 0.1:
            return {"notional_usd": 0, "reason": "Stop loss too close to price"}
        notional_by_risk = risk_per_trade_usd / (stop_loss_pct / 100)
        notional_by_available = available_capital * 0.95
        notional_usd = min(notional_by_pct, notional_by_risk, notional_by_available)
        notional_usd = max(notional_usd, 0)
        if notional_usd < min_notional:
            return {"notional_usd": 0, "reason": f"Below min notional (${notional_usd:.0f} < ${min_notional:.0f})"}
        estimated_shares = round(notional_usd / price, 4)
        estimated_risk_usd = notional_usd * (stop_loss_pct / 100)
        return {
            "notional_usd": round(notional_usd, 2),
            "estimated_shares": estimated_shares,
            "estimated_risk_usd": round(estimated_risk_usd, 2),
            "stop_loss_pct": round(stop_loss_pct, 2),
            "pct_of_equity": round((notional_usd / equity) * 100, 2) if equity > 0 else 0,
            "sizing_by_pct": round(notional_by_pct, 2),
            "sizing_by_risk": round(notional_by_risk, 2),
            "sizing_by_available": round(notional_by_available, 2),
        }

    def _calc_position_size(self, price: float, stop_loss: float, equity: float,
                            buying_power: float, risk_per_trade: float,
                            max_position_value: float) -> dict:
        risk_per_share = abs(price - stop_loss)
        if risk_per_share <= 0.01 or price <= 0:
            return {"shares": 0, "reason": "Invalid stop loss"}
        shares_by_risk = int(risk_per_trade / risk_per_share)
        shares_by_value = int(max_position_value / price) if max_position_value > price else 0
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

    # ============================================
    # ANALYZE
    # ============================================
    async def analyze(self, context: dict) -> dict:
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
        
        loss_check = await self._check_loss_limits(account, params)
        exposure_from_losses = loss_check["exposure_modifier"]
        regime_multiplier = market_ctx.get("exposure_multiplier", 0.5)
        final_multiplier = regime_multiplier * exposure_from_losses
        
        sizing_mode = params.get("position_sizing_mode", "notional")
        fractionable_only = params.get("fractionable_only", True)
        min_notional = params.get("min_notional_per_trade", 100.0)
        position_size_pct = params.get("position_size_pct", 20.0) * final_multiplier
        risk_pct = params.get("risk_pct_per_trade", 2.0) / 100
        max_pos_pct = params.get("max_position_pct", 20.0) / 100
        max_positions = params.get("max_positions", 5)
        max_per_sector = params.get("max_per_sector", 2)
        min_rr = params.get("min_risk_reward", 1.5)
        min_cash_reserve = params.get("min_cash_reserve_pct", 10.0) / 100
        risk_per_trade_usd = equity * risk_pct * final_multiplier
        max_position_value = equity * max_pos_pct * final_multiplier
        cash_reserve = equity * min_cash_reserve
        num_positions = len(positions)
        
        db = get_db()
        assets_all = await db.assets.find({}, {"ticker": 1, "sector_code": 1}).to_list(300)
        ticker_to_sector = {a["ticker"]: a.get("sector_code", "UNKNOWN") for a in assets_all}
        sector_exposure = {}
        for p in positions:
            sec = ticker_to_sector.get(p.get("symbol"), "UNKNOWN")
            sector_exposure[sec] = sector_exposure.get(sec, 0) + 1
        
        approved_sells = []
        for s in sell_signals:
            approved_sells.append({
                **s,
                "approved": True,
                "priority": {"critical": 1, "high": 2, "normal": 3}.get(s.get("urgency"), 3),
            })
        approved_sells.sort(key=lambda x: x["priority"])
        
        approved_trades = []
        rejected_trades = []
        
        # 🆕 v4.2 — Calcola Kelly UNA volta (comune a tutti i candidati)
        kelly_result = await self._calc_kelly_multiplier(params, market_ctx)
        kelly_multiplier = kelly_result["combined"]
        if kelly_multiplier != 1.0:
            print(f"  💰 Kelly: {kelly_result['reason']}")
        
        if loss_check["status"] == "stopped":
            for c in candidates:
                rejected_trades.append({**c, "reason": f"Trading stopped: {loss_check['reason']}"})
            print(f"🛡️ RiskManager: TRADING STOPPED — {loss_check['reason']}")
        else:
            available_buying_power = buying_power - cash_reserve
            committed = sum(float(p.get("market_value", 0)) for p in positions)
            from app.services.alpaca_trader import get_orders as get_alpaca_orders
            pending_value = 0
            try:
                open_orders = await get_alpaca_orders(status="open", limit=100)
                if open_orders:
                    for o in open_orders:
                        if o.get("side") == "buy" and o.get("status") in ("new", "accepted", "pending_new"):
                            if o.get("notional"):
                                pending_value += float(o.get("notional", 0))
                            else:
                                qty = float(o.get("qty", 0))
                                price = float(o.get("limit_price", 0) or 0)
                                pending_value += qty * price
            except:
                pass
            real_available = equity - committed - pending_value - cash_reserve
            if real_available < 0:
                real_available = 0
            available_buying_power = min(available_buying_power, real_available)
            
            fractionable_cache = {}
            candidate_tickers = [c["ticker"] for c in candidates]
            if candidate_tickers and fractionable_only:
                cursor = db.assets.find(
                    {"ticker": {"$in": candidate_tickers}},
                    {"ticker": 1, "fractionable": 1, "fractionable_checked_at": 1}
                )
                async for doc in cursor:
                    t = doc.get("ticker")
                    if "fractionable" in doc:
                        fractionable_cache[t] = doc.get("fractionable", False)
            
            for c in candidates:
                ticker = c["ticker"]
                price = c["price"]
                stop_loss = c["stop_loss"]
                sector = c.get("sector", "UNKNOWN")
                rr = c.get("risk_reward", 0)
                
                if num_positions + len(approved_trades) >= max_positions:
                    rejected_trades.append({**c, "reason": "Max positions reached"})
                    continue
                current_sector_count = sector_exposure.get(sector, 0)
                approved_sector_count = sum(1 for t in approved_trades if t.get("sector") == sector)
                if current_sector_count + approved_sector_count >= max_per_sector:
                    rejected_trades.append({**c, "reason": f"Sector {sector} full ({max_per_sector})"})
                    continue
                if rr < min_rr:
                    rejected_trades.append({**c, "reason": f"R/R too low: {rr} < {min_rr}"})
                    continue
                if fractionable_only:
                    is_frac = fractionable_cache.get(ticker)
                    if is_frac is None:
                        try:
                            from app.services.alpaca_trader import is_fractionable
                            is_frac = await is_fractionable(ticker)
                            fractionable_cache[ticker] = is_frac
                            await db.assets.update_one(
                                {"ticker": ticker},
                                {"$set": {
                                    "fractionable": is_frac,
                                    "fractionable_checked_at": datetime.utcnow(),
                                }},
                                upsert=False
                            )
                        except Exception as e:
                            print(f"  ⚠️ Fractionable check error {ticker}: {e}")
                            is_frac = False
                            fractionable_cache[ticker] = False
                    if not is_frac:
                        rejected_trades.append({**c, "reason": "Not fractionable"})
                        continue
                
                if sizing_mode == "notional":
                    # 🆕 v4.1 DPS multiplier
                    dps_result = self._calc_smart_multiplier(c, params)
                    dps_multiplier = dps_result["multiplier"]
                    
                    # 🆕 v4.2 Combined: DPS × Kelly
                    combined_multiplier = dps_multiplier * kelly_multiplier
                    dynamic_size_pct = position_size_pct * combined_multiplier
                    
                    if combined_multiplier != 1.0:
                        print(f"  🎯 {ticker}: {dps_result['breakdown']}")
                        print(f"     Size: {position_size_pct:.1f}% x DPS {dps_multiplier:.2f} x Kelly {kelly_multiplier:.2f} = {dynamic_size_pct:.1f}%")
                    
                    sizing = self._calc_position_size_notional(
                        price=price, stop_loss=stop_loss, equity=equity,
                        available_capital=available_buying_power,
                        risk_per_trade_usd=risk_per_trade_usd,
                        position_size_pct=dynamic_size_pct,
                        min_notional=min_notional,
                    )
                    notional_usd = sizing.get("notional_usd", 0)
                    if notional_usd <= 0:
                        rejected_trades.append({**c, "reason": sizing.get("reason", "Zero notional")})
                        continue
                    used_cash = sum(t.get("notional_usd", 0) for t in approved_trades)
                    if cash - used_cash - notional_usd < 0:
                        rejected_trades.append({**c, "reason": "Cash would go negative"})
                        continue
                    approved_trade = {
                        **c,
                        "sizing_mode": "notional",
                        "notional_usd": notional_usd,
                        "estimated_shares": sizing.get("estimated_shares", 0),
                        "estimated_risk_usd": sizing.get("estimated_risk_usd", 0),
                        "stop_loss_pct": sizing.get("stop_loss_pct", 0),
                        "pct_of_equity": sizing.get("pct_of_equity", 0),
                        "approved": True,
                        "dps_multiplier": dps_multiplier,
                        "dps_base_pct": position_size_pct,
                        "dps_final_pct": dynamic_size_pct,
                        "dps_breakdown": dps_result["breakdown"],
                        "kelly_multiplier": kelly_multiplier,
                        "kelly_pct": kelly_result.get("kelly_pct", 0),
                        "kelly_reason": kelly_result.get("reason", ""),
                        "final_multiplier_combined": combined_multiplier,
                    }
                    approved_trades.append(approved_trade)
                    available_buying_power -= notional_usd
                else:
                    sizing = self._calc_position_size(
                        price, stop_loss, equity, available_buying_power,
                        risk_per_trade_usd, max_position_value
                    )
                    if sizing["shares"] <= 0:
                        rejected_trades.append({**c, "reason": sizing.get("reason", "Zero shares")})
                        continue
                    used_cash = sum(t.get("total_value", 0) for t in approved_trades)
                    if cash - used_cash - sizing["total_value"] < 0:
                        rejected_trades.append({**c, "reason": "Cash would go negative"})
                        continue
                    if sizing["total_value"] > available_buying_power:
                        rejected_trades.append({**c, "reason": "Insufficient buying power"})
                        continue
                    approved_trade = {
                        **c,
                        "sizing_mode": "shares",
                        "shares": sizing["shares"],
                        "total_value": sizing["total_value"],
                        "total_risk": sizing["total_risk"],
                        "risk_per_share": sizing["risk_per_share"],
                        "pct_of_equity": sizing["pct_of_equity"],
                        "approved": True,
                    }
                    approved_trades.append(approved_trade)
                    available_buying_power -= sizing["total_value"]
        
        total_market_value = sum(float(p.get("market_value", 0)) for p in positions)
        total_unrealized_pnl = sum(float(p.get("unrealized_pl", 0)) for p in positions)
        new_exposure = sum(t.get("notional_usd", t.get("total_value", 0)) for t in approved_trades)
        
        risk_report = {
            "equity": round(equity, 2),
            "cash": round(cash, 2),
            "buying_power": round(buying_power, 2),
            "total_market_value": round(total_market_value, 2),
            "total_unrealized_pnl": round(total_unrealized_pnl, 2),
            "new_exposure": round(new_exposure, 2),
            "total_exposure_pct": round(((total_market_value + new_exposure) / equity) * 100, 1) if equity > 0 else 0,
            "sector_exposure": sector_exposure,
            "current_positions": num_positions,
            "max_positions": max_positions,
            "regime_multiplier": round(regime_multiplier, 2),
            "loss_check": loss_check,
            "final_multiplier": round(final_multiplier, 2),
            "risk_per_trade_usd": round(risk_per_trade_usd, 2),
            "cash_reserve": round(cash_reserve, 2),
            "sizing_mode": sizing_mode,
            "position_size_pct": round(position_size_pct, 2),
            "kelly_multiplier": kelly_multiplier,
            "kelly_pct": kelly_result.get("kelly_pct", 0),
        }
        
        from app.services.llm_service import llm_ask, llm_available
        llm_reasoning = None
        if llm_available():
            try:
                agents_context = ""
                try:
                    from app.agents.shared_brain import brain
                    brain_data = await brain.get_full_state()
                    macro_r = brain_data.get("market", {}).get("llm_reasoning", "")
                    if macro_r:
                        agents_context += f"\nMacro says: {macro_r[:150]}"
                except:
                    pass
                approved_summary = ", ".join(
                    f"{t['ticker']}(${t.get('notional_usd', t.get('total_value', 0)):.0f})"
                    for t in approved_trades
                )
                risk_summary = (
                    f"Equity: ${equity:.0f} | Cash: ${cash:.0f}\n"
                    f"Positions: {num_positions}/{max_positions}\n"
                    f"Exposure: {risk_report['total_exposure_pct']:.1f}%\n"
                    f"Regime: {market_ctx.get('market_regime', 'NEUTRAL')}\n"
                    f"Kelly: {kelly_multiplier:.2f}x\n"
                    f"Approved: {len(approved_trades)} ({approved_summary})\n"
                    f"Rejected: {len(rejected_trades)}\n"
                )
                llm_reasoning = llm_ask(
                    system_prompt="Sei un risk manager. Max 3 frasi italiano. Sii diretto.",
                    user_prompt=f"Risk report:\n{risk_summary}{agents_context}",
                    max_tokens=200,
                    temperature=0.3,
                    agent_name="risk_manager",
                )
            except Exception as e:
                print(f"  Risk LLM error: {e}")
        risk_report["llm_reasoning"] = llm_reasoning
        
        for t in approved_trades:
            await self.log_decision(
                decision_type="trade_approved",
                data={
                    "ticker": t["ticker"],
                    "notional_usd": t.get("notional_usd"),
                    "confluence": t.get("confluence", 0),
                    "risk_reward": t.get("risk_reward", 0),
                    "dps_multiplier": t.get("dps_multiplier"),
                    "kelly_multiplier": t.get("kelly_multiplier"),
                },
                reasoning=f"Approved {t['ticker']}: ${t.get('notional_usd', 0):.0f}",
                confidence=t.get("confluence", 50),
            )
        for t in rejected_trades:
            await self.log_decision(
                decision_type="trade_rejected",
                data={"ticker": t["ticker"], "reason": t["reason"]},
                reasoning=f"Rejected {t['ticker']}: {t['reason']}",
                confidence=20,
            )
        
        print(f"🛡️ RiskManager v4.2: {len(approved_trades)} approved, {len(rejected_trades)} rejected | Kelly {kelly_multiplier:.2f}x | Exposure: {risk_report['total_exposure_pct']:.1f}%")
        
        return {
            "approved_trades": approved_trades,
            "rejected_trades": rejected_trades,
            "approved_sells": approved_sells,
            "risk_report": risk_report,
        }

    async def learn(self) -> dict:
        db = get_db()
        params = await self.get_params()
        trades = await db.trade_history.find({"side": "sell"}).to_list(500)
        if len(trades) < self.min_decisions_to_learn:
            return {"message": "Not enough data to learn", "trades": len(trades)}
        wins = [t for t in trades if t.get("pnl_pct", 0) > 0]
        losses = [t for t in trades if t.get("pnl_pct", 0) <= 0]
        total = len(trades)
        win_rate = len(wins) / total * 100 if total > 0 else 50
        loss_pcts = [abs(t.get("pnl_pct", 0)) for t in losses]
        avg_loss = sum(loss_pcts) / len(loss_pcts) if loss_pcts else 2.0
        win_pcts = [t.get("pnl_pct", 0) for t in wins]
        avg_win = sum(win_pcts) / len(win_pcts) if win_pcts else 3.0
        total_wins = sum(win_pcts) if win_pcts else 0
        total_losses = sum(loss_pcts) if loss_pcts else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 1.0
        risk_pct = params.get("risk_pct_per_trade", 2.0)
        min_rr = params.get("min_risk_reward", 1.5)
        if avg_loss > 3.0 and risk_pct >= 2.0:
            risk_pct = max(1.0, risk_pct - 0.25)
        elif avg_loss < 1.5 and win_rate > 55:
            risk_pct = min(3.0, risk_pct + 0.25)
        if profit_factor < 1.2:
            min_rr = min(2.5, min_rr + 0.2)
        elif profit_factor > 2.0 and min_rr > 1.5:
            min_rr = max(1.3, min_rr - 0.1)
        params["risk_pct_per_trade"] = round(risk_pct, 2)
        params["min_risk_reward"] = round(min_rr, 2)
        await self.save_params(params)
        return {
            "win_rate": round(win_rate, 1),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "risk_pct_per_trade": risk_pct,
            "min_risk_reward": min_rr,
        }
