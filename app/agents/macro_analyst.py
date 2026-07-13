from datetime import datetime, timedelta
from app.agents.base_agent import BaseAgent
from app.db.mongodb import get_db


class MacroAnalyst(BaseAgent):
    """
    🌍 AGENTE 1: Macro Analyst v2.0 — Formula continua (no più bucket)
    """

    def __init__(self):
        super().__init__(name="macro_analyst", version="2.0")

    def default_params(self) -> dict:
        return {
            "w_spy_trend": 0.18,
            "w_spy_rsi": 0.10,
            "w_vix": 0.12,
            "w_breadth": 0.15,
            "w_indices_alignment": 0.08,
            "w_crypto": 0.04,
            "w_dollar": 0.08,
            "w_bonds": 0.10,
            "w_commodities": 0.05,
            "w_risk_appetite": 0.10,
            "vixy_high": 22,
            "vixy_extreme": 28,
            "vixy_low": 14,
            "breadth_healthy": 60,
            "breadth_weak": 40,
            "breadth_critical": 25,
            "bull_exposure": 1.0,
            "neutral_exposure": 0.6,
            "bear_exposure": 0.3,
            "crash_exposure": 0.0,
        }

    def _clamp(self, value, min_val=0, max_val=100):
        """Limita valore in range."""
        return max(min_val, min(max_val, value))

    async def analyze(self, context: dict = None) -> dict:
        db = get_db()
        params = await self.get_params()

        # ============================================
        # 1. SPY - trend + RSI (formule continue)
        # ============================================
        spy = await db.market_regime.find_one({"symbol": "SPY"})
        spy_price = spy.get("price", 0) if spy else 0
        spy_ema20 = spy.get("ema20", 0) if spy else 0
        spy_ema50 = spy.get("ema50", 0) if spy else 0
        spy_rsi = spy.get("rsi", 50) if spy else 50
        spy_return_20d = spy.get("return_20d", 0) if spy else 0

        # 🆕 SPY TREND SCORE — proporzionale al distacco EMA
        if spy_ema50 > 0 and spy_price > 0:
            # % distacco price da EMA50 + bonus se sopra EMA20
            dist_from_ema50 = ((spy_price - spy_ema50) / spy_ema50) * 100  # -X% a +X%
            # Base score: 50 = neutro. +2 punti ogni +1% sopra EMA50
            spy_trend_score = 50 + (dist_from_ema50 * 2.5)
            # Bonus allineamento EMA20 > EMA50
            if spy_ema20 > spy_ema50:
                ema_slope = ((spy_ema20 - spy_ema50) / spy_ema50) * 100
                spy_trend_score += min(15, ema_slope * 3)
            # Return 20d contribuisce
            spy_trend_score += self._clamp(spy_return_20d * 1.5, -10, 15)
            spy_trend_score = self._clamp(spy_trend_score, 5, 95)
        else:
            spy_trend_score = 50

        # 🆕 RSI SCORE — proporzionale alla distanza da 52 (sweet spot)
        # 52 = ideale → 100. 20 o 80 = 40.
        rsi_distance = abs(spy_rsi - 52)
        rsi_score = self._clamp(100 - (rsi_distance * 1.8), 20, 100)

        # ============================================
        # 2. VOLATILITY (VIXY) — score continuo
        # ============================================
        vixy = await db.market_regime.find_one({"symbol": "VIXY"})
        vixy_price = vixy.get("price", 18) if vixy else 18

        # 🆕 Formula continua: VIXY 14 = 90, VIXY 22 = 60, VIXY 28 = 30
        # Linear decay: score = 130 - (vixy_price * 3.5)
        vol_score = self._clamp(130 - (vixy_price * 3.5), 5, 95)

        vixy_low = params.get("vixy_low", 14)
        vixy_high = params.get("vixy_high", 22)
        vixy_extreme = params.get("vixy_extreme", 28)
        if vixy_price <= vixy_low:
            volatility_regime = "LOW"
        elif vixy_price <= vixy_high:
            volatility_regime = "NORMAL"
        elif vixy_price <= vixy_extreme:
            volatility_regime = "HIGH"
        else:
            volatility_regime = "EXTREME"

        # ============================================
        # 3. INDICES ALIGNMENT (con magnitudine)
        # ============================================
        indices_return_sum = 0
        indices_bullish = 0
        indices_total = 0
        for sym in ["QQQ", "IWM", "DIA"]:
            idx = await db.market_regime.find_one({"symbol": sym})
            if idx:
                indices_total += 1
                ret = idx.get("return_20d", 0) or idx.get("change_pct", 0)
                indices_return_sum += ret
                if ret > 0:
                    indices_bullish += 1

        if indices_total > 0:
            # 🆕 Score basato su return medio + allineamento
            avg_return = indices_return_sum / indices_total
            alignment_pct = (indices_bullish / indices_total) * 100
            # Peso 60% allineamento, 40% magnitudine return
            alignment_score = (alignment_pct * 0.6) + (self._clamp(50 + avg_return * 8, 0, 100) * 0.4)
            alignment_score = self._clamp(alignment_score, 5, 95)
        else:
            alignment_score = 50

        # ============================================
        # 4. CRYPTO SENTIMENT — continua
        # ============================================
        btc = await db.market_regime.find_one({"symbol": "BTC/USD"})
        eth = await db.market_regime.find_one({"symbol": "ETH/USD"})
        btc_change = btc.get("change_pct", 0) if btc else 0
        eth_change = eth.get("change_pct", 0) if eth else 0

        # 🆕 Score continuo: media pesata dei change
        crypto_avg = (btc_change + eth_change) / 2
        # 0% change = 50 score. +5% = 90. -5% = 10.
        crypto_score = self._clamp(50 + (crypto_avg * 8), 10, 95)

        if crypto_avg > 1:
            crypto_sentiment = "strong_risk_on" if crypto_avg > 2 else "risk_on"
        elif crypto_avg < -1:
            crypto_sentiment = "risk_off" if crypto_avg < -2 else "cautious"
        else:
            crypto_sentiment = "neutral"

        # ============================================
        # 5. DOLLAR STRENGTH — continua
        # ============================================
        fxe = await db.market_regime.find_one({"symbol": "FXE"})
        uup = await db.market_regime.find_one({"symbol": "UUP"})
        fxe_change = fxe.get("change_pct", 0) if fxe else 0
        uup_change = uup.get("change_pct", 0) if uup else 0

        dollar_net = uup_change - fxe_change

        # 🆕 Dollaro debole = buono per stocks USA. Score inverso.
        # dollar_net = 0 → 55 score. -1% → 75. +1% → 35.
        dollar_score = self._clamp(55 - (dollar_net * 20), 15, 90)

        if dollar_net < -0.3:
            dollar_strength = "weak"
        elif dollar_net > 0.3:
            dollar_strength = "strong"
        else:
            dollar_strength = "neutral"

        # ============================================
        # 5B. BONDS & CREDIT — continuo
        # ============================================
        tlt = await db.market_regime.find_one({"symbol": "TLT"})
        hyg = await db.market_regime.find_one({"symbol": "HYG"})
        lqd = await db.market_regime.find_one({"symbol": "LQD"})

        tlt_change = tlt.get("change_pct", 0) if tlt else 0
        hyg_change = hyg.get("change_pct", 0) if hyg else 0
        lqd_change = lqd.get("change_pct", 0) if lqd else 0

        credit_spread = hyg_change - lqd_change

        # 🆕 Score base 60. Credit spread positivo = risk on. TLT flight = penalità.
        bonds_score = 60 + (credit_spread * 30) - (max(0, tlt_change - 0.5) * 20)
        bonds_score = self._clamp(bonds_score, 10, 90)

        if credit_spread < -0.3 and tlt_change > 0.5:
            bonds_signal = "risk_off"
        elif credit_spread < -0.2:
            bonds_signal = "credit_stress"
        elif credit_spread > 0.2:
            bonds_signal = "risk_on"
        elif tlt_change > 0.5:
            bonds_signal = "flight_to_safety"
        else:
            bonds_signal = "neutral"

        # ============================================
        # 5C. COMMODITIES — continuo
        # ============================================
        gld = await db.market_regime.find_one({"symbol": "GLD"})
        uso = await db.market_regime.find_one({"symbol": "USO"})

        gld_change = gld.get("change_pct", 0) if gld else 0
        uso_change = uso.get("change_pct", 0) if uso else 0

        # 🆕 Growth signal: oil up, gold down = risk on.
        # commodities_score base 55, penalità se gold sale con SPY debole
        commodities_score = 55 + (uso_change * 8) - (gld_change * 5)
        if spy_return_20d < 0 and gld_change > 0.5:
            commodities_score -= 20  # gold flight
        commodities_score = self._clamp(commodities_score, 15, 90)

        if gld_change > 1 and spy_return_20d < 0:
            commodities_signal = "risk_off"
        elif gld_change > 0.5 and uso_change > 1:
            commodities_signal = "inflation"
        elif gld_change < -0.5 and uso_change > 0:
            commodities_signal = "growth"
        elif gld_change < 0:
            commodities_signal = "risk_on"
        else:
            commodities_signal = "neutral"

        # ============================================
        # 5D. BREADTH DIVERGENCE — continuo
        # ============================================
        rsp = await db.market_regime.find_one({"symbol": "RSP"})
        rsp_change = rsp.get("change_pct", 0) if rsp else 0
        spy_change = spy.get("change_pct", 0) if spy else 0

        breadth_gap = rsp_change - spy_change

        # 🆕 Gap 0 = normale (60). Positivo = broad rally (90). Negativo = narrow (30).
        breadth_div_score = self._clamp(60 + (breadth_gap * 40), 10, 95)

        if breadth_gap > 0.4:
            breadth_divergence = "broad_rally"
        elif breadth_gap > -0.2:
            breadth_divergence = "normal"
        elif breadth_gap > -0.8:
            breadth_divergence = "narrow"
        else:
            breadth_divergence = "very_narrow"

        # ============================================
        # 5E. RISK APPETITE — continuo
        # ============================================
        iwo = await db.market_regime.find_one({"symbol": "IWO"})
        eem = await db.market_regime.find_one({"symbol": "EEM"})
        iyt = await db.market_regime.find_one({"symbol": "IYT"})

        iwo_change = iwo.get("change_pct", 0) if iwo else 0
        eem_change = eem.get("change_pct", 0) if eem else 0
        iyt_change = iyt.get("change_pct", 0) if iyt else 0

        # 🆕 Score continuo: media pesata dei 3 cambio
        risk_avg = (iwo_change + eem_change + iyt_change) / 3
        risk_appetite_score = self._clamp(50 + (risk_avg * 10), 10, 95)

        if risk_avg > 1.5:
            risk_appetite = "strong"
        elif risk_avg > 0.5:
            risk_appetite = "moderate"
        elif risk_avg > -0.5:
            risk_appetite = "low"
        else:
            risk_appetite = "risk_off"

        # ============================================
        # 6. SECTOR ROTATION
        # ============================================
        sectors = await db.sectors.find().sort("composite_score", -1).to_list(20)
        sector_rankings = []
        for i, s in enumerate(sectors):
            sector_rankings.append({
                "rank": i + 1,
                "code": s.get("code", ""),
                "name": s.get("name", ""),
                "score": round(s.get("composite_score", 0), 2),
                "strength": round(s.get("strength_score", 0), 2),
                "rsi": s.get("rsi", 50),
            })

        defensive_codes = {"XLU", "XLP", "XLV"}
        top3_codes = {s["code"] for s in sector_rankings[:3]}
        defensive_in_top3 = len(defensive_codes & top3_codes)

        rotation_signal = "offensive"
        if defensive_in_top3 >= 2:
            rotation_signal = "defensive"
        elif defensive_in_top3 == 1:
            rotation_signal = "mixed"

        # ============================================
        # 7. MARKET BREADTH — continuo
        # ============================================
        assets = await db.assets.find({}, {"price": 1, "ema50": 1}).to_list(300)

        total_stocks = 0
        above_ema50 = 0
        for a in assets:
            price = a.get("price", 0)
            ema50 = a.get("ema50", 0)
            if price > 0 and ema50 > 0:
                total_stocks += 1
                if price > ema50:
                    above_ema50 += 1

        breadth_pct = round((above_ema50 / total_stocks * 100), 1) if total_stocks > 0 else 50

        # 🆕 Score continuo: 50% breadth = 50 score. 70% = 78. 30% = 22.
        breadth_score = self._clamp(breadth_pct * 1.4 - 20, 5, 95)

        breadth_healthy = params.get("breadth_healthy", 60)
        breadth_weak = params.get("breadth_weak", 40)
        breadth_critical = params.get("breadth_critical", 25)

        if breadth_pct >= breadth_healthy:
            market_breadth = "healthy"
        elif breadth_pct >= breadth_weak:
            market_breadth = "mixed"
        elif breadth_pct >= breadth_critical:
            market_breadth = "weak"
        else:
            market_breadth = "critical"

        # ============================================
        # 8. COMPOSITE REGIME CONFIDENCE
        # ============================================
        w = params
        regime_confidence = round(
            spy_trend_score * w.get("w_spy_trend", 0.18) +
            rsi_score * w.get("w_spy_rsi", 0.10) +
            vol_score * w.get("w_vix", 0.12) +
            breadth_score * w.get("w_breadth", 0.15) +
            alignment_score * w.get("w_indices_alignment", 0.08) +
            crypto_score * w.get("w_crypto", 0.04) +
            dollar_score * w.get("w_dollar", 0.08) +
            bonds_score * w.get("w_bonds", 0.10) +
            commodities_score * w.get("w_commodities", 0.05) +
            risk_appetite_score * w.get("w_risk_appetite", 0.10)
        , 1)

        if regime_confidence >= 70:
            market_regime = "BULL"
        elif regime_confidence >= 50:
            market_regime = "NEUTRAL"
        elif regime_confidence >= 30:
            market_regime = "BEAR"
        else:
            market_regime = "CRASH"

        exposure_map = {
            "BULL": w.get("bull_exposure", 1.0),
            "NEUTRAL": w.get("neutral_exposure", 0.6),
            "BEAR": w.get("bear_exposure", 0.3),
            "CRASH": w.get("crash_exposure", 0.0),
        }
        exposure_multiplier = exposure_map.get(market_regime, 0.5)

        if rotation_signal == "defensive" and market_regime == "BULL":
            market_regime = "NEUTRAL"
            exposure_multiplier = min(exposure_multiplier, 0.7)
            regime_confidence = min(regime_confidence, 65)

        # ============================================
        # BUILD MARKET CONTEXT
        # ============================================
        market_context = {
            "market_regime": market_regime,
            "regime_confidence": regime_confidence,
            "exposure_multiplier": round(exposure_multiplier, 2),
            "volatility_regime": volatility_regime,
            "rotation_signal": rotation_signal,
            "crypto_sentiment": crypto_sentiment,
            "dollar_strength": dollar_strength,
            "market_breadth": market_breadth,
            "breadth_pct": breadth_pct,
            "bonds_signal": bonds_signal,
            "commodities_signal": commodities_signal,
            "breadth_divergence": breadth_divergence,
            "risk_appetite": risk_appetite,
            "sector_rankings": sector_rankings,
            "details": {
                "spy": {"price": spy_price, "ema20": spy_ema20, "ema50": spy_ema50,
                        "rsi": spy_rsi, "return_20d": spy_return_20d,
                        "trend_score": round(spy_trend_score, 1), "rsi_score": round(rsi_score, 1)},
                "vixy": {"price": vixy_price, "score": round(vol_score, 1)},
                "indices": {"bullish": indices_bullish, "total": indices_total,
                            "score": round(alignment_score, 1)},
                "crypto": {"btc_change": btc_change, "eth_change": eth_change,
                           "score": round(crypto_score, 1)},
                "dollar": {"fxe_change": fxe_change, "uup_change": uup_change,
                           "net": dollar_net, "score": round(dollar_score, 1)},
                "breadth": {"above_ema50": above_ema50, "total": total_stocks,
                            "pct": breadth_pct, "score": round(breadth_score, 1)},
                "bonds": {"tlt_change": tlt_change, "hyg_change": hyg_change,
                          "lqd_change": lqd_change, "credit_spread": round(credit_spread, 2),
                          "signal": bonds_signal, "score": round(bonds_score, 1)},
                "commodities": {"gld_change": gld_change, "uso_change": uso_change,
                                "signal": commodities_signal, "score": round(commodities_score, 1)},
                "breadth_div": {"spy_change": spy_change, "rsp_change": rsp_change,
                                "gap": round(breadth_gap, 2), "signal": breadth_divergence,
                                "score": round(breadth_div_score, 1)},
                "risk_appetite_detail": {"iwo_change": iwo_change, "eem_change": eem_change,
                                         "iyt_change": iyt_change, "signal": risk_appetite,
                                         "score": round(risk_appetite_score, 1)},
            },
            "analyzed_at": datetime.utcnow().isoformat(),
        }

        # ============================================
        # 9. LLM REASONING
        # ============================================
        from app.services.llm_service import llm_ask, llm_available
        llm_reasoning = None
        if llm_available():
            try:
                data_summary = (
                    f"SPY: ${spy_price:.2f} (RSI {spy_rsi:.0f}, 20d return {spy_return_20d:+.1f}%)\n"
                    f"VIX/VIXY: ${vixy_price:.1f}\n"
                    f"Indices bullish: {indices_bullish}/{indices_total}\n"
                    f"Breadth: {breadth_pct:.1f}% stocks above EMA50\n"
                    f"Crypto: BTC {btc_change:+.1f}%, ETH {eth_change:+.1f}%\n"
                    f"Dollar: UUP {uup_change:+.1f}%, FXE {fxe_change:+.1f}%\n"
                    f"Bonds: TLT {tlt_change:+.1f}%, HYG {hyg_change:+.1f}%, LQD {lqd_change:+.1f}%\n"
                    f"Gold: {gld_change:+.1f}%, Oil: {uso_change:+.1f}%\n"
                    f"RSP vs SPY gap: {breadth_gap:+.2f}%\n"
                    f"Top sectors: {', '.join(s['code'] for s in sector_rankings[:3])}\n"
                    f"Rotation: {rotation_signal}\n"
                    f"Calculated regime: {market_regime} (confidence {regime_confidence}%)"
                )
                llm_reasoning = llm_ask(
                    system_prompt=(
                        "Sei un analista macro esperto di swing trading. "
                        "Analizza i dati di mercato in max 3 frasi in italiano. "
                        "Indica: 1) Regime e perché, 2) Rischio principale, 3) Suggerimento operativo. "
                        "Sii diretto, no disclaimers."
                    ),
                    user_prompt=f"Dati di mercato oggi:\n{data_summary}",
                    max_tokens=200,
                    temperature=0.3,
                    agent_name="macro_analyst",
                )
            except Exception as e:
                print(f"  LLM reasoning error: {e}")

        market_context["llm_reasoning"] = llm_reasoning

        await self.log_decision(
            decision_type="regime_assessment",
            data=market_context,
            reasoning=f"Regime={market_regime} conf={regime_confidence}",
            confidence=regime_confidence,
        )

        await db.market_context.update_one(
            {"_id": "latest"},
            {"$set": market_context},
            upsert=True,
        )

        print(f"🌍 MacroAnalyst v2.0: {market_regime} (conf={regime_confidence}, "
              f"exposure={exposure_multiplier}, breadth={breadth_pct}%)")

        return market_context

    async def learn(self) -> dict:
        db = get_db()
        params = await self.get_params()
        cutoff_old = datetime.utcnow() - timedelta(days=5)
        cutoff_max = datetime.utcnow() - timedelta(days=90)

        pending = await self._col_decisions().find({
            "agent": self.name,
            "type": "regime_assessment",
            "outcome": None,
            "created_at": {"$lte": cutoff_old, "$gte": cutoff_max},
        }).to_list(100)

        if not pending:
            return {"message": "No pending decisions to evaluate", "params": params}

        spy_now = await db.market_regime.find_one({"symbol": "SPY"})
        spy_price_now = spy_now.get("price", 0) if spy_now else 0

        correct = 0
        total = 0

        for dec in pending:
            data = dec.get("data", {})
            spy_then = data.get("details", {}).get("spy", {}).get("price", 0)
            regime_then = data.get("market_regime", "NEUTRAL")

            if spy_then <= 0 or spy_price_now <= 0:
                continue

            actual_return = ((spy_price_now - spy_then) / spy_then) * 100
            total += 1

            was_correct = False
            if regime_then == "BULL" and actual_return > 0:
                was_correct = True
            elif regime_then == "BEAR" and actual_return < -1:
                was_correct = True
            elif regime_then == "NEUTRAL" and -2 < actual_return < 3:
                was_correct = True
            elif regime_then == "CRASH" and actual_return < -3:
                was_correct = True

            if was_correct:
                correct += 1

            outcome = {
                "correct": was_correct,
                "spy_price_then": spy_then,
                "spy_price_now": spy_price_now,
                "actual_return_pct": round(actual_return, 2),
                "regime_predicted": regime_then,
            }
            await self.record_outcome(str(dec["_id"]), outcome)

        accuracy = (correct / total * 100) if total > 0 else 50

        await self.save_params(params)
        await self.save_performance({
            "accuracy": round(accuracy, 1),
            "total_evaluated": total,
        })

        return {
            "total_evaluated": total,
            "correct": correct,
            "accuracy": round(accuracy, 1),
        }
