from datetime import datetime, timedelta
from app.agents.base_agent import BaseAgent
from app.db.mongodb import get_db


class MacroAnalyst(BaseAgent):
    """
    🌍 AGENTE 1: Macro Analyst
    Studia l'economia generale, indici, settori e macroeconomia.
    Produce un MarketContext che guida tutti gli altri agenti.

    Analizza:
    - SPY/QQQ/IWM/DIA (indici USA) → market regime
    - VIXY (proxy VIX) → volatility regime
    - BTC/USD, ETH/USD → crypto sentiment (risk on/off)
    - FXE/UUP → dollar strength
    - 11 settori SPDR → sector rotation & rankings
    - Market breadth (% stock sopra EMA50)

    Output: MarketContext dict
    """

    def __init__(self):
        super().__init__(name="macro_analyst", version="1.0")

    def default_params(self) -> dict:
        return {
            # Pesi per il calcolo del regime_confidence
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
            # Soglie per VIXY (non VIX spot!)
            "vixy_high": 22,
            "vixy_extreme": 28,
            "vixy_low": 14,
            # Soglie mercato
            "breadth_healthy": 60,   # % stock sopra EMA50
            "breadth_weak": 40,
            "breadth_critical": 25,
            # Regime multiplier per esposizione
            "bull_exposure": 1.0,
            "neutral_exposure": 0.6,
            "bear_exposure": 0.3,
            "crash_exposure": 0.0,
        }

    async def analyze(self, context: dict = None) -> dict:
        """
        Analisi macro completa. Legge dati da MongoDB (gia' fetchati da data_fetcher).
        Ritorna MarketContext usato da tutti gli altri agenti.
        """
        db = get_db()
        params = await self.get_params()

        # ============================================
        # 1. MARKET REGIME (SPY-based)
        # ============================================
        spy = await db.market_regime.find_one({"symbol": "SPY"})
        spy_price = spy.get("price", 0) if spy else 0
        spy_ema20 = spy.get("ema20", 0) if spy else 0
        spy_ema50 = spy.get("ema50", 0) if spy else 0
        spy_rsi = spy.get("rsi", 50) if spy else 50
        spy_return_20d = spy.get("return_20d", 0) if spy else 0

        # Trend score SPY (0-100)
        spy_trend_score = 50
        if spy_price > spy_ema20 > spy_ema50:
            spy_trend_score = 85  # Strong uptrend
        elif spy_price > spy_ema50 > 0:
            spy_trend_score = 65  # Mild uptrend
        elif spy_price > spy_ema50 * 0.97 if spy_ema50 > 0 else False:
            spy_trend_score = 45  # Near support
        elif spy_price < spy_ema50 and spy_rsi > 35:
            spy_trend_score = 30  # Downtrend
        else:
            spy_trend_score = 15  # Strong downtrend / crash

        # RSI score (0-100): meglio se tra 40-65
        if 40 <= spy_rsi <= 65:
            rsi_score = 80
        elif 30 <= spy_rsi < 40:
            rsi_score = 50  # Oversold, possibile rimbalzo
        elif spy_rsi > 75:
            rsi_score = 35  # Overbought
        elif spy_rsi < 30:
            rsi_score = 25  # Forte oversold
        else:
            rsi_score = 60

        # ============================================
        # 2. VOLATILITY REGIME (VIXY-based, soglie calibrate)
        # ============================================
        vixy = await db.market_regime.find_one({"symbol": "VIXY"})
        vixy_price = vixy.get("price", 18) if vixy else 18

        vixy_high = params.get("vixy_high", 22)
        vixy_extreme = params.get("vixy_extreme", 28)
        vixy_low = params.get("vixy_low", 14)

        if vixy_price <= vixy_low:
            volatility_regime = "LOW"
            vol_score = 90
        elif vixy_price <= vixy_high:
            volatility_regime = "NORMAL"
            vol_score = 70
        elif vixy_price <= vixy_extreme:
            volatility_regime = "HIGH"
            vol_score = 40
        else:
            volatility_regime = "EXTREME"
            vol_score = 15

        # ============================================
        # 3. INDICES ALIGNMENT (QQQ, IWM, DIA)
        # ============================================
        indices_bullish = 0
        indices_total = 0
        for sym in ["QQQ", "IWM", "DIA"]:
            idx = await db.market_regime.find_one({"symbol": sym})
            if idx:
                indices_total += 1
                ret = idx.get("return_20d", 0) or idx.get("change_pct", 0)
                if ret > 0:
                    indices_bullish += 1

        if indices_total > 0:
            alignment_score = (indices_bullish / indices_total) * 100
        else:
            alignment_score = 50

        # ============================================
        # 4. CRYPTO SENTIMENT
        # ============================================
        btc = await db.market_regime.find_one({"symbol": "BTC/USD"})
        eth = await db.market_regime.find_one({"symbol": "ETH/USD"})
        btc_change = btc.get("change_pct", 0) if btc else 0
        eth_change = eth.get("change_pct", 0) if eth else 0

        if btc_change > 1 and eth_change > 1:
            crypto_sentiment = "strong_risk_on"
            crypto_score = 90
        elif btc_change > 0 and eth_change > 0:
            crypto_sentiment = "risk_on"
            crypto_score = 70
        elif btc_change < -2 and eth_change < -2:
            crypto_sentiment = "risk_off"
            crypto_score = 20
        elif btc_change < 0 or eth_change < 0:
            crypto_sentiment = "cautious"
            crypto_score = 40
        else:
            crypto_sentiment = "neutral"
            crypto_score = 50

        # ============================================
        # 5. DOLLAR STRENGTH (FXE=euro strength, UUP=dollar strength)
        # ============================================
        fxe = await db.market_regime.find_one({"symbol": "FXE"})
        uup = await db.market_regime.find_one({"symbol": "UUP"})
        fxe_change = fxe.get("change_pct", 0) if fxe else 0
        uup_change = uup.get("change_pct", 0) if uup else 0

        # Dollaro debole = buono per azioni USA (esportazioni)
        dollar_net = uup_change - fxe_change
        if dollar_net < -0.5:
            dollar_strength = "weak"
            dollar_score = 75
        elif dollar_net > 0.5:
            dollar_strength = "strong"
            dollar_score = 35
        else:
            dollar_strength = "neutral"
            dollar_score = 55
# ============================================
        # 5B. BONDS & CREDIT (TLT, HYG, LQD)
        # ============================================
        tlt = await db.market_regime.find_one({"symbol": "TLT"})
        hyg = await db.market_regime.find_one({"symbol": "HYG"})
        lqd = await db.market_regime.find_one({"symbol": "LQD"})

        tlt_change = tlt.get("change_pct", 0) if tlt else 0
        hyg_change = hyg.get("change_pct", 0) if hyg else 0
        lqd_change = lqd.get("change_pct", 0) if lqd else 0

        # Credit spread: HYG dropping more than LQD = credit stress
        credit_spread = hyg_change - lqd_change
        # TLT rising = flight to safety
        bonds_flight = tlt_change > 0.5

        if credit_spread < -0.5 and bonds_flight:
            bonds_signal = "risk_off"
            bonds_score = 20
        elif credit_spread < -0.3:
            bonds_signal = "credit_stress"
            bonds_score = 35
        elif credit_spread > 0.3 and tlt_change < 0:
            bonds_signal = "risk_on"
            bonds_score = 80
        elif tlt_change > 0.5:
            bonds_signal = "flight_to_safety"
            bonds_score = 40
        else:
            bonds_signal = "neutral"
            bonds_score = 60

        # ============================================
        # 5C. COMMODITIES (GLD, USO)
        # ============================================
        gld = await db.market_regime.find_one({"symbol": "GLD"})
        uso = await db.market_regime.find_one({"symbol": "USO"})

        gld_change = gld.get("change_pct", 0) if gld else 0
        uso_change = uso.get("change_pct", 0) if uso else 0

        if gld_change > 1 and spy_return_20d < 0:
            commodities_signal = "risk_off"
            commodities_score = 25
        elif gld_change > 0.5 and uso_change > 1:
            commodities_signal = "inflation"
            commodities_score = 40
        elif gld_change < -0.5 and uso_change > 0:
            commodities_signal = "growth"
            commodities_score = 75
        elif gld_change < 0:
            commodities_signal = "risk_on"
            commodities_score = 70
        else:
            commodities_signal = "neutral"
            commodities_score = 55

        # ============================================
        # 5D. BREADTH DIVERGENCE (RSP vs SPY)
        # ============================================
        rsp = await db.market_regime.find_one({"symbol": "RSP"})
        rsp_change = rsp.get("change_pct", 0) if rsp else 0
        spy_change = spy.get("change_pct", 0) if spy else 0

        breadth_gap = rsp_change - spy_change
        if breadth_gap > 0.5:
            breadth_divergence = "broad_rally"
            breadth_div_score = 85
        elif breadth_gap > -0.3:
            breadth_divergence = "normal"
            breadth_div_score = 60
        elif breadth_gap > -1.0:
            breadth_divergence = "narrow"
            breadth_div_score = 40
        else:
            breadth_divergence = "very_narrow"
            breadth_div_score = 20

        # ============================================
        # 5E. RISK APPETITE (IWO, EEM, IYT)
        # ============================================
        iwo = await db.market_regime.find_one({"symbol": "IWO"})
        eem = await db.market_regime.find_one({"symbol": "EEM"})
        iyt = await db.market_regime.find_one({"symbol": "IYT"})

        iwo_change = iwo.get("change_pct", 0) if iwo else 0
        eem_change = eem.get("change_pct", 0) if eem else 0
        iyt_change = iyt.get("change_pct", 0) if iyt else 0

        risk_signals = sum(1 for x in [iwo_change, eem_change, iyt_change] if x > 0)

        if risk_signals == 3 and iyt_change > 0.5:
            risk_appetite = "strong"
            risk_appetite_score = 90
        elif risk_signals >= 2:
            risk_appetite = "moderate"
            risk_appetite_score = 70
        elif risk_signals == 1:
            risk_appetite = "low"
            risk_appetite_score = 40
        else:
            risk_appetite = "risk_off"
            risk_appetite_score = 20
        # ============================================
        # 6. SECTOR ROTATION & RANKINGS
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

        # Rotation signal: se difensivi (XLU, XLP, XLV) in top 3 = risk off
        defensive_codes = {"XLU", "XLP", "XLV"}
        top3_codes = {s["code"] for s in sector_rankings[:3]}
        defensive_in_top3 = len(defensive_codes & top3_codes)

        rotation_signal = "offensive"
        if defensive_in_top3 >= 2:
            rotation_signal = "defensive"
        elif defensive_in_top3 == 1:
            rotation_signal = "mixed"

        # ============================================
        # 7. MARKET BREADTH (% stock sopra EMA50)
        # ============================================
        assets = await db.assets.find(
            {}, {"price": 1, "ema50": 1}
        ).to_list(300)

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
        breadth_healthy = params.get("breadth_healthy", 60)
        breadth_weak = params.get("breadth_weak", 40)
        breadth_critical = params.get("breadth_critical", 25)

        if breadth_pct >= breadth_healthy:
            market_breadth = "healthy"
            breadth_score = 85
        elif breadth_pct >= breadth_weak:
            market_breadth = "mixed"
            breadth_score = 55
        elif breadth_pct >= breadth_critical:
            market_breadth = "weak"
            breadth_score = 30
        else:
            market_breadth = "critical"
            breadth_score = 10

        # ============================================
        # 8. COMPOSITE REGIME CONFIDENCE (0-100)
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

        # Determine final regime
        if regime_confidence >= 70:
            market_regime = "BULL"
        elif regime_confidence >= 50:
            market_regime = "NEUTRAL"
        elif regime_confidence >= 30:
            market_regime = "BEAR"
        else:
            market_regime = "CRASH"

        # Exposure multiplier
        exposure_map = {
            "BULL": w.get("bull_exposure", 1.0),
            "NEUTRAL": w.get("neutral_exposure", 0.6),
            "BEAR": w.get("bear_exposure", 0.3),
            "CRASH": w.get("crash_exposure", 0.0),
        }
        exposure_multiplier = exposure_map.get(market_regime, 0.5)

        # Aggiustamento per rotation difensiva
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
            # Dettagli per debug/monitoring
            "details": {
                "spy": {"price": spy_price, "ema20": spy_ema20, "ema50": spy_ema50,
                        "rsi": spy_rsi, "return_20d": spy_return_20d,
                        "trend_score": spy_trend_score, "rsi_score": rsi_score},
                "vixy": {"price": vixy_price, "score": vol_score},
                "indices": {"bullish": indices_bullish, "total": indices_total,
                            "score": alignment_score},
                "crypto": {"btc_change": btc_change, "eth_change": eth_change,
                           "score": crypto_score},
                "dollar": {"fxe_change": fxe_change, "uup_change": uup_change,
                           "net": dollar_net, "score": dollar_score},
                "breadth": {"above_ema50": above_ema50, "total": total_stocks,
                            "pct": breadth_pct, "score": breadth_score},
                "bonds": {"tlt_change": tlt_change, "hyg_change": hyg_change,
                          "lqd_change": lqd_change, "credit_spread": round(credit_spread, 2),
                          "signal": bonds_signal, "score": bonds_score},
                "commodities": {"gld_change": gld_change, "uso_change": uso_change,
                                "signal": commodities_signal, "score": commodities_score},
                "breadth_div": {"spy_change": spy_change, "rsp_change": rsp_change,
                                "gap": round(breadth_gap, 2), "signal": breadth_divergence,
                                "score": breadth_div_score},
                "risk_appetite_detail": {"iwo_change": iwo_change, "eem_change": eem_change,
                                  "iyt_change": iyt_change, "signal": risk_appetite,
                                  "score": risk_appetite_score},
            },
            "analyzed_at": datetime.utcnow().isoformat(),
        }

# ============================================
        # 9. LLM REASONING (optional)
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
                    f"Risk signals (IWO/EEM/IYT): {risk_signals}/3 positive\n"
                    f"Top sectors: {', '.join(s['code'] for s in sector_rankings[:3])}\n"
                    f"Bottom sectors: {', '.join(s['code'] for s in sector_rankings[-3:])}\n"
                    f"Rotation: {rotation_signal}\n"
                    f"My calculated regime: {market_regime} (confidence {regime_confidence}%)"
                )
                llm_reasoning = llm_ask(
                    system_prompt=(
                        "Sei un analista macro esperto di swing trading. "
                        "Analizza i dati di mercato e fornisci una valutazione breve (max 3 frasi) in italiano. "
                        "Indica: 1) Il regime attuale e perché, 2) Il rischio principale, 3) Suggerimento operativo. "
                        "Se siamo in periodo di earnings season (gen/apr/lug/ott), menzionalo. "
                    "Sii diretto e concreto, no disclaimers."
                    ),
                    user_prompt=f"Dati di mercato oggi:\n{data_summary}",
                    max_tokens=200,
                    temperature=0.3,
                )
                if llm_reasoning:
                    print(f"  🧠 LLM: {llm_reasoning[:100]}...")
            except Exception as e:
                print(f"  LLM reasoning error: {e}")

        market_context["llm_reasoning"] = llm_reasoning
        
        # Log decision
        await self.log_decision(
            decision_type="regime_assessment",
            data=market_context,
            reasoning=f"Regime={market_regime} conf={regime_confidence} "
                      f"breadth={breadth_pct}% vol={volatility_regime} "
                      f"rotation={rotation_signal}",
            confidence=regime_confidence,
        )

        # Save to market_regime collection per retrocompatibilita'
        await db.market_context.update_one(
            {"_id": "latest"},
            {"$set": market_context},
            upsert=True,
        )

        print(f"🌍 MacroAnalyst: {market_regime} (conf={regime_confidence}, "
              f"exposure={exposure_multiplier}, breadth={breadth_pct}%)")

        return market_context

    async def learn(self) -> dict:
        """
        Learning loop del MacroAnalyst.
        Per ogni prediction passata, controlla se il mercato si e' mosso
        nella direzione prevista dopo 5 giorni.
        Aggiusta i pesi delle variabili di conseguenza.
        """
        db = get_db()
        params = await self.get_params()

        # Prendi decisioni vecchie di almeno 5 giorni e senza outcome
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

        # Per valutare, confronta SPY price al momento della decisione vs ora
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

            # Il regime era corretto?
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

        # Aggiusta i pesi se abbiamo abbastanza dati
        accuracy = (correct / total * 100) if total > 0 else 50

        if total >= self.min_decisions_to_learn:
            evaluated = await self.evaluate_past_decisions(lookback_days=60)
            correct_weighted = sum(d["weight"] for d in evaluated
                                  if d.get("outcome", {}).get("correct"))
            total_weighted = sum(d["weight"] for d in evaluated) or 1

            weighted_accuracy = (correct_weighted / total_weighted) * 100

            # Se breadth era il fattore piu' correlato con le previsioni corrette
            # aumenta il suo peso, altrimenti diminuiscilo
            # (logica semplificata - in futuro si puo' usare gradient descent)
            adjustment = 0.02 if weighted_accuracy > 60 else -0.01

            # Mantieni i pesi normalizzati e tra 0.05 e 0.35
            for key in ["w_spy_trend", "w_spy_rsi", "w_vix", "w_breadth",
                        "w_indices_alignment", "w_crypto", "w_dollar"]:
                current = params.get(key, 0.15)
                params[key] = round(max(0.05, min(0.35, current)), 3)

            # Se accuracy bassa, aumenta peso del breadth (indicatore leading)
            if weighted_accuracy < 50:
                params["w_breadth"] = min(0.30, params.get("w_breadth", 0.20) + 0.02)
                params["w_spy_trend"] = min(0.30, params.get("w_spy_trend", 0.25) + 0.01)

        learn_result = {
            "total_evaluated": total,
            "correct": correct,
            "accuracy": round(accuracy, 1),
            "params_updated": params,
        }

        await self.save_params(params)
        await self.save_performance({
            "accuracy": round(accuracy, 1),
            "total_evaluated": total,
        })

        print(f"🌍 MacroAnalyst LEARN: {correct}/{total} correct ({accuracy:.1f}%)")
        return learn_result
