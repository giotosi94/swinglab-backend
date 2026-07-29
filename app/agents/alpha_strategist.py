from datetime import datetime
from app.agents.base_agent import BaseAgent
from app.db.mongodb import get_db


class AlphaStrategist(BaseAgent):
    """
    🎯 AGENTE 2: Alpha Strategist v2.0 — "Lo Stock Picker with ML"
    Seleziona le migliori opportunita' di acquisto e identifica segnali di vendita.
    Usa il MarketContext prodotto dal MacroAnalyst per contestualizzare le decisioni.
    
    v2.0 — 🆕 ML Integration:
    - Factor 14: WIN/LOSS XGBoost predictor
    - Factor 15: Trend Predictor (5d UP/FLAT/DOWN)
    
    Input: market_context, positions, ml_predictions, trend_predictions
    Output: buy_candidates[], sell_signals[], analysis_summary
    """

    def __init__(self):
        super().__init__(name="alpha_strategist", version="2.0")
        # v2.0 — Confluence max score teorico (13 factors + 2 ML)
        # Original: 15.0 (13 factors)
        # + Factor 14 (ML WIN/LOSS): 2.5 max
        # + Factor 15 (Trend Predictor): 2.0 max
        # = 19.5 total
        # v2.1 — +Factor 16 MTF Weekly Alignment (2.5 max) = 22.0
        self.MAX_RAW_CONFLUENCE = 22.0

    def default_params(self) -> dict:
        return {
            # Confluenze — pesi dei fattori (moltiplicatori, default 1.0)
            "factor_weights": {
                "poc_proximity": 1.0,
                "bullish_patterns": 1.0,
                "rsi_sweet_spot": 1.0,
                "macd_positive": 1.0,
                "ema_alignment": 1.0,
                "relative_volume": 1.0,
                "sector_rank": 1.0,
                "wyckoff_signal": 1.0,
                "accumulation": 1.0,
                "fvg_support": 1.0,
                "range_position": 1.0,
                "daily_change": 1.0,
                "near_high": 1.0,
                # 🆕 v2.0 — ML factors (pesi bassi perché modelli poco affidabili)
                "ml_winloss": 0.8,       # WIN/LOSS accuracy 46.7% → peso ridotto
                "trend_predictor": 0.7,   # Trend accuracy 50.7% → peso ridotto
                "mtf_alignment": 1.0,     # 🆕 v2.1 MTF Weekly trend filter
            },
            # Filtri
            "min_confluence": 35,
            "max_rsi_entry": 68,
            "min_rsi_entry": 25,
            "min_price": 2.0,
            "max_relative_volume": 3.0,
            "max_per_sector": 2,
            # Setup preferences
            "best_setups": ["pullback_to_poc", "ema_bounce", "breakout",
                            "oversold_reversal"],
            "worst_setups": [],
            "weak_sectors": [],
            # Sell thresholds
            "sell_rsi_extreme": 78,
            "sell_score_collapsed": 20,
            "sell_min_pnl_for_rsi_sell": 3.0,
            # 🆕 v2.0 — ML thresholds
            "ml_winloss_threshold_strong": 0.75,   # WIN score >75% = forte
            "ml_winloss_threshold_medium": 0.60,   # WIN score >60% = medio
            "trend_confidence_threshold_strong": 0.60,  # UP prob >60% = forte
            "trend_confidence_threshold_medium": 0.50,  # UP prob >50% = medio
            # 🆕 v2.0 — Sell signal ML
            "sell_ml_loss_threshold": 0.30,  # se ML dice WIN score <30% + in perdita → sell
        }

    async def _load_ml_predictions(self, db, assets: list, market_context: dict) -> dict:
        """
        🆕 v2.0 — Carica ML predictions chiamando i modelli direttamente.
        
        I modelli (WIN/LOSS XGBoost + Trend Predictor) calcolano le predizioni
        on-the-fly. Non ci sono collection dedicate in MongoDB.
        
        Ritorna un dict {ticker: {ml_score, ml_prediction, trend_prediction, ...}}
        """
        ml_map = {}
        
        # ============================================
        # 1. Load WIN/LOSS predictions dal modello
        # ============================================
        try:
            from app.ml.model import ml_model
            predictions = await ml_model.predict_batch(assets, market_context)
            
            for ticker, pred in predictions.items():
                if pred and pred.get("ml_score") is not None:
                    ml_map[ticker] = {
                        "ml_score": pred.get("ml_score", 0),
                        "ml_prediction": pred.get("prediction", "unknown"),
                        "ml_confidence": pred.get("confidence", 0),
                    }
            print(f"  📊 ML WIN/LOSS: {len(ml_map)} predictions loaded")
        except Exception as e:
            print(f"  ⚠️ ml_model predict_batch error: {e}")
        
        # ============================================
        # 2. Load Trend predictions dal modello
        # ============================================
        try:
            from app.ml.trend_model import trend_predictor
            trend_results = await trend_predictor.predict_all()
            
            for pred in trend_results:
                ticker = pred.get("ticker")
                if not ticker:
                    continue
                
                # Se ticker non era in WIN/LOSS, inizializza
                if ticker not in ml_map:
                    ml_map[ticker] = {}
                
                # Aggiungi campi trend
                ml_map[ticker].update({
                    "trend_prediction": pred.get("prediction", "FLAT"),
                    "trend_up_prob": pred.get("up_prob", 0),
                    "trend_flat_prob": pred.get("flat_prob", 0),
                    "trend_down_prob": pred.get("down_prob", 0),
                    "trend_confidence": pred.get("confidence", 0),
                })
            print(f"  📊 Trend Predictor: {len(trend_results)} predictions loaded")
        except Exception as e:
            print(f"  ⚠️ trend_predictor predict_all error: {e}")
        
        return ml_map

    def _calc_confluence(self, asset: dict, market_ctx: dict, params: dict, ml_data: dict = None) -> dict:
        """
        Calcola il confluence score multi-fattore per un singolo asset.
        v2.0: aggiunge Factor 14 (ML WIN/LOSS) e Factor 15 (Trend Predictor).
        """
        fw = params.get("factor_weights", {})
        factors = []
        raw_score = 0

        price = asset.get("price", 0)
        rsi = asset.get("rsi", 50)
        poc = asset.get("poc_price")
        va_high = asset.get("value_area_high")
        va_low = asset.get("value_area_low")
        ema10 = asset.get("ema10", 0)
        ema20 = asset.get("ema20", 0)
        ema50 = asset.get("ema50", 0)
        rel_vol = asset.get("relative_volume", 1)
        macd_hist = asset.get("macd", {}).get("histogram", 0)
        patterns = asset.get("candlestick_patterns", [])
        wyckoff = asset.get("wyckoff", {})
        accum = asset.get("accumulation", {})
        fvgs = asset.get("fvg", [])
        sector = asset.get("sector_code", "")
        change_pct = asset.get("change_pct", 0)
        pct_from_high = asset.get("pct_from_high", -50)
        range_pos = asset.get("range_position", 50)

        sector_rankings = market_ctx.get("sector_rankings", [])
        sector_codes = [s["code"] for s in sector_rankings]

        # --- 1. POC Proximity ---
        w = fw.get("poc_proximity", 1.0)
        if poc and price and poc > 0:
            dist = abs(price - poc) / poc * 100
            if dist <= 2:
                pts = 2.0 * w
                factors.append({"name": "POC", "pts": pts, "max": 2.0, "detail": f"{dist:.1f}%", "pass": True})
            elif dist <= 5:
                pts = 1.0 * w
                factors.append({"name": "POC", "pts": pts, "max": 2.0, "detail": f"{dist:.1f}%", "pass": True})
            else:
                pts = 0
                factors.append({"name": "POC", "pts": 0, "max": 2.0, "detail": f"{dist:.1f}%", "pass": False})
        else:
            pts = 0
            factors.append({"name": "POC", "pts": 0, "max": 2.0, "detail": "N/A", "pass": False})
        raw_score += pts

        # --- 2. Bullish Patterns ---
        w = fw.get("bullish_patterns", 1.0)
        bullish = [p for p in patterns if p.get("type") == "bullish"]
        if bullish:
            strong = [p for p in bullish if p.get("strength") == "strong"]
            pts = (2.0 if strong else 1.5) * w
            names = ", ".join(p["name"] for p in bullish[:3])
            factors.append({"name": "Patterns", "pts": pts, "max": 2.0, "detail": names, "pass": True})
        else:
            pts = 0
            factors.append({"name": "Patterns", "pts": 0, "max": 2.0, "detail": "None", "pass": False})
        raw_score += pts

        # --- 3. RSI Sweet Spot ---
        w = fw.get("rsi_sweet_spot", 1.0)
        if 40 <= rsi <= 60:
            pts = 1.0 * w
            factors.append({"name": "RSI", "pts": pts, "max": 1.0, "detail": f"{rsi:.0f}", "pass": True})
        elif 30 <= rsi < 40:
            pts = 0.5 * w
            factors.append({"name": "RSI", "pts": pts, "max": 1.0, "detail": f"{rsi:.0f} (reversal?)", "pass": True})
        else:
            pts = 0
            factors.append({"name": "RSI", "pts": 0, "max": 1.0, "detail": f"{rsi:.0f}", "pass": False})
        raw_score += pts

        # --- 4. MACD Positive ---
        w = fw.get("macd_positive", 1.0)
        if macd_hist > 0:
            pts = 1.0 * w
            factors.append({"name": "MACD", "pts": pts, "max": 1.0, "detail": "+", "pass": True})
        else:
            pts = 0
            factors.append({"name": "MACD", "pts": 0, "max": 1.0, "detail": "-", "pass": False})
        raw_score += pts

        # --- 5. EMA Alignment ---
        w = fw.get("ema_alignment", 1.0)
        if price > ema10 > ema20 > ema50 and ema50 > 0:
            pts = 1.5 * w
            factors.append({"name": "EMA", "pts": pts, "max": 1.5, "detail": "Full align", "pass": True})
        elif price > ema20 > ema50 and ema50 > 0:
            pts = 0.75 * w
            factors.append({"name": "EMA", "pts": pts, "max": 1.5, "detail": "Partial", "pass": True})
        else:
            pts = 0
            factors.append({"name": "EMA", "pts": 0, "max": 1.5, "detail": "No", "pass": False})
        raw_score += pts

        # --- 6. Relative Volume ---
        w = fw.get("relative_volume", 1.0)
        if rel_vol >= 1.5:
            pts = 1.0 * w
            factors.append({"name": "Volume", "pts": pts, "max": 1.0, "detail": f"{rel_vol:.1f}x", "pass": True})
        else:
            pts = 0
            factors.append({"name": "Volume", "pts": 0, "max": 1.0, "detail": f"{rel_vol:.1f}x", "pass": False})
        raw_score += pts

        # --- 7. Sector Rank ---
        w = fw.get("sector_rank", 1.0)
        if sector in sector_codes:
            rank = sector_codes.index(sector) + 1
            if rank <= 3:
                pts = 1.5 * w
                factors.append({"name": "Sector", "pts": pts, "max": 1.5, "detail": f"#{rank}", "pass": True})
            elif rank <= 5:
                pts = 1.0 * w
                factors.append({"name": "Sector", "pts": pts, "max": 1.5, "detail": f"#{rank}", "pass": True})
            else:
                pts = 0
                factors.append({"name": "Sector", "pts": 0, "max": 1.5, "detail": f"#{rank}", "pass": False})
        else:
            pts = 0
            factors.append({"name": "Sector", "pts": 0, "max": 1.5, "detail": "N/A", "pass": False})
        raw_score += pts

        # --- 8. Wyckoff Signal ---
        w = fw.get("wyckoff_signal", 1.0)
        wy_signal = wyckoff.get("signal", "neutral")
        if wy_signal == "strong_bullish":
            pts = 2.0 * w
            factors.append({"name": "Wyckoff", "pts": pts, "max": 2.0, "detail": "Spring!", "pass": True})
        elif wy_signal in ("bullish", "bullish_soon"):
            pts = 1.5 * w
            factors.append({"name": "Wyckoff", "pts": pts, "max": 2.0, "detail": wy_signal, "pass": True})
        elif wy_signal in ("bearish", "bearish_soon"):
            pts = -2.0 * w
            factors.append({"name": "Wyckoff", "pts": pts, "max": 2.0, "detail": wy_signal, "pass": False})
        else:
            pts = 0
            factors.append({"name": "Wyckoff", "pts": 0, "max": 2.0, "detail": wy_signal, "pass": False})
        raw_score += pts

        # --- 9. Accumulation Score ---
        w = fw.get("accumulation", 1.0)
        accum_score = accum.get("score", 0)
        if accum_score >= 70:
            pts = 1.0 * w
            factors.append({"name": "Accum", "pts": pts, "max": 1.0, "detail": f"{accum_score}", "pass": True})
        elif accum_score >= 40:
            pts = 0.5 * w
            factors.append({"name": "Accum", "pts": pts, "max": 1.0, "detail": f"{accum_score}", "pass": True})
        else:
            pts = 0
            factors.append({"name": "Accum", "pts": 0, "max": 1.0, "detail": f"{accum_score}", "pass": False})
        raw_score += pts

        # --- 10. FVG Support ---
        w = fw.get("fvg_support", 1.0)
        bullish_fvgs = [f for f in fvgs if f.get("type") == "bullish" and not f.get("filled")]
        if bullish_fvgs:
            pts = 0.5 * w
            factors.append({"name": "FVG", "pts": pts, "max": 0.5, "detail": f"{len(bullish_fvgs)} gap(s)", "pass": True})
        else:
            pts = 0
            factors.append({"name": "FVG", "pts": 0, "max": 0.5, "detail": "None", "pass": False})
        raw_score += pts

        # --- 11. Range Position ---
        w = fw.get("range_position", 1.0)
        if range_pos < 30:
            pts = 0.5 * w
            factors.append({"name": "Range", "pts": pts, "max": 0.5, "detail": f"{range_pos:.0f}%", "pass": True})
        else:
            pts = 0
            factors.append({"name": "Range", "pts": 0, "max": 0.5, "detail": f"{range_pos:.0f}%", "pass": False})
        raw_score += pts

        # --- 12. Daily Change ---
        w = fw.get("daily_change", 1.0)
        if 0 < change_pct <= 5:
            pts = 0.5 * w
            factors.append({"name": "Change", "pts": pts, "max": 0.5, "detail": f"+{change_pct:.1f}%", "pass": True})
        else:
            pts = 0
            factors.append({"name": "Change", "pts": 0, "max": 0.5, "detail": f"{change_pct:.1f}%", "pass": False})
        raw_score += pts

        # --- 13. Near 52w High ---
        w = fw.get("near_high", 1.0)
        if pct_from_high is not None and pct_from_high >= -10:
            pts = 0.5 * w
            factors.append({"name": "52wHigh", "pts": pts, "max": 0.5, "detail": f"{pct_from_high:.1f}%", "pass": True})
        else:
            pts = 0
            factors.append({"name": "52wHigh", "pts": 0, "max": 0.5, "detail": f"{pct_from_high}%", "pass": False})
        raw_score += pts

        # ============================================
        # 🆕 v2.0 — ML FACTORS (Factor 14 + 15)
        # ============================================
        
        # --- 14. ML WIN/LOSS Predictor (XGBoost) ---
        w = fw.get("ml_winloss", 0.8)
        ml_threshold_strong = params.get("ml_winloss_threshold_strong", 0.75)
        ml_threshold_medium = params.get("ml_winloss_threshold_medium", 0.60)
        
        if ml_data:
            ml_score_raw = ml_data.get("ml_score", 0)
            # ml_score è 0-100 nell'endpoint, normalizziamo a 0-1
            ml_score = ml_score_raw / 100 if ml_score_raw > 1 else ml_score_raw
            ml_prediction = ml_data.get("ml_prediction", "unknown")
            
            if ml_prediction == "WIN" and ml_score >= ml_threshold_strong:
                pts = 2.5 * w  # forte segnale ML
                factors.append({"name": "ML", "pts": pts, "max": 2.5, "detail": f"WIN {ml_score*100:.0f}% (strong)", "pass": True})
            elif ml_prediction == "WIN" and ml_score >= ml_threshold_medium:
                pts = 1.5 * w  # medio
                factors.append({"name": "ML", "pts": pts, "max": 2.5, "detail": f"WIN {ml_score*100:.0f}%", "pass": True})
            elif ml_prediction == "LOSS" and ml_score >= ml_threshold_medium:
                pts = -1.0 * w  # penalizza (ML dice LOSS)
                factors.append({"name": "ML", "pts": pts, "max": 2.5, "detail": f"LOSS {ml_score*100:.0f}%", "pass": False})
            else:
                pts = 0
                factors.append({"name": "ML", "pts": 0, "max": 2.5, "detail": "no signal", "pass": False})
        else:
            pts = 0
            factors.append({"name": "ML", "pts": 0, "max": 2.5, "detail": "N/A", "pass": False})
        raw_score += pts

        # --- 15. Trend Predictor (5d UP/FLAT/DOWN) ---
        w = fw.get("trend_predictor", 0.7)
        trend_strong = params.get("trend_confidence_threshold_strong", 0.60)
        trend_medium = params.get("trend_confidence_threshold_medium", 0.50)
        
        if ml_data:
            trend_pred = ml_data.get("trend_prediction", "FLAT")
            up_prob_raw = ml_data.get("trend_up_prob", 0)
            up_prob = up_prob_raw / 100 if up_prob_raw > 1 else up_prob_raw
            down_prob_raw = ml_data.get("trend_down_prob", 0)
            down_prob = down_prob_raw / 100 if down_prob_raw > 1 else down_prob_raw
            
            if trend_pred == "UP" and up_prob >= trend_strong:
                pts = 2.0 * w  # forte trend UP
                factors.append({"name": "Trend", "pts": pts, "max": 2.0, "detail": f"UP {up_prob*100:.0f}%", "pass": True})
            elif trend_pred == "UP" and up_prob >= trend_medium:
                pts = 1.0 * w  # medio trend UP
                factors.append({"name": "Trend", "pts": pts, "max": 2.0, "detail": f"UP {up_prob*100:.0f}%", "pass": True})
            elif trend_pred == "DOWN" and down_prob >= trend_medium:
                pts = -1.5 * w  # penalizza fortemente
                factors.append({"name": "Trend", "pts": pts, "max": 2.0, "detail": f"DOWN {down_prob*100:.0f}%", "pass": False})
            elif trend_pred == "FLAT":
                pts = 0  # neutrale
                factors.append({"name": "Trend", "pts": 0, "max": 2.0, "detail": "FLAT", "pass": False})
            else:
                pts = 0
                factors.append({"name": "Trend", "pts": 0, "max": 2.0, "detail": trend_pred, "pass": False})
        else:
            pts = 0
            factors.append({"name": "Trend", "pts": 0, "max": 2.0, "detail": "N/A", "pass": False})
        raw_score += pts

        # --- 16. 🆕 v2.1 MTF Weekly Alignment ---
        w = fw.get("mtf_alignment", 1.0)
        mtf = asset.get("mtf", {})
        wtrend = mtf.get("weekly_trend", "UNKNOWN")
        wslope = mtf.get("weekly_ema20_slope", "flat")
        if wtrend == "BULL" and wslope == "rising":
            pts = 2.5 * w
            factors.append({"name": "MTF", "pts": pts, "max": 2.5, "detail": "Weekly BULL rising", "pass": True})
        elif wtrend == "BULL":
            pts = 1.5 * w
            factors.append({"name": "MTF", "pts": pts, "max": 2.5, "detail": "Weekly BULL", "pass": True})
        elif wtrend == "NEUTRAL":
            pts = 0.5 * w
            factors.append({"name": "MTF", "pts": pts, "max": 2.5, "detail": "Weekly NEUTRAL", "pass": True})
        elif wtrend == "BEAR":
            pts = -2.0 * w
            factors.append({"name": "MTF", "pts": pts, "max": 2.5, "detail": "Weekly BEAR", "pass": False})
        else:
            pts = 0
            factors.append({"name": "MTF", "pts": 0, "max": 2.5, "detail": "N/A", "pass": False})
        raw_score += pts

        # Normalize to 0-100
        normalized = round(max(0, min(100, (raw_score / self.MAX_RAW_CONFLUENCE) * 100)), 1)

        # Count passing factors
        passing = sum(1 for f in factors if f["pass"])

        return {
            "raw_score": round(raw_score, 2),
            "score": normalized,
            "factors": factors,
            "passing_factors": passing,
            "total_factors": len(factors),
            # 🆕 v2.0 — ML contribution breakdown
            "ml_contribution": round(sum(f["pts"] for f in factors if f["name"] in ("ML", "Trend")), 2),
            "rules_contribution": round(sum(f["pts"] for f in factors if f["name"] not in ("ML", "Trend")), 2),
        }

    async def _check_sells(self, positions: list, assets_map: dict,
                           market_ctx: dict, params: dict, ml_map: dict = None) -> list:
        """
        Verifica se le posizioni aperte vanno vendute.
        v2.0: aggiunge sell signal se ML dice LOSS + in perdita.
        """
        sell_signals = []

        for p in positions:
            symbol = p.get("symbol")
            asset = assets_map.get(symbol)
            if not asset:
                continue

            current_price = float(p.get("current_price", 0))
            entry_price = float(p.get("avg_entry_price", 0))
            pnl_pct = float(p.get("unrealized_plpc", 0)) * 100
            rsi = asset.get("rsi", 50)
            setup_score = asset.get("setup_score", 50)
            patterns = asset.get("candlestick_patterns", [])
            wyckoff = asset.get("wyckoff", {})
            regime = market_ctx.get("market_regime", "NEUTRAL")

            sell_reason = None
            urgency = "normal"

            rsi_threshold = params.get("sell_rsi_extreme", 78)
            min_pnl_rsi = params.get("sell_min_pnl_for_rsi_sell", 3.0)
            score_threshold = params.get("sell_score_collapsed", 20)
            ml_loss_threshold = params.get("sell_ml_loss_threshold", 0.30)

            # 1. RSI Extreme + in profitto
            if rsi > rsi_threshold and pnl_pct > min_pnl_rsi:
                sell_reason = "RSI_EXTREME"
                urgency = "high"

            # 2. Score collapsed + in perdita
            elif setup_score < score_threshold and pnl_pct < -2:
                sell_reason = "SCORE_COLLAPSED"
                urgency = "high"

            # 3. Bearish pattern + in perdita
            elif pnl_pct < -1:
                bearish = [pat for pat in patterns
                          if pat.get("type") == "bearish" and pat.get("strength") == "strong"]
                if bearish:
                    sell_reason = "BEARISH_PATTERN"
                    urgency = "normal"

            # 4. Wyckoff distribution + in perdita
            elif wyckoff.get("phase") in ("distribution", "markdown") and pnl_pct < 0:
                sell_reason = "WYCKOFF_BEARISH"
                urgency = "normal"

            # 5. Market crash — vendi tutto in perdita
            elif regime == "CRASH" and pnl_pct < -1:
                sell_reason = "MARKET_CRASH"
                urgency = "critical"

            # 🆕 6. v2.0 — ML LOSS signal + in perdita significativa
            elif ml_map and symbol in ml_map:
                ml_data = ml_map[symbol]
                ml_score_raw = ml_data.get("ml_score", 100)
                ml_score = ml_score_raw / 100 if ml_score_raw > 1 else ml_score_raw
                ml_pred = ml_data.get("ml_prediction", "unknown")
                trend_pred = ml_data.get("trend_prediction", "UP")
                
                if ml_pred == "LOSS" and pnl_pct < -1.5:
                    sell_reason = "ML_LOSS_SIGNAL"
                    urgency = "high"
                elif trend_pred == "DOWN" and pnl_pct < -1.0:
                    sell_reason = "TREND_DOWN_ML"
                    urgency = "normal"

            if sell_reason:
                sell_signals.append({
                    "ticker": symbol,
                    "reason": sell_reason,
                    "urgency": urgency,
                    "pnl_pct": round(pnl_pct, 2),
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "rsi": rsi,
                    "setup_score": setup_score,
                })

        return sell_signals

    async def _enrich_with_sentiment(self, candidates: list) -> list:
        """
        🆕 SentimentAgent — arricchisce i top candidati con news sentiment.
        Solo sui top (efficiente). Aggiusta confluence + flag earnings.
        """
        from app.services.news_service import get_stock_news_with_sentiment
        for c in candidates:
            try:
                data = await get_stock_news_with_sentiment(c["ticker"])
                raw = (data.get("sentiment") or "").upper()
                c["news_count"] = data.get("news_count", 0)

                sent = "NEUTRO"
                if "SENTIMENT: POSITIVO" in raw or "\nPOSITIVO" in raw:
                    sent = "POSITIVO"
                elif "SENTIMENT: NEGATIVO" in raw or "\nNEGATIVO" in raw:
                    sent = "NEGATIVO"

                # Rileva earnings imminenti — legge il campo dedicato dell'LLM
                earnings_soon = ("EARNINGS_IMMINENTI: SI" in raw or
                                 "EARNINGS_IMMINENTI:SI" in raw)

                adj = 0
                if sent == "POSITIVO":
                    adj += 5
                elif sent == "NEGATIVO":
                    adj -= 8
                if earnings_soon:
                    adj -= 6  # rischio gap notturno

                c["sentiment"] = sent
                c["earnings_soon"] = earnings_soon
                c["sentiment_adj"] = adj
                c["confluence"] = round(max(0, c["confluence"] + adj), 1)
            except Exception as e:
                c["sentiment"] = "N/A"
                c["sentiment_adj"] = 0
        # Ri-ordina con il sentiment incluso
        candidates.sort(key=lambda x: x["confluence"], reverse=True)
        return candidates

    async def analyze(self, context: dict) -> dict:
        db = get_db()
        params = await self.get_params()
        market_ctx = context.get("market_context", {})
        positions = context.get("positions", [])

        # 🆕 v2.0 — Load ML predictions (calcolate on-the-fly dai modelli)
        # NOTA: dobbiamo passare gli assets che caricheremo tra poco
        # Per efficienza, carichiamo prima gli assets
        assets_for_ml = await db.assets.find({}, {
            "price_history": 0, "vp_distribution": 0, "multi_tf_vp": 0
        }).to_list(300)
        
        ml_map = await self._load_ml_predictions(db, assets_for_ml, market_ctx)
        print(f"  📊 ML data loaded: {len(ml_map)} tickers with predictions")

        # Riutilizza assets già caricati per ML (ottimizzazione)
        assets = assets_for_ml

        if not assets:
            return {"buy_candidates": [], "sell_signals": [], "error": "No assets data"}

        assets_map = {a["ticker"]: a for a in assets}
        open_tickers = [p.get("symbol") for p in positions]

        # Conta settori delle posizioni aperte
        open_sectors = []
        for t in open_tickers:
            a = assets_map.get(t)
            if a:
                open_sectors.append(a.get("sector_code", ""))

        # ============================================
        # SELL SIGNALS (con ML)
        # ============================================
        sell_signals = await self._check_sells(positions, assets_map, market_ctx, params, ml_map)

        # ============================================
        # BUY CANDIDATES
        # ============================================
        min_conf = params.get("min_confluence", 35)
        max_rsi = params.get("max_rsi_entry", 68)
        min_rsi = params.get("min_rsi_entry", 25)
        min_price_val = params.get("min_price", 2.0)
        max_rv = params.get("max_relative_volume", 3.0)
        max_per_sector = params.get("max_per_sector", 2)
        best_setups = params.get("best_setups", [])
        worst_setups = params.get("worst_setups", [])
        weak_sectors = params.get("weak_sectors", [])

        candidates = []
        skipped_reasons = {"low_confluence": 0, "rsi_filter": 0, "setup_filter": 0,
                          "sector_full": 0, "price_filter": 0, "volume_filter": 0,
                          "already_open": 0}

        for a in assets:
            ticker = a.get("ticker", "")

            if ticker in open_tickers:
                skipped_reasons["already_open"] += 1
                continue

            price = a.get("price", 0)
            rsi = a.get("rsi", 50)
            stype = a.get("setup_type", "neutral")
            sector = a.get("sector_code", "")
            rel_vol = a.get("relative_volume", 1)
            va_high = a.get("value_area_high")
            va_low = a.get("value_area_low")

            # Filtri base
            if price < min_price_val:
                skipped_reasons["price_filter"] += 1
                continue
            if rsi > max_rsi or rsi < min_rsi:
                skipped_reasons["rsi_filter"] += 1
                continue

            # Volume filter smart
            change_pct = a.get("change_pct", 0)
            if rel_vol >= max_rv:
                if rel_vol < 5.0 and 2.0 <= change_pct <= 8.0:
                    pass
                elif rel_vol >= 5.0 or change_pct > 8.0:
                    skipped_reasons["volume_filter"] += 1
                    continue
                elif change_pct < 2.0:
                    skipped_reasons["volume_filter"] += 1
                    continue
                else:
                    skipped_reasons["volume_filter"] += 1
                    continue

            if best_setups and stype not in best_setups:
                skipped_reasons["setup_filter"] += 1
                continue
            if stype in worst_setups:
                skipped_reasons["setup_filter"] += 1
                continue
            sector_count = open_sectors.count(sector)
            if sector_count >= max_per_sector:
                skipped_reasons["sector_full"] += 1
                continue

            sector_penalty = -5 if sector in weak_sectors else 0

            # 🆕 v2.0 — Passa ml_data al calc_confluence
            ml_data = ml_map.get(ticker)
            conf = self._calc_confluence(a, market_ctx, params, ml_data)
            conf_score = conf["score"] + sector_penalty

            if conf_score < min_conf:
                skipped_reasons["low_confluence"] += 1
                continue

            # 🔧 v1.2 — Target e stop loss safety
            if va_low and 0 < va_low < price:
                raw_stop = va_low
            else:
                raw_stop = round(price * 0.96, 2)
            if raw_stop >= price:
                raw_stop = round(price * 0.96, 2)
            min_stop = round(price * 0.92, 2)
            stop_loss = max(raw_stop, min_stop)
            if stop_loss >= price:
                stop_loss = round(price * 0.96, 2)

            if va_high and va_high > price:
                raw_target = va_high
            else:
                raw_target = round(price * 1.08, 2)
            if raw_target <= price:
                raw_target = round(price * 1.08, 2)
            min_target = round(price * 1.06, 2)
            target_price = max(raw_target, min_target)

            risk = abs(price - stop_loss)
            reward = abs(target_price - price)
            rr_ratio = round(reward / risk, 2) if risk > 0 else 0

            candidates.append({
                "ticker": ticker,
                "price": round(price, 2),
                "confluence": conf_score,
                "confluence_detail": conf,
                "setup_score": a.get("setup_score", 0),
                "setup_type": stype,
                "sector": sector,
                "rsi": rsi,
                "relative_volume": rel_vol,
                "stop_loss": round(stop_loss, 2),
                "target_price": round(target_price, 2),
                "risk_reward": rr_ratio,
                "wyckoff_phase": a.get("wyckoff", {}).get("phase", "unknown"),
                # 🆕 v2.0 — ML info
                "ml_prediction": ml_data.get("ml_prediction", "N/A") if ml_data else "N/A",
                "ml_score": ml_data.get("ml_score", 0) if ml_data else 0,
                "trend_prediction": ml_data.get("trend_prediction", "N/A") if ml_data else "N/A",
                "trend_up_prob": ml_data.get("trend_up_prob", 0) if ml_data else 0,
                "weekly_trend": a.get("mtf", {}).get("weekly_trend", "UNKNOWN"),
            })

        candidates.sort(key=lambda x: x["confluence"], reverse=True)
        top_candidates = candidates[:10]

        # 🆕 SentimentAgent — arricchisce i top con news sentiment + earnings
        top_candidates = await self._enrich_with_sentiment(top_candidates)

        # ============================================
        # LLM REASONING per top candidates
        # ============================================
        from app.services.llm_service import llm_ask, llm_available
        if llm_available() and top_candidates:
            for candidate in top_candidates[:5]:
                try:
                    factors_pass = [f["name"] for f in candidate.get("confluence_detail", {}).get("factors", []) if f.get("pass")]
                    factors_fail = [f["name"] for f in candidate.get("confluence_detail", {}).get("factors", []) if not f.get("pass")]

                    macro_reasoning = ""
                    try:
                        from app.agents.shared_brain import brain
                        brain_market = await brain.get_market()
                        if brain_market.get("llm_reasoning"):
                            macro_reasoning = f"\nAnalisi Macro: {brain_market['llm_reasoning'][:200]}"
                    except:
                        pass
                    
                    # 🆕 v2.0 — Include ML data in prompt
                    ml_info = ""
                    if candidate.get("ml_prediction") != "N/A":
                        ml_info = f"\nML WIN/LOSS: {candidate['ml_prediction']} ({candidate.get('ml_score', 0):.0f}%)"
                        ml_info += f"\nTrend 5d: {candidate.get('trend_prediction', 'N/A')} (up_prob {candidate.get('trend_up_prob', 0):.0f}%)"
                    
                    stock_data = (
                        f"Ticker: {candidate['ticker']} ({candidate.get('sector', '')})\n"
                        f"Prezzo: ${candidate['price']}\n"
                        f"Setup: {candidate.get('setup_type', 'unknown')}\n"
                        f"Confluence: {candidate.get('confluence', 0)}/100\n"
                        f"RSI: {candidate.get('rsi', 50)}\n"
                        f"R/R: {candidate.get('risk_reward', 0)}\n"
                        f"Target: ${candidate.get('target_price', 0)} | Stop: ${candidate.get('stop_loss', 0)}\n"
                        f"Wyckoff: {candidate.get('wyckoff_phase', 'unknown')}\n"
                        f"Fattori positivi: {', '.join(factors_pass)}\n"
                        f"Fattori negativi: {', '.join(factors_fail)}\n"
                        f"Regime mercato: {market_ctx.get('market_regime', 'NEUTRAL')}"
                        f"{ml_info}"
                    )

                    earnings_context = ""
                    try:
                        from app.services.news_service import fetch_news
                        news = await fetch_news(candidate["ticker"], limit=3)
                        if news:
                            headlines = "; ".join([n["headline"] for n in news])
                            earnings_context = f"\nNews recenti: {headlines}"
                    except:
                        pass

                    analysis = llm_ask(
                        system_prompt=(
                            "Sei un analista di swing trading esperto. "
                            "Valuta questo candidato BUY in max 3 frasi in italiano. "
                            "Considera anche i segnali ML (WIN/LOSS e Trend Predictor). "
                            "Indica: 1) se è un buon entry e perché, "
                            "2) se dalle news emergono earnings/trimestrali imminenti. "
                            "Se ci sono earnings entro 7 giorni, AVVISA. "
                            "Sii diretto, concreto, no disclaimers."
                        ),
                        user_prompt=stock_data + earnings_context + macro_reasoning,
                        max_tokens=150,
                        temperature=0.3,
                        agent_name="alpha_strategist",
                    )
                    if analysis:
                        candidate["llm_analysis"] = analysis
                        print(f"    🧠 {candidate['ticker']}: {analysis[:60]}...")
                        await db.assets.update_one(
                            {"ticker": candidate["ticker"]},
                            {"$set": {"llm_analysis": analysis, "llm_analysis_at": datetime.utcnow().isoformat()}}
                        )
                except Exception as e:
                    print(f"    LLM error {candidate.get('ticker')}: {e}")
        
        summary = {
            "total_assets_scanned": len(assets),
            "buy_candidates": len(top_candidates),
            "sell_signals": len(sell_signals),
            "skipped_reasons": skipped_reasons,
            "market_regime": market_ctx.get("market_regime", "UNKNOWN"),
            "top_confluence": top_candidates[0]["confluence"] if top_candidates else 0,
            # 🆕 v2.0 — ML stats
            "ml_data_loaded": len(ml_map),
            # 🆕 Sentiment stats
            "sentiment_summary": {
                c["ticker"]: {
                    "sentiment": c.get("sentiment", "N/A"),
                    "earnings_soon": c.get("earnings_soon", False),
                    "adj": c.get("sentiment_adj", 0),
                } for c in top_candidates
            },
        }

        await self.log_decision(
            decision_type="scan_complete",
            data={
                "candidates_count": len(top_candidates),
                "sell_count": len(sell_signals),
                "top_tickers": [c["ticker"] for c in top_candidates[:5]],
                "sell_tickers": [s["ticker"] for s in sell_signals],
                "skipped": skipped_reasons,
                "ml_data_loaded": len(ml_map),
            },
            reasoning=f"Found {len(top_candidates)} buys, {len(sell_signals)} sells. "
                      f"Regime={market_ctx.get('market_regime')} "
                      f"Min confluence={min_conf} ML={len(ml_map)}",
            confidence=min(100, summary["top_confluence"]) if top_candidates else 20,
        )

        print(f"🎯 AlphaStrategist v2.0: {len(top_candidates)} candidates, "
              f"{len(sell_signals)} sell signals (ML: {len(ml_map)} tickers)")

        return {
            "buy_candidates": top_candidates,
            "sell_signals": sell_signals,
            "summary": summary,
        }

    async def learn(self) -> dict:
        """Learning loop (invariato dalla v1.0)."""
        db = get_db()
        params = await self.get_params()
        fw = params.get("factor_weights", self.default_params()["factor_weights"])

        trades = await db.trade_history.find({"side": "sell"}).to_list(500)

        if len(trades) < self.min_decisions_to_learn:
            return {"message": "Not enough trades to learn", "trades": len(trades)}

        wins = [t for t in trades if t.get("pnl_pct", 0) > 0]
        losses = [t for t in trades if t.get("pnl_pct", 0) <= 0]
        total = len(trades)
        win_rate = len(wins) / total * 100 if total > 0 else 50

        setup_stats = {}
        for t in trades:
            st = t.get("setup_type", "unknown")
            weight = self.calc_weight(t.get("date", datetime.utcnow()))
            if st not in setup_stats:
                setup_stats[st] = {"wins": 0, "losses": 0, "w_wins": 0, "w_losses": 0, "total_pnl": 0}
            if t.get("pnl_pct", 0) > 0:
                setup_stats[st]["wins"] += 1
                setup_stats[st]["w_wins"] += weight
            else:
                setup_stats[st]["losses"] += 1
                setup_stats[st]["w_losses"] += weight
            setup_stats[st]["total_pnl"] += t.get("pnl_pct", 0)

        best_setups = []
        worst_setups = []
        for st, stats in setup_stats.items():
            w_total = stats["w_wins"] + stats["w_losses"]
            raw_total = stats["wins"] + stats["losses"]
            if raw_total >= 3:
                w_wr = (stats["w_wins"] / w_total * 100) if w_total > 0 else 50
                if w_wr >= 55:
                    best_setups.append(st)
                elif w_wr < 35:
                    worst_setups.append(st)

        sector_stats = {}
        for t in trades:
            sec = t.get("sector", "unknown")
            weight = self.calc_weight(t.get("date", datetime.utcnow()))
            if sec not in sector_stats:
                sector_stats[sec] = {"w_wins": 0, "w_losses": 0, "total": 0}
            if t.get("pnl_pct", 0) > 0:
                sector_stats[sec]["w_wins"] += weight
            else:
                sector_stats[sec]["w_losses"] += weight
            sector_stats[sec]["total"] += 1

        weak_sectors = []
        for sec, stats in sector_stats.items():
            w_total = stats["w_wins"] + stats["w_losses"]
            if stats["total"] >= 3 and w_total > 0:
                if (stats["w_wins"] / w_total) < 0.35:
                    weak_sectors.append(sec)

        conf_buckets = {"high": {"w": 0, "l": 0}, "mid": {"w": 0, "l": 0}, "low": {"w": 0, "l": 0}}
        for t in trades:
            conf = t.get("confluence", 50)
            bucket = "high" if conf >= 60 else ("mid" if conf >= 35 else "low")
            if t.get("pnl_pct", 0) > 0:
                conf_buckets[bucket]["w"] += 1
            else:
                conf_buckets[bucket]["l"] += 1

        min_conf = params.get("min_confluence", 35)
        low_total = conf_buckets["low"]["w"] + conf_buckets["low"]["l"]
        mid_total = conf_buckets["mid"]["w"] + conf_buckets["mid"]["l"]

        if low_total >= 3:
            low_wr = conf_buckets["low"]["w"] / low_total
            if low_wr < 0.35:
                min_conf = min(min_conf + 3, 55)
            elif low_wr < 0.45:
                min_conf = min(min_conf + 1, 50)
        if mid_total >= 3:
            mid_wr = conf_buckets["mid"]["w"] / mid_total
            if mid_wr > 0.65:
                min_conf = max(min_conf - 2, 25)

        rsi_losses = [t.get("rsi_at_entry", 50) for t in losses if t.get("rsi_at_entry")]
        max_rsi = params.get("max_rsi_entry", 68)
        if rsi_losses:
            avg_loss_rsi = sum(rsi_losses) / len(rsi_losses)
            if avg_loss_rsi > 62:
                max_rsi = 60
            elif avg_loss_rsi > 55:
                max_rsi = 65

        params["best_setups"] = best_setups if best_setups else self.default_params()["best_setups"]
        params["worst_setups"] = worst_setups
        params["weak_sectors"] = weak_sectors
        params["min_confluence"] = round(min_conf, 1)
        params["max_rsi_entry"] = max_rsi

        await self.save_params(params)

        learn_result = {
            "win_rate": round(win_rate, 1),
            "total_trades": total,
            "best_setups": best_setups,
            "worst_setups": worst_setups,
            "weak_sectors": weak_sectors,
            "min_confluence": min_conf,
            "max_rsi": max_rsi,
            "setup_stats": {k: {"win_rate": round(v["wins"]/(v["wins"]+v["losses"])*100, 1)
                                if (v["wins"]+v["losses"]) > 0 else 0,
                                "trades": v["wins"]+v["losses"]}
                           for k, v in setup_stats.items()},
        }

        await self.save_performance({
            "win_rate": round(win_rate, 1),
            "total_trades": total,
            "min_confluence": min_conf,
        })

        print(f"🎯 AlphaStrategist LEARN: WR={win_rate:.1f}%, "
              f"best={best_setups}, worst={worst_setups}")

        return learn_result
