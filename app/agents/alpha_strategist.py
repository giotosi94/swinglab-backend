from datetime import datetime
from app.agents.base_agent import BaseAgent
from app.db.mongodb import get_db


# ==========================================================
# PROIEZIONI MONGODB — controllo della memoria
# ==========================================================
# Dopo Max Strategy v1.5.2 il campo max_strategy e' diventato il blocco
# piu' pesante del documento asset: contiene weekly_context,
# daily_confirmation, execution_4h, structural_base, entry_plan e i
# profili POC. Su 300 titoli sono circa 13 MB di documenti, che decodificati
# in dizionari Python occupano molto di piu'.
#
# Ne' il calcolo della confluence ne' i modelli ML usano quei dati, quindi
# vengono esclusi. Lo shadow di Max Strategy li rilegge a parte, ma solo
# nei singoli campi che gli servono davvero.
ASSET_PROJECTION = {
    "price_history": 0,
    "vp_distribution": 0,
    "multi_tf_vp": 0,
    "max_strategy": 0,
    "alpha_snapshot": 0,
    "history": 0,
}

# Solo i campi effettivamente letti da _build_max_strategy_shadow.
# Una proiezione positiva su sottocampi annidati evita di riportare in
# memoria l'intero blocco Max Strategy.
MAX_SHADOW_PROJECTION = {
    "ticker": 1,
    "price": 1,
    "max_strategy.strategy_type": 1,
    "max_strategy.strategy_eligible": 1,
    "max_strategy.trade_ready": 1,
    "max_strategy.max_score": 1,
    "max_strategy.rejection_reasons": 1,
    "max_strategy.market_phase.phase": 1,
    "max_strategy.entry_plan.status": 1,
    "max_strategy.entry_plan.order_action": 1,
    "max_strategy.entry_plan.execution_mode": 1,
    "max_strategy.entry_plan.trigger_price": 1,
    "max_strategy.entry_plan.maximum_entry_price": 1,
    "max_strategy.entry_plan.invalidation_price": 1,
    "max_strategy.entry_plan.blocking_phase": 1,
    "max_strategy.entry_plan.weekly_plan_qualified": 1,
}

# Barre necessarie al calcolo ATR: 14 periodi piu' la chiusura precedente.
ATR_BARS = 20


class AlphaStrategist(BaseAgent):
    """
    🎯 AGENTE 2: Alpha Strategist v2.2 — "Stock Picker with ML + Leadership"

    v2.2 — DUE INTERVENTI

    1) MEMORIA
       analyze() caricava 300 asset escludendo solo tre campi, quindi si
       portava in RAM anche tutto max_strategy. Ora gli asset arrivano
       senza i blocchi pesanti e lo shadow Max Strategy rilegge a parte
       soltanto i campi che usa davvero.

    2) LEADERSHIP
       Il MacroAnalyst v3.3 calcola dove sta andando il capitale e lo
       scrive in market_context. L'Alpha riceveva quei dati ma non li
       leggeva mai.

       Il collegamento NON e' un bonus di punti sulla confluence: quella
       strada era gia' stata bocciata dai backtest, perche' gonfiare lo
       score corrompe il confronto fra titoli di settori diversi.

       Qui la leadership modula la SOGLIA, non il punteggio:

         settore in focus    -> soglia leggermente piu' bassa
         settore da evitare  -> soglia piu' alta
         settore in deflusso -> soglia piu' alta

       Un titolo mantiene quindi la sua confluence reale. Cambia solo
       quanto dobbiamo essere esigenti per accettarlo.

       La soglia non scende mai sotto min_confluence_floor: la leadership
       puo' rendere il sistema piu' attento, mai indiscriminato.

    Seleziona le migliori opportunita' di acquisto e identifica segnali di vendita.
    Usa il MarketContext prodotto dal MacroAnalyst per contestualizzare le decisioni.
    
    v2.0 — 🆕 ML Integration:
    - Factor 14: WIN/LOSS XGBoost predictor
    - Factor 15: Trend Predictor (5d UP/FLAT/DOWN)
    
    Input: market_context, positions, ml_predictions, trend_predictions
    Output: buy_candidates[], sell_signals[], analysis_summary
    """

    def __init__(self):
        super().__init__(name="alpha_strategist", version="2.2")
        # v2.0 — Confluence max score teorico (13 factors + 2 ML)
        # Original: 15.0 (13 factors)
        # + Factor 14 (ML WIN/LOSS): 2.5 max
        # + Factor 15 (Trend Predictor): 2.0 max
        # = 19.5 total
        # v2.1 — +Factor 16 MTF Weekly Alignment (2.5 max) = 22.0
        self.MAX_RAW_CONFLUENCE = 25.0  # +3.0 POC Shift factor

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
                "sector_rank": 0.0,
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
                "poc_shift": 1.0,         # 🆕 POC Shift Contrarian (Rea)
            },
            # Filtri
            "min_confluence": 48,
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
            "sell_ml_loss_threshold": 0.30,

            # ============================================
            # 🆕 v2.2 — LEADERSHIP: modula la soglia, non il punteggio
            # ============================================
            "leadership_enabled": True,

            # Quanto abbassare la soglia dove il capitale sta arrivando
            "focus_threshold_discount": 3.0,

            # Quanto alzarla dove la struttura si sta deteriorando
            "avoid_threshold_premium": 5.0,

            # Quanto alzarla dove oggi esce denaro
            "outflow_threshold_premium": 3.0,

            # La soglia non scende mai sotto questo valore, qualunque sia
            # la leadership. Allineato al floor del learning loop.
            "min_confluence_floor": 42,

            # Peso del riordino: influenza solo CHI entra nei top 10,
            # non la confluence riportata al RiskManager.
            "leadership_sort_weight": 2.0,

            # weak_sectors resta salvato e visibile ma non penalizza piu'
            # gli acquisti: la leadership del MacroAnalyst e' piu' aggiornata
            # e guarda il mercato, non solo lo storico dei nostri trade.
            "weak_sector_penalty_enabled": False,  # se ML dice WIN score <30% + in perdita → sell
        }

    def _build_leadership_context(self, market_ctx: dict, params: dict) -> dict:
        """
        🆕 v2.2 — Estrae dal MarketContext dove sta andando il capitale.

        Il MacroAnalyst v3.3 distingue gia' fra denaro che conferma un trend
        e balzo isolato di una giornata: gli ONE_DAY_SPIKE sono esclusi a
        monte dai focus_sectors. Qui li riceviamo solo per poterli mostrare.
        """
        enabled = bool(params.get("leadership_enabled", True))

        ctx = {
            "enabled": enabled,
            "available": False,
            "focus": [],
            "avoid": [],
            "inflow": [],
            "outflow": [],
            "spike": [],
            "state": (market_ctx.get("leadership") or {}).get("state", "UNKNOWN"),
            "rotation_state": market_ctx.get("rotation_state", "UNKNOWN"),
            "regime_detail": market_ctx.get("regime_detail", "UNKNOWN"),
            "flow_summary": market_ctx.get("flow_summary", ""),
            "intraday_available": bool(market_ctx.get("intraday_available", False)),
        }

        if not enabled:
            return ctx

        focus = [s for s in (market_ctx.get("focus_sectors") or []) if s]
        avoid = [s for s in (market_ctx.get("avoid_sectors") or []) if s]

        # Il MacroAnalyst garantisce gia' che le due liste siano disgiunte,
        # ma non vogliamo dipendere da quella garanzia: se un settore finisse
        # in entrambe, prudenza prima di tutto e vince avoid.
        focus = [s for s in focus if s not in avoid]

        ctx["focus"] = focus
        ctx["avoid"] = avoid
        ctx["inflow"] = [s for s in (market_ctx.get("inflow_sectors") or []) if s]
        ctx["outflow"] = [s for s in (market_ctx.get("outflow_sectors") or []) if s]
        ctx["spike"] = [s for s in (market_ctx.get("spike_sectors") or []) if s]
        ctx["available"] = bool(focus or avoid or ctx["outflow"])

        return ctx

    def _sector_threshold(self, sector: str, base_threshold: float,
                          leadership: dict, params: dict) -> tuple:
        """
        🆕 v2.2 — Soglia di confluence specifica per il settore.

        Ritorna (soglia, motivo).

        La logica e' deliberatamente asimmetrica: lo sconto dove arriva
        denaro e' piccolo, il premio dove esce e' piu' grande. Sbagliare
        entrando dove il capitale sta uscendo costa piu' che perdere
        un'occasione dove sta arrivando.
        """
        if not leadership.get("enabled") or not leadership.get("available"):
            return float(base_threshold), "no_leadership"

        floor = float(params.get("min_confluence_floor", 42))
        threshold = float(base_threshold)
        reason = "neutral"

        if sector in leadership["avoid"]:
            threshold += float(params.get("avoid_threshold_premium", 5.0))
            reason = "avoid_sector"

        elif sector in leadership["focus"]:
            threshold -= float(params.get("focus_threshold_discount", 3.0))
            reason = "focus_sector"

        elif sector in leadership["outflow"]:
            threshold += float(params.get("outflow_threshold_premium", 3.0))
            reason = "outflow_sector"

        # In CRASH la leadership non deve mai allentare nulla.
        if leadership.get("regime_detail") == "CRASH":
            threshold = max(threshold, float(base_threshold))
            reason = "crash_no_discount"

        # Lo sconto non puo' portare sotto il floor: la leadership rende il
        # sistema piu' selettivo, non piu' permissivo in assoluto.
        threshold = max(threshold, floor)

        return round(threshold, 1), reason

    async def _load_max_shadow_assets(self, db) -> list:
        """
        🆕 v2.2 — Carica SOLO i campi Max Strategy usati dallo shadow.

        Gli asset principali ora escludono max_strategy per non saturare la
        memoria. Lo shadow ne ha comunque bisogno, ma di pochi campi: li
        rileggiamo con una proiezione stretta e in streaming, senza mai
        costruire una lista di documenti completi.
        """
        rows = []
        cursor = db.assets.find({}, MAX_SHADOW_PROJECTION)

        async for doc in cursor:
            rows.append(doc)

        return rows

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
        w = 0.0 if params.get("sector_intelligence_enabled", True) else fw.get("sector_rank", 1.0)
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

        # --- 17. 🆕 POC SHIFT CONTRARIAN (metodo Rea) ---
        # Segnale forte: POC passato da sopra a sotto (shift bull) CONFERMATO da
        # accumulazione + rottura. Non "vicino al POC alla cieca" — serve la conferma.
        w = fw.get("poc_shift", 1.0)
        poc_shift = asset.get("poc_shift", {})
        accum_sc = accum.get("score", 0)
        has_bullish = any(p.get("type") == "bullish" for p in patterns)
        wy_ph = wyckoff.get("phase", "")

        if poc_shift.get("shifted_bull") and accum_sc >= 50 and (has_bullish or wy_ph in ("accumulation", "spring")):
            # SHIFT confermato: POC sotto + accumulo + rottura → setup contrarian top
            pts = 3.0 * w
            factors.append({"name": "POCShift", "pts": pts, "max": 3.0, "detail": "Shift bull + accum + rottura!", "pass": True})
        elif poc_shift.get("poc_position") == "below" and accum_sc >= 40:
            # POC già sotto (supporto) + accumulo → buono ma non lo shift fresco
            pts = 1.0 * w
            factors.append({"name": "POCShift", "pts": pts, "max": 3.0, "detail": "POC sotto + accum", "pass": True})
        else:
            pts = 0
            factors.append({"name": "POCShift", "pts": 0, "max": 3.0, "detail": "no shift", "pass": False})
        raw_score += pts

        # Normalize to 0-100
        normalization_max = self.MAX_RAW_CONFLUENCE - 1.5 if params.get("sector_intelligence_enabled", True) else self.MAX_RAW_CONFLUENCE
        normalized = round(max(0, min(100, (raw_score / normalization_max) * 100)), 1)

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

        from datetime import datetime
        db = get_db()

        for p in positions:
            symbol = p.get("symbol")
            asset = assets_map.get(symbol)
            if not asset:
                continue

            # 🆕 Minimum holding anche lato Alpha (anti-churning): non generare
            # sell signal "soft" (pattern/score/ML) su posizioni aperte da <24h.
            # Evita il loop vende→ricompra quando Alpha vende e poi ricompra.
            try:
                buy_t = await db.trade_history.find_one(
                    {"ticker": symbol, "side": "buy", "sell_linked": {"$ne": True}},
                    sort=[("date", -1)]
                )
                if buy_t and buy_t.get("date"):
                    hours_held = (datetime.utcnow() - buy_t["date"]).total_seconds() / 3600
                    if hours_held < 24:
                        continue  # troppo fresca: niente sell soft
            except Exception:
                pass

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

    async def _recalc_targets_atr(self, db, candidates: list) -> list:
        """
        🆕 Target/Stop ATR-based (allineato al backtest) sui top candidati.
        Risolve il R/R schiacciato a 1.0 sui titoli estesi: target proporzionale
        alla volatilità + setup, stop stretto ATR-based → R/R realistici.
        """
        for c in candidates:
            try:
                # Il calcolo usa 14 periodi: chiedere a MongoDB solo le
                # ultime barre evita di caricare in memoria tutto lo storico
                # di ogni candidato.
                doc = await db.stock_bars.find_one(
                    {"ticker": c["ticker"]},
                    {"bars": {"$slice": -ATR_BARS}},
                )
                bars = (doc or {}).get("bars", [])
                if len(bars) < 16:
                    continue
                price = c["price"]

                # ATR% sugli ultimi 14 giorni
                period = 14
                trs = []
                for i in range(max(1, len(bars) - period), len(bars)):
                    h, l, pc = bars[i]["h"], bars[i]["l"], bars[i - 1]["c"]
                    trs.append(max(h - l, abs(h - pc), abs(l - pc)))
                atr = sum(trs) / len(trs) if trs else price * 0.02
                atr_pct = (atr / price * 100) if price > 0 else 2.0

                # Target multiplier per setup (come backtest)
                mult = {
                    "breakout": 4.0,
                    "ema_bounce": 3.0,
                    "pullback_to_poc": 3.5,
                    "oversold_reversal": 3.5,
                    "neutral": 3.0,
                }.get(c.get("setup_type", "neutral"), 3.0)

                target_dist = min(40, max(5, atr_pct * mult))
                sl_dist = min(12, max(3, atr_pct * 1.5))

                target_price = round(price * (1 + target_dist / 100), 2)
                stop_loss = round(price * (1 - sl_dist / 100), 2)

                risk = price - stop_loss
                reward = target_price - price
                rr = round(reward / risk, 2) if risk > 0 else 0

                c["target_price"] = target_price
                c["stop_loss"] = stop_loss
                c["risk_reward"] = rr
                c["atr_pct"] = round(atr_pct, 2)
            except Exception as e:
                print(f"  ATR recalc error {c.get('ticker')}: {e}")
        return candidates

    async def _enrich_with_sentiment(self, candidates: list) -> list:
        """
        🆕 SentimentAgent — arricchisce i top candidati con news sentiment.
        🔧 Cache 4h per ticker: se già analizzato di recente riusa il risultato
        senza chiamare l'LLM (le news non cambiano ogni 15 minuti).
        """
        from app.services.news_service import get_stock_news_with_sentiment
        from datetime import timedelta
        db = get_db()
        for c in candidates:
            try:
                # 🔧 CACHE: riusa il sentiment salvato se recente (<4h)
                cached = await db.sentiment_cache.find_one({"_id": c["ticker"]})
                if cached and cached.get("ts"):
                    age_h = (datetime.utcnow() - cached["ts"]).total_seconds() / 3600
                    if age_h < 4:
                        c["sentiment"] = cached.get("sentiment", "N/A")
                        c["earnings_soon"] = cached.get("earnings_soon", False)
                        c["sentiment_adj"] = cached.get("adj", 0)
                        c["confluence"] = round(max(0, c["confluence"] + cached.get("adj", 0)), 1)
                        c["sentiment_cached"] = True
                        continue

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
                    adj -= 15  # 🆕 rischio gap notturno FORTE — earnings = pericolo n.1 swing

                c["sentiment"] = sent
                c["earnings_soon"] = earnings_soon
                c["sentiment_adj"] = adj
                c["confluence"] = round(max(0, c["confluence"] + adj), 1)

                # 🔧 Salva in cache (valida 4h)
                await db.sentiment_cache.update_one(
                    {"_id": c["ticker"]},
                    {"$set": {"sentiment": sent, "earnings_soon": earnings_soon,
                              "adj": adj, "ts": datetime.utcnow()}},
                    upsert=True,
                )
            except Exception as e:
                c["sentiment"] = "N/A"
                c["sentiment_adj"] = 0
        # Ri-ordina con il sentiment incluso
        candidates.sort(key=lambda x: x["confluence"], reverse=True)
        return candidates

    def _build_max_strategy_shadow(self, assets: list, open_tickers: list, market_ctx: dict) -> dict:
        shadow_candidates = []
        status_counts = {}
        action_counts = {"WOULD_ARM": 0, "WOULD_BUY": 0, "WOULD_WAIT": 0, "WOULD_REJECT": 0}
        regime = market_ctx.get("market_regime", "NEUTRAL")
        for asset in assets:
            ticker = asset.get("ticker", "")
            max_strategy = asset.get("max_strategy") or {}
            plan = max_strategy.get("entry_plan") or {}
            phase = max_strategy.get("market_phase") or {}
            status = plan.get("status", "DETECTED")
            status_counts[status] = status_counts.get(status, 0) + 1
            blocking = bool(plan.get("blocking_phase"))
            eligible = bool(max_strategy.get("strategy_eligible"))
            trade_ready = bool(max_strategy.get("trade_ready"))
            weekly_qualified = bool(plan.get("weekly_plan_qualified"))
            order_action = plan.get("order_action", "WAIT")
            reasons = list(max_strategy.get("rejection_reasons") or [])
            if ticker in open_tickers:
                shadow_action = "WOULD_REJECT"
                reasons.append("ALREADY_OPEN")
            elif blocking or not eligible or not weekly_qualified:
                shadow_action = "WOULD_REJECT"
            elif order_action == "BUY_ALLOWED" and trade_ready:
                shadow_action = "WOULD_BUY"
            elif order_action == "PLACE_CONDITIONAL_BUY" and status == "ARMED":
                shadow_action = "WOULD_ARM"
            else:
                shadow_action = "WOULD_WAIT"
            if regime == "CRASH" and shadow_action in ("WOULD_ARM", "WOULD_BUY"):
                shadow_action = "WOULD_REJECT"
                reasons.append("CRASH_REGIME_BLOCK")
            action_counts[shadow_action] += 1
            if shadow_action == "WOULD_WAIT" and status == "DETECTED":
                continue
            shadow_candidates.append({
                "ticker": ticker,
                "shadow_action": shadow_action,
                "plan_status": status,
                "order_action": order_action,
                "execution_mode": plan.get("execution_mode"),
                "market_phase": phase.get("phase"),
                "strategy_type": max_strategy.get("strategy_type"),
                "max_score": max_strategy.get("max_score", 0),
                "signal_price": asset.get("price"),
                "trigger_price": plan.get("trigger_price"),
                "maximum_entry_price": plan.get("maximum_entry_price"),
                "invalidation_price": plan.get("invalidation_price"),
                "weekly_qualified": weekly_qualified,
                "strategy_eligible": eligible,
                "trade_ready": trade_ready,
                "blocking_phase": blocking,
                "rejection_reasons": list(dict.fromkeys(reasons)),
                "live_execution_enabled": False,
            })
        priority = {"WOULD_BUY": 0, "WOULD_ARM": 1, "WOULD_WAIT": 2, "WOULD_REJECT": 3}
        shadow_candidates.sort(key=lambda item: (priority.get(item["shadow_action"], 9), -float(item.get("max_score") or 0)))
        return {
            "mode": "SHADOW",
            "live_execution_enabled": False,
            "strategy_version": "max_structure_v1_5_2",
            "regime": regime,
            "action_counts": action_counts,
            "status_counts": status_counts,
            "candidates": shadow_candidates[:100],
        }

    async def analyze(self, context: dict) -> dict:
        db = get_db()
        params = await self.get_params()
        market_ctx = context.get("market_context", {})
        positions = context.get("positions", [])

        # 🆕 v2.0 — Load ML predictions (calcolate on-the-fly dai modelli)
        # NOTA: dobbiamo passare gli assets che caricheremo tra poco
        # Per efficienza, carichiamo prima gli assets
        # 🆕 v2.2 — FIX MEMORIA
        # La versione precedente escludeva solo tre campi e quindi caricava
        # in RAM anche tutto max_strategy per 300 titoli. Ne' il calcolo
        # della confluence ne' i modelli ML lo usano.
        assets = await db.assets.find({}, ASSET_PROJECTION).to_list(400)
        
        ml_map = await self._load_ml_predictions(db, assets, market_ctx)
        print(f"  📊 ML data loaded: {len(ml_map)} tickers with predictions")


        if not assets:
            return {"buy_candidates": [], "sell_signals": [], "error": "No assets data"}

        assets_map = {a["ticker"]: a for a in assets}
        open_tickers = [p.get("symbol") for p in positions]
        # Lo shadow Max Strategy rilegge a parte solo i campi che usa,
        # poi libera subito la lista: non deve restare in memoria per tutto
        # il resto della scansione.
        shadow_assets = await self._load_max_shadow_assets(db)
        max_strategy_shadow = self._build_max_strategy_shadow(
            shadow_assets, open_tickers, market_ctx
        )
        del shadow_assets

        await db.agent_state.update_one(
            {"_id": "max_strategy_shadow"},
            {"$set": {**max_strategy_shadow, "updated_at": datetime.utcnow()}},
            upsert=True,
        )

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

        # 🆕 v2.2 — Dove sta andando il capitale, secondo il MacroAnalyst
        leadership = self._build_leadership_context(market_ctx, params)

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

        # 🆕 v2.2 — Quanto ha pesato davvero la leadership su questa scansione
        leadership_stats = {
            "focus_passed": 0,
            "avoid_blocked": 0,
            "outflow_blocked": 0,
            "focus_candidates": 0,
            "avoid_candidates": 0,
        }

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

            # weak_sectors resta un dato osservato, non una penalita': deriva
            # dallo storico dei nostri trade, mentre la leadership del
            # MacroAnalyst guarda cosa sta facendo il mercato adesso.
            if params.get("weak_sector_penalty_enabled", False):
                sector_penalty = -5 if sector in weak_sectors else 0
            else:
                sector_penalty = 0

            # 🆕 v2.0 — Passa ml_data al calc_confluence
            ml_data = ml_map.get(ticker)
            conf = self._calc_confluence(a, market_ctx, params, ml_data)
            conf_score = conf["score"] + sector_penalty

            # 🆕 v2.2 — La soglia dipende da dove sta andando il capitale.
            # La confluence resta quella reale: cambia solo quanto siamo
            # esigenti per accettarla.
            sector_threshold, threshold_reason = self._sector_threshold(
                sector, min_conf, leadership, params
            )

            if conf_score < sector_threshold:
                skipped_reasons["low_confluence"] += 1

                # Se il titolo sarebbe passato con la soglia base, e' stata
                # la leadership a fermarlo: va tracciato.
                if conf_score >= min_conf:
                    if threshold_reason == "avoid_sector":
                        leadership_stats["avoid_blocked"] += 1
                    elif threshold_reason == "outflow_sector":
                        leadership_stats["outflow_blocked"] += 1

                continue

            # Accettato grazie allo sconto sul settore di focus
            if threshold_reason == "focus_sector" and conf_score < min_conf:
                leadership_stats["focus_passed"] += 1

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

                # 🆕 v2.2 — Contesto leadership del settore
                "sector_flow": (
                    "FOCUS" if sector in leadership["focus"]
                    else "AVOID" if sector in leadership["avoid"]
                    else "OUTFLOW" if sector in leadership["outflow"]
                    else "NEUTRAL"
                ),
                "sector_threshold": sector_threshold,
                "threshold_reason": threshold_reason,
            })

        # 🆕 v2.2 — Ordinamento.
        #
        # La confluence resta intatta: quella va al RiskManager e viene
        # mostrata a schermo. Il riordino usa un punteggio separato, che
        # serve solo a decidere CHI entra nei primi 10 quando i punteggi
        # sono vicini. A parita' sostanziale preferiamo il titolo che sta
        # dove il capitale sta arrivando.
        sort_weight = float(params.get("leadership_sort_weight", 2.0))

        for c in candidates:
            flow = c.get("sector_flow", "NEUTRAL")
            if flow == "FOCUS":
                adjustment = sort_weight
            elif flow in ("AVOID", "OUTFLOW"):
                adjustment = -sort_weight
            else:
                adjustment = 0.0
            c["_sort_score"] = c["confluence"] + adjustment

        candidates.sort(key=lambda x: x["_sort_score"], reverse=True)
        top_candidates = candidates[:10]

        for c in candidates:
            c.pop("_sort_score", None)

        leadership_stats["focus_candidates"] = sum(
            1 for c in top_candidates if c.get("sector_flow") == "FOCUS"
        )
        leadership_stats["avoid_candidates"] = sum(
            1 for c in top_candidates if c.get("sector_flow") in ("AVOID", "OUTFLOW")
        )

        # 🆕 Ricalcola target/stop ATR-based (R/R realistici, come backtest)
        top_candidates = await self._recalc_targets_atr(db, top_candidates)

        # 🆕 SentimentAgent — solo TOP 3 e max 1 volta ogni 2h (risparmio quota LLM)
        try:
            from datetime import timedelta
            sstate = await db.agent_state.find_one({"_id": "sentiment_last_run"})
            last_ts = sstate.get("ts") if sstate else None
            hours_since = (datetime.utcnow() - last_ts).total_seconds() / 3600 if last_ts else 999
            if hours_since >= 2 and top_candidates:
                enriched = await self._enrich_with_sentiment(top_candidates[:3])
                top_candidates[:3] = enriched
                await db.agent_state.update_one(
                    {"_id": "sentiment_last_run"},
                    {"$set": {"ts": datetime.utcnow()}}, upsert=True)
                top_candidates.sort(key=lambda x: x["confluence"], reverse=True)
                print(f"  📰 Sentiment: analizzati {len(enriched)} candidati")
            else:
                print(f"  📰 Sentiment: skip (ultimo run {hours_since:.1f}h fa)")
        except Exception as e:
            print(f"  Sentiment skip: {e}")

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
                    
                    flow_info = ""
                    sector_flow = candidate.get("sector_flow", "NEUTRAL")
                    if sector_flow == "FOCUS":
                        flow_info = (
                            f"\nFlusso settore: capitale in ingresso su "
                            f"{candidate.get('sector', '')}"
                        )
                    elif sector_flow in ("AVOID", "OUTFLOW"):
                        flow_info = (
                            f"\nFlusso settore: ATTENZIONE, capitale in uscita da "
                            f"{candidate.get('sector', '')}"
                        )

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
                        f"{flow_info}"
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

            # 🆕 v2.2 — Leadership
            "leadership": {
                "enabled": leadership["enabled"],
                "available": leadership["available"],
                "state": leadership["state"],
                "rotation_state": leadership["rotation_state"],
                "regime_detail": leadership["regime_detail"],
                "intraday_available": leadership["intraday_available"],
                "flow_summary": leadership["flow_summary"],
                "focus_sectors": leadership["focus"],
                "avoid_sectors": leadership["avoid"],
                "outflow_sectors": leadership["outflow"],
                "spike_sectors_ignored": leadership["spike"],
                "base_threshold": min_conf,
                "stats": leadership_stats,
            },

            # 🆕 Sentiment stats
            "max_strategy_shadow": {
                "mode": max_strategy_shadow["mode"],
                "live_execution_enabled": False,
                "action_counts": max_strategy_shadow["action_counts"],
                "status_counts": max_strategy_shadow["status_counts"],
                "visible_candidates": len(max_strategy_shadow["candidates"]),
            },
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
                "leadership_state": leadership["state"],
                "focus_sectors": leadership["focus"],
                "avoid_sectors": leadership["avoid"],
                "leadership_stats": leadership_stats,
            },
            reasoning=f"Found {len(top_candidates)} buys, {len(sell_signals)} sells. "
                      f"Regime={market_ctx.get('market_regime')} "
                      f"Min confluence={min_conf} ML={len(ml_map)}",
            confidence=min(100, summary["top_confluence"]) if top_candidates else 20,
        )

        print(f"🎯 AlphaStrategist v2.2: {len(top_candidates)} candidates, "
              f"{len(sell_signals)} sell signals (ML: {len(ml_map)} tickers)")

        if leadership["available"]:
            focus_list = ", ".join(leadership["focus"]) or "nessuno"
            avoid_list = ", ".join(leadership["avoid"]) or "nessuno"
            discount = params.get("focus_threshold_discount", 3.0)
            premium = params.get("avoid_threshold_premium", 5.0)

            print(f"  🧭 Leadership: focus [{focus_list}] soglia {min_conf - discount:.0f} | "
                  f"evita [{avoid_list}] soglia {min_conf + premium:.0f}")

            if (leadership_stats["focus_passed"]
                    or leadership_stats["avoid_blocked"]
                    or leadership_stats["outflow_blocked"]):
                print(f"     Effetto: +{leadership_stats['focus_passed']} accettati in focus, "
                      f"-{leadership_stats['avoid_blocked']} bloccati in avoid, "
                      f"-{leadership_stats['outflow_blocked']} bloccati in deflusso")

            if leadership["spike"]:
                print(f"     Ignorati (solo balzo di giornata): "
                      f"{', '.join(leadership['spike'])}")
        else:
            print(f"  🧭 Leadership: non disponibile, soglia uniforme {min_conf} "
                  f"su tutti i settori")

        return {
            "buy_candidates": top_candidates,
            "sell_signals": sell_signals,
            "summary": summary,
            "max_strategy_shadow": max_strategy_shadow,
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
                min_conf = min(min_conf + 3, 60)
            elif low_wr < 0.45:
                min_conf = min(min_conf + 1, 55)
        if mid_total >= 3:
            mid_wr = conf_buckets["mid"]["w"] / mid_total
            if mid_wr > 0.65:
                min_conf = max(min_conf - 2, 42)  # 🔧 floor a 42 (non scende sotto)

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
        # weak_sectors viene ancora calcolato e salvato perche' e' utile
        # saperlo, ma non penalizza piu' gli acquisti: se ne occupa la
        # leadership del MacroAnalyst, che guarda il mercato e non solo
        # lo storico dei nostri trade.
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
