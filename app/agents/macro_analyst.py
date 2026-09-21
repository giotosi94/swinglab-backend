from datetime import datetime, timedelta
from app.agents.base_agent import BaseAgent
from app.db.mongodb import get_db


MACRO_INDICES = ["SPY", "QQQ", "IWM", "DIA", "RSP"]

RISK_SYMBOLS = [
    "TLT", "HYG", "LQD", "GLD", "USO",
    "IWO", "EEM", "IYT", "UUP", "FXE",
]

SECTOR_ETFS = [
    "XLK", "XLF", "XLV", "XLI", "XLY",
    "XLP", "XLE", "XLU", "XLB", "XLRE", "XLC",
]

DEFENSIVE_SECTORS = {"XLU", "XLP", "XLV"}
CYCLICAL_SECTORS = {"XLE", "XLI", "XLB", "XLY", "XLF"}

REGIME_LEVELS = ["CRASH", "BEAR", "NEUTRAL", "BULL"]


class MacroAnalyst(BaseAgent):
    """
    AGENTE 1: Macro Analyst v3.0

    Novita' rispetto alla v2.0:

    1. MULTI-TIMEFRAME
       Ogni componente usa rendimenti a 5, 20 e 63 sedute invece del solo
       change_pct giornaliero. Il regime smette di oscillare per il rumore
       di una singola seduta.

    2. MARKET LEADERSHIP ENGINE
       Calcola i ratio SPY/RSP, QQQ/SPY, IWM/SPY, DIA/SPY in trend e la
       forza relativa settoriale a 63 sedute. Risponde alla domanda:
       "chi sta realmente trainando il mercato?"

    3. ISTERESI + CONFERMA + SMOOTHING
       Il regime cambia solo se la confidence supera la soglia di un
       margine e se la lettura viene confermata su piu' cicli.

    4. REGIME DETTAGLIATO
       Oltre a BULL/NEUTRAL/BEAR/CRASH (mantenuti per compatibilita' con
       gli altri agenti) produce un regime_detail:
       NARROW_BULL, PULLBACK_IN_UPTREND, EARLY_RECOVERY, ROTATION.
       L'exposure_multiplier deriva dal regime dettagliato.
    """

    def __init__(self):
        super().__init__(name="macro_analyst", version="3.0")

    def default_params(self) -> dict:
        return {
            # ---- pesi composite (somma 1.00) ----
            "w_spy_trend": 0.22,
            "w_spy_rsi": 0.06,
            "w_vix": 0.12,
            "w_breadth": 0.18,
            "w_indices_alignment": 0.12,
            "w_crypto": 0.03,
            "w_dollar": 0.05,
            "w_bonds": 0.10,
            "w_commodities": 0.04,
            "w_risk_appetite": 0.08,

            # ---- volatilita' ----
            "vixy_high": 22,
            "vixy_extreme": 28,
            "vixy_low": 14,

            # ---- breadth ----
            "breadth_healthy": 60,
            "breadth_weak": 40,
            "breadth_critical": 25,

            # ---- soglie regime ----
            "bull_threshold": 65,
            "neutral_threshold": 47,
            "bear_threshold": 30,
            "hysteresis_buffer": 3,
            "regime_confirm_cycles": 2,
            "confidence_smoothing": 3,

            # ---- multi-timeframe ----
            "short_window": 5,
            "swing_window": 20,
            "structural_window": 63,

            # ---- leadership ----
            "concentration_threshold": 1.5,
            "broad_tolerance": 0.5,
            "early_recovery_short_return": 1.5,
            "intraday_strong_move": 1.0,

            # ---- exposure ----
            "bull_exposure": 1.0,
            "narrow_bull_exposure": 0.70,
            "pullback_exposure": 0.75,
            "neutral_exposure": 0.60,
            "rotation_exposure": 0.55,
            "early_recovery_exposure": 0.55,
            "bear_exposure": 0.35,
            "crash_exposure": 0.0,
        }

    # ==========================================================
    # HELPERS
    # ==========================================================

    def _clamp(self, value, min_val=0, max_val=100):
        return max(min_val, min(max_val, value))

    @staticmethod
    def _pct_return(closes, periods):
        """Rendimento percentuale su N sedute chiuse."""
        if not closes or len(closes) <= periods:
            return 0.0
        past = closes[-(periods + 1)]
        if past <= 0:
            return 0.0
        return round((closes[-1] - past) / past * 100, 2)

    def _windows(self, closes, params):
        short = params.get("short_window", 5)
        swing = params.get("swing_window", 20)
        structural = params.get("structural_window", 63)
        return {
            "r_short": self._pct_return(closes, short),
            "r_swing": self._pct_return(closes, swing),
            "r_structural": self._pct_return(closes, structural),
        }

    async def _load_price_series(self, db, symbols):
        """Carica le chiusure daily per una lista di simboli."""
        series = {}
        cursor = db.stock_bars.find(
            {"ticker": {"$in": symbols}},
            {"ticker": 1, "bars": 1},
        )
        async for doc in cursor:
            closes = [b["c"] for b in doc.get("bars", []) if b.get("c")]
            if len(closes) >= 25:
                series[doc["ticker"]] = closes
        return series

    async def _symbol_windows(self, db, symbols, params):
        """
        Finestre multi-timeframe per ogni simbolo.
        Se mancano le barre, ripiega sui campi di market_regime.
        """
        series = await self._load_price_series(db, symbols)
        out = {}

        for sym in symbols:
            closes = series.get(sym)
            if closes:
                data = self._windows(closes, params)
                data["fallback"] = False
                data["price"] = closes[-1]
                out[sym] = data
                continue

            doc = await db.market_regime.find_one({"symbol": sym})
            change = doc.get("change_pct", 0) if doc else 0
            ret20 = doc.get("return_20d", 0) if doc else 0
            out[sym] = {
                "r_short": change or 0,
                "r_swing": ret20 or 0,
                "r_structural": ret20 or 0,
                "fallback": True,
                "price": doc.get("price", 0) if doc else 0,
            }

        return out

    # ==========================================================
    # MARKET LEADERSHIP ENGINE
    # ==========================================================

    async def _calc_market_leadership(self, db, params) -> dict:
        """
        Chi traina il mercato, ragionando in trend e non sulla singola seduta.

        Ratio calcolati contro SPY:
          RSP/SPY  -> partecipazione (equal weight vs cap weight)
          QQQ/SPY  -> leadership tecnologica
          IWM/SPY  -> partecipazione small cap
          DIA/SPY  -> leadership industriale

        Piu' la forza relativa settoriale a 63 sedute.
        """
        result = {
            "available": False,
            "state": "UNKNOWN",
            "description": "Dati insufficienti",
            "indices": {},
            "concentration_swing": 0.0,
            "concentration_structural": 0.0,
            "participation_score": 50.0,
            "sectors": [],
            "sectors_above_spy": 0,
            "sectors_total": 0,
            "sector_dispersion": 0.0,
            "sector_leadership": "UNKNOWN",
            "top_sectors": [],
            "bottom_sectors": [],
        }

        symbols = MACRO_INDICES + SECTOR_ETFS
        series = await self._load_price_series(db, symbols)

        spy_closes = series.get("SPY")
        if not spy_closes or len(spy_closes) < 25:
            return result

        spy_w = self._windows(spy_closes, params)

        # ---- 1. Indici vs SPY ----
        indices = {}
        for sym in ["QQQ", "IWM", "DIA", "RSP"]:
            closes = series.get(sym)
            if not closes:
                continue
            w = self._windows(closes, params)
            indices[sym] = {
                "r_short": w["r_short"],
                "r_swing": w["r_swing"],
                "r_structural": w["r_structural"],
                "rs_swing": round(w["r_swing"] - spy_w["r_swing"], 2),
                "rs_structural": round(w["r_structural"] - spy_w["r_structural"], 2),
            }

        result["indices"] = indices
        result["spy"] = spy_w

        rsp = indices.get("RSP", {})
        qqq = indices.get("QQQ", {})
        iwm = indices.get("IWM", {})

        # Concentrazione: SPY meglio di RSP significa che comandano le mega cap
        concentration_swing = round(-rsp.get("rs_swing", 0.0), 2)
        concentration_structural = round(-rsp.get("rs_structural", 0.0), 2)
        result["concentration_swing"] = concentration_swing
        result["concentration_structural"] = concentration_structural

        # ---- 2. Forza relativa settoriale a 63 sedute ----
        sector_rs = []
        for sym in SECTOR_ETFS:
            closes = series.get(sym)
            if not closes:
                continue
            w = self._windows(closes, params)
            sector_rs.append({
                "sector": sym,
                "rs_swing": round(w["r_swing"] - spy_w["r_swing"], 2),
                "rs_structural": round(w["r_structural"] - spy_w["r_structural"], 2),
                "r_short": w["r_short"],
            })

        sector_rs.sort(key=lambda x: x["rs_structural"], reverse=True)
        result["sectors"] = sector_rs
        result["sectors_total"] = len(sector_rs)

        if sector_rs:
            above = [s for s in sector_rs if s["rs_structural"] > 0]
            result["sectors_above_spy"] = len(above)
            result["sector_dispersion"] = round(
                sector_rs[0]["rs_structural"] - sector_rs[-1]["rs_structural"], 2
            )
            result["top_sectors"] = [s["sector"] for s in sector_rs[:3]]
            result["bottom_sectors"] = [s["sector"] for s in sector_rs[-3:]]

            top3 = set(result["top_sectors"])
            defensive_in_top = len(DEFENSIVE_SECTORS & top3)
            cyclical_in_top = len(CYCLICAL_SECTORS & top3)

            if defensive_in_top >= 2:
                result["sector_leadership"] = "DEFENSIVE_LED"
            elif cyclical_in_top >= 2:
                result["sector_leadership"] = "CYCLICAL_LED"
            elif "XLK" in top3 or "XLC" in top3:
                result["sector_leadership"] = "TECH_LED"
            else:
                result["sector_leadership"] = "MIXED"

        # ---- 3. Participation score ----
        # 50 = partecipazione normale. Sopra = ampia. Sotto = stretta.
        participation = 50.0
        participation -= concentration_swing * 8
        participation += iwm.get("rs_swing", 0.0) * 4

        if result["sectors_total"] > 0:
            pct_above = result["sectors_above_spy"] / result["sectors_total"] * 100
            participation += (pct_above - 45) * 0.4

        participation = round(self._clamp(participation, 5, 95), 1)
        result["participation_score"] = participation

        # ---- 4. Stato di leadership ----
        concentration_threshold = params.get("concentration_threshold", 1.5)
        broad_tolerance = params.get("broad_tolerance", 0.5)

        if iwm.get("rs_swing", 0.0) > 1.0 and rsp.get("rs_swing", 0.0) > 0:
            state = "SMALL_CAP_LED"
            description = "Rally ampio guidato dalle small cap"
        elif rsp.get("rs_swing", 0.0) >= -broad_tolerance:
            state = "BROAD_PARTICIPATION"
            description = "Partecipazione ampia, equal weight in linea con SPY"
        elif concentration_swing > concentration_threshold and qqq.get("rs_swing", 0.0) > 0:
            state = "MEGA_CAP_NARROW"
            description = "Rialzo stretto guidato dalle mega cap tecnologiche"
        elif concentration_swing > concentration_threshold:
            state = "NARROW_NON_TECH"
            description = "Rialzo stretto non guidato dalla tecnologia"
        elif result["sector_leadership"] == "DEFENSIVE_LED":
            state = "DEFENSIVE_LED"
            description = "Comandano i settori difensivi"
        else:
            state = "NO_CLEAR_LEADERSHIP"
            description = "Nessuna leadership definita"

        if result["sector_leadership"] == "DEFENSIVE_LED" and state in (
            "BROAD_PARTICIPATION", "NO_CLEAR_LEADERSHIP"
        ):
            state = "DEFENSIVE_LED"
            description = "Partecipazione presente ma guidata dai difensivi"

        result["state"] = state
        result["description"] = description
        result["available"] = True

        return result

    # ==========================================================
    # CRASH RADAR
    # ==========================================================

    async def _calc_crash_radar(self, db, vixy_price: float) -> dict:
        spy_dd_pct = 0.0
        spy_bars_doc = await db.stock_bars.find_one({"ticker": "SPY"})
        if spy_bars_doc and spy_bars_doc.get("bars"):
            closes = [b["c"] for b in spy_bars_doc["bars"][-252:] if b.get("c")]
            if len(closes) >= 20:
                peak = max(closes)
                current = closes[-1]
                spy_dd_pct = round((current - peak) / peak * 100, 2)

        dd_abs = abs(spy_dd_pct)
        if dd_abs >= 30:
            dd_score = 100
        elif dd_abs >= 20:
            dd_score = 70 + (dd_abs - 20)
        elif dd_abs >= 10:
            dd_score = 30 + (dd_abs - 10) * 4
        else:
            dd_score = dd_abs * 3

        if vixy_price >= 45:
            vix_score = 100
        elif vixy_price >= 35:
            vix_score = 70 + (vixy_price - 35) * 3
        elif vixy_price >= 28:
            vix_score = 40 + (vixy_price - 28) * 4.3
        else:
            vix_score = self._clamp(vixy_price * 1.4, 0, 40)

        crash_risk_score = round(self._clamp(dd_score * 0.65 + vix_score * 0.35), 1)

        if crash_risk_score >= 70:
            crash_level = "DEPLOY_MAX"
        elif crash_risk_score >= 50:
            crash_level = "DEPLOY"
        elif crash_risk_score >= 30:
            crash_level = "WATCH"
        else:
            crash_level = "NORMAL"

        return {
            "crash_risk_score": crash_risk_score,
            "crash_level": crash_level,
            "spy_drawdown_pct": spy_dd_pct,
            "vixy_price": round(vixy_price, 2),
            "dd_score": round(dd_score, 1),
            "vix_score": round(vix_score, 1),
            "deploy_signal": crash_level in ("DEPLOY", "DEPLOY_MAX"),
        }

    # ==========================================================
    # SECTOR BOTTOM DETECTOR
    # ==========================================================

    async def _calc_sector_bottom_detector(self, db) -> dict:
        from app.services.data_fetcher import SECTOR_STOCKS

        SECTOR_CFG = {
            "XLE":  {"threshold": 1,  "weight": 13.63},
            "XLC":  {"threshold": 5,  "weight": 18.58},
            "XLI":  {"threshold": 5,  "weight": 18.58},
            "XLU":  {"threshold": 5,  "weight": 10.0},
            "XLF":  {"threshold": 6,  "weight": 15.44},
            "XLK":  {"threshold": 6,  "weight": 17.56},
            "XLRE": {"threshold": 8,  "weight": 8.81},
            "XLY":  {"threshold": 8,  "weight": 23.22},
            "XLB":  {"threshold": 10, "weight": 12.82},
            "XLP":  {"threshold": 11, "weight": 10.94},
            "XLV":  {"threshold": 11, "weight": 16.27},
        }

        all_tickers = [t for lst in SECTOR_STOCKS.values() for t in lst]
        bars_cursor = db.stock_bars.find(
            {"ticker": {"$in": all_tickers}}, {"ticker": 1, "bars": 1}
        )

        bars_map = {}
        async for doc in bars_cursor:
            bars_map[doc["ticker"]] = doc.get("bars", [])

        sectors_status = []
        bottom_sectors = []

        for sec, tickers in SECTOR_STOCKS.items():
            cfg = SECTOR_CFG.get(sec)
            if not cfg:
                continue

            counted = 0
            above = 0

            for tk in tickers:
                bars = bars_map.get(tk, [])
                closes = [b["c"] for b in bars[-200:] if b.get("c")]
                if len(closes) < 100:
                    continue
                sma = sum(closes) / len(closes)
                counted += 1
                if closes[-1] > sma:
                    above += 1

            if counted == 0:
                continue

            pct_above = round(above / counted * 100, 1)
            is_bottom = pct_above < cfg["threshold"]

            entry = {
                "sector": sec,
                "pct_above_200sma": pct_above,
                "threshold": cfg["threshold"],
                "weight": cfg["weight"],
                "is_bottom": is_bottom,
                "n_stocks": counted,
            }
            sectors_status.append(entry)

            if is_bottom:
                bottom_sectors.append(entry)

        bottom_sectors.sort(key=lambda x: x["weight"], reverse=True)

        return {
            "any_bottom": len(bottom_sectors) > 0,
            "bottom_sectors": bottom_sectors,
            "all_sectors": sorted(sectors_status, key=lambda x: x["pct_above_200sma"]),
        }

    # ==========================================================
    # REGIME: ISTERESI + CONFERMA
    # ==========================================================

    def _raw_level(self, confidence, params):
        if confidence >= params.get("bull_threshold", 65):
            return "BULL"
        if confidence >= params.get("neutral_threshold", 47):
            return "NEUTRAL"
        if confidence >= params.get("bear_threshold", 30):
            return "BEAR"
        return "CRASH"

    def _apply_hysteresis(self, confidence, last_regime, params):
        """
        Per salire di livello serve superare la soglia di un margine.
        Per scendere serve perderla dello stesso margine.
        """
        raw = self._raw_level(confidence, params)

        if not last_regime or last_regime not in REGIME_LEVELS:
            return raw

        buffer = params.get("hysteresis_buffer", 3)
        bounds = {
            "BEAR": params.get("bear_threshold", 30),
            "NEUTRAL": params.get("neutral_threshold", 47),
            "BULL": params.get("bull_threshold", 65),
        }

        last_index = REGIME_LEVELS.index(last_regime)
        raw_index = REGIME_LEVELS.index(raw)

        if raw_index > last_index:
            needed = bounds.get(raw, 0) + buffer
            if confidence < needed:
                return last_regime

        elif raw_index < last_index:
            needed = bounds.get(last_regime, 0) - buffer
            if confidence >= needed:
                return last_regime

        return raw

    # ==========================================================
    # ANALYZE
    # ==========================================================

    async def analyze(self, context: dict = None) -> dict:
        db = get_db()
        params = await self.get_params()

        short_w = params.get("short_window", 5)
        swing_w = params.get("swing_window", 20)
        structural_w = params.get("structural_window", 63)

        # ============================================
        # 1. SPY multi-timeframe
        # ============================================
        spy = await db.market_regime.find_one({"symbol": "SPY"})
        spy_price = spy.get("price", 0) if spy else 0
        spy_ema20 = spy.get("ema20", 0) if spy else 0
        spy_ema50 = spy.get("ema50", 0) if spy else 0
        spy_rsi = spy.get("rsi", 50) if spy else 50
        spy_change = spy.get("change_pct", 0) if spy else 0

        index_windows = await self._symbol_windows(db, MACRO_INDICES, params)
        spy_win = index_windows.get("SPY", {})
        spy_r_short = spy_win.get("r_short", 0)
        spy_r_swing = spy_win.get("r_swing", 0)
        spy_r_structural = spy_win.get("r_structural", 0)

        if spy_ema50 > 0 and spy_price > 0:
            dist_from_ema50 = ((spy_price - spy_ema50) / spy_ema50) * 100
            spy_trend_score = 50 + (dist_from_ema50 * 2.5)

            if spy_ema20 > spy_ema50:
                ema_slope = ((spy_ema20 - spy_ema50) / spy_ema50) * 100
                spy_trend_score += min(15, ema_slope * 3)

            # Multi-timeframe: swing pesa di piu', short segnala le svolte
            spy_trend_score += self._clamp(spy_r_swing * 1.2, -12, 12)
            spy_trend_score += self._clamp(spy_r_short * 1.0, -6, 8)
            spy_trend_score += self._clamp(spy_r_structural * 0.4, -6, 8)

            spy_trend_score = self._clamp(spy_trend_score, 5, 95)
        else:
            spy_trend_score = 50

        rsi_distance = abs(spy_rsi - 52)
        rsi_score = self._clamp(100 - (rsi_distance * 1.8), 20, 100)

        # ============================================
        # 2. VOLATILITA'
        # ============================================
        vixy = await db.market_regime.find_one({"symbol": "VIXY"})
        vixy_price = vixy.get("price", 18) if vixy else 18
        vixy_fresh = bool(vixy)

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
        # 3. INDICI: allineamento multi-timeframe
        # ============================================
        indices_bullish_swing = 0
        indices_bullish_short = 0
        indices_total = 0
        swing_sum = 0.0
        short_sum = 0.0

        for sym in ["QQQ", "IWM", "DIA"]:
            w = index_windows.get(sym)
            if not w:
                continue
            indices_total += 1
            swing_sum += w.get("r_swing", 0)
            short_sum += w.get("r_short", 0)
            if w.get("r_swing", 0) > 0:
                indices_bullish_swing += 1
            if w.get("r_short", 0) > 0:
                indices_bullish_short += 1

        if indices_total > 0:
            avg_swing = swing_sum / indices_total
            avg_short = short_sum / indices_total
            alignment_pct = (indices_bullish_swing / indices_total) * 100

            alignment_score = (
                alignment_pct * 0.45
                + self._clamp(50 + avg_swing * 6, 0, 100) * 0.35
                + self._clamp(50 + avg_short * 7, 0, 100) * 0.20
            )
            alignment_score = self._clamp(alignment_score, 5, 95)
        else:
            avg_swing = 0.0
            avg_short = 0.0
            alignment_score = 50

        # ============================================
        # 4. CRYPTO
        # ============================================
        btc = await db.market_regime.find_one({"symbol": "BTC/USD"})
        eth = await db.market_regime.find_one({"symbol": "ETH/USD"})
        btc_change = btc.get("change_pct", 0) if btc else 0
        eth_change = eth.get("change_pct", 0) if eth else 0
        btc_r20 = btc.get("return_20d", 0) if btc else 0
        eth_r20 = eth.get("return_20d", 0) if eth else 0

        crypto_trend = (btc_r20 + eth_r20) / 2
        crypto_short = (btc_change + eth_change) / 2
        crypto_score = self._clamp(50 + (crypto_trend * 2.5) + (crypto_short * 3), 10, 95)

        if crypto_trend > 5:
            crypto_sentiment = "strong_risk_on"
        elif crypto_trend > 1:
            crypto_sentiment = "risk_on"
        elif crypto_trend < -5:
            crypto_sentiment = "risk_off"
        elif crypto_trend < -1:
            crypto_sentiment = "cautious"
        else:
            crypto_sentiment = "neutral"

        # ============================================
        # 5. RISK SYMBOLS multi-timeframe
        # ============================================
        risk_windows = await self._symbol_windows(db, RISK_SYMBOLS, params)

        def swing_of(sym):
            return risk_windows.get(sym, {}).get("r_swing", 0)

        def short_of(sym):
            return risk_windows.get(sym, {}).get("r_short", 0)

        # ---- 5A. Dollaro ----
        uup_swing = swing_of("UUP")
        fxe_swing = swing_of("FXE")
        dollar_net = round(uup_swing - fxe_swing, 2)
        dollar_score = self._clamp(55 - (dollar_net * 4), 15, 90)

        if dollar_net < -1.5:
            dollar_strength = "weak"
        elif dollar_net > 1.5:
            dollar_strength = "strong"
        else:
            dollar_strength = "neutral"

        # ---- 5B. Bond e credito ----
        tlt_swing = swing_of("TLT")
        hyg_swing = swing_of("HYG")
        lqd_swing = swing_of("LQD")
        credit_spread = round(hyg_swing - lqd_swing, 2)

        bonds_score = 60 + (credit_spread * 8) - (max(0, tlt_swing - 1.0) * 4)
        bonds_score = self._clamp(bonds_score, 10, 90)

        if credit_spread < -1.5 and tlt_swing > 1.5:
            bonds_signal = "risk_off"
        elif credit_spread < -1.0:
            bonds_signal = "credit_stress"
        elif credit_spread > 1.0:
            bonds_signal = "risk_on"
        elif tlt_swing > 2.0:
            bonds_signal = "flight_to_safety"
        else:
            bonds_signal = "neutral"

        # ---- 5C. Commodities ----
        gld_swing = swing_of("GLD")
        uso_swing = swing_of("USO")

        commodities_score = 55 + (uso_swing * 2) - (gld_swing * 1.2)
        if spy_r_swing < 0 and gld_swing > 3:
            commodities_score -= 15
        commodities_score = self._clamp(commodities_score, 15, 90)

        if gld_swing > 5 and spy_r_swing < 0:
            commodities_signal = "risk_off"
        elif gld_swing > 3 and uso_swing > 3:
            commodities_signal = "inflation"
        elif gld_swing < -2 and uso_swing > 0:
            commodities_signal = "growth"
        elif gld_swing < 0:
            commodities_signal = "risk_on"
        else:
            commodities_signal = "neutral"

        # ---- 5D. Risk appetite ----
        iwo_swing = swing_of("IWO")
        eem_swing = swing_of("EEM")
        iyt_swing = swing_of("IYT")

        risk_avg = (iwo_swing + eem_swing + iyt_swing) / 3
        risk_appetite_score = self._clamp(50 + (risk_avg * 4), 10, 95)

        if risk_avg > 4:
            risk_appetite = "strong"
        elif risk_avg > 1:
            risk_appetite = "moderate"
        elif risk_avg > -2:
            risk_appetite = "low"
        else:
            risk_appetite = "risk_off"

        # ============================================
        # 6. MARKET LEADERSHIP ENGINE
        # ============================================
        leadership = await self._calc_market_leadership(db, params)

        rsp_doc = await db.market_regime.find_one({"symbol": "RSP"})
        rsp_change = rsp_doc.get("change_pct", 0) if rsp_doc else 0

        rsp_rs_swing = leadership.get("indices", {}).get("RSP", {}).get("rs_swing", 0.0)
        breadth_div_score = self._clamp(60 + (rsp_rs_swing * 8), 10, 95)

        if rsp_rs_swing > 1.0:
            breadth_divergence = "broad_rally"
        elif rsp_rs_swing > -0.5:
            breadth_divergence = "normal"
        elif rsp_rs_swing > -2.5:
            breadth_divergence = "narrow"
        else:
            breadth_divergence = "very_narrow"

        # ============================================
        # 7. SECTOR ROTATION (compatibilita')
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

        sector_leadership = leadership.get("sector_leadership", "UNKNOWN")
        if sector_leadership == "DEFENSIVE_LED":
            rotation_signal = "defensive"
        elif sector_leadership in ("CYCLICAL_LED", "TECH_LED"):
            rotation_signal = "offensive"
        else:
            rotation_signal = "mixed"

        # ============================================
        # 8. BREADTH
        # ============================================
        assets = await db.assets.find({}, {"price": 1, "ema50": 1}).to_list(400)
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
        breadth_score = self._clamp(breadth_pct * 1.4 - 20, 5, 95)

        # La partecipazione misurata dai ratio corregge il breadth puro
        participation_score = leadership.get("participation_score", 50)
        breadth_score = self._clamp(breadth_score * 0.7 + participation_score * 0.3, 5, 95)

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
        # 9. COMPOSITE
        # ============================================
        w = params
        raw_confidence = round(
            spy_trend_score * w.get("w_spy_trend", 0.22) +
            rsi_score * w.get("w_spy_rsi", 0.06) +
            vol_score * w.get("w_vix", 0.12) +
            breadth_score * w.get("w_breadth", 0.18) +
            alignment_score * w.get("w_indices_alignment", 0.12) +
            crypto_score * w.get("w_crypto", 0.03) +
            dollar_score * w.get("w_dollar", 0.05) +
            bonds_score * w.get("w_bonds", 0.10) +
            commodities_score * w.get("w_commodities", 0.04) +
            risk_appetite_score * w.get("w_risk_appetite", 0.08)
        , 1)

        # ---- Smoothing ----
        state_doc = await db.agent_state.find_one({"_id": "macro_regime_state"})
        history = list(state_doc.get("confidence_history", [])) if state_doc else []
        last_regime = state_doc.get("regime") if state_doc else None
        pending_regime = state_doc.get("pending_regime") if state_doc else None
        pending_count = state_doc.get("pending_count", 0) if state_doc else 0

        smoothing = max(1, int(params.get("confidence_smoothing", 3)))
        history.append(raw_confidence)
        history = history[-smoothing:]
        smoothed_confidence = round(sum(history) / len(history), 1)

        # ---- Isteresi ----
        candidate_regime = self._apply_hysteresis(smoothed_confidence, last_regime, params)

        # ---- Conferma su piu' cicli ----
        confirm_cycles = max(1, int(params.get("regime_confirm_cycles", 2)))
        regime_changed = False

        if not last_regime:
            market_regime = candidate_regime
            pending_regime = None
            pending_count = 0
            regime_changed = True
        elif candidate_regime == last_regime:
            market_regime = last_regime
            pending_regime = None
            pending_count = 0
        else:
            if pending_regime == candidate_regime:
                pending_count += 1
            else:
                pending_regime = candidate_regime
                pending_count = 1

            if pending_count >= confirm_cycles:
                market_regime = candidate_regime
                pending_regime = None
                pending_count = 0
                regime_changed = True
            else:
                market_regime = last_regime

        # ============================================
        # 10. REGIME DETTAGLIATO
        # ============================================
        leadership_state = leadership.get("state", "UNKNOWN")
        early_threshold = params.get("early_recovery_short_return", 1.5)
        intraday_threshold = params.get("intraday_strong_move", 1.0)

        # I rendimenti multi-timeframe usano barre CHIUSE, quindi durante la
        # seduta non vedono il movimento del giorno. Il change_pct di SPY e'
        # invece aggiornato live: lo usiamo come condizione alternativa, non
        # sommandolo, per evitare doppi conteggi.
        intraday_strong = spy_change >= intraday_threshold
        short_strong = (
            (spy_r_short >= early_threshold and avg_short > 0)
            or (intraday_strong and spy_r_short > -1.0)
        )
        swing_weak = spy_r_swing < 0
        swing_strong = spy_r_swing > 1.0
        narrow = leadership_state in ("MEGA_CAP_NARROW", "NARROW_NON_TECH")

        regime_detail = market_regime
        detail_reason = "Lettura standard del composite"

        if market_regime == "BULL":
            if narrow or breadth_pct < breadth_weak:
                regime_detail = "NARROW_BULL"
                detail_reason = "Trend positivo ma partecipazione stretta"
            elif spy_r_short < -1.0:
                regime_detail = "PULLBACK_IN_UPTREND"
                detail_reason = "Trend di fondo intatto, debolezza di breve"
            else:
                regime_detail = "BULL"
                detail_reason = "Trend e partecipazione allineati"

        elif market_regime in ("NEUTRAL", "BEAR"):
            if short_strong and swing_weak:
                regime_detail = "EARLY_RECOVERY"
                detail_reason = "Momentum di breve in ripresa su trend ancora debole"
            elif leadership_state == "DEFENSIVE_LED" and not swing_strong:
                regime_detail = "ROTATION"
                detail_reason = "Rotazione verso i settori difensivi"
            elif market_regime == "NEUTRAL" and narrow:
                regime_detail = "NARROW_BULL" if spy_r_swing > 0 else "NEUTRAL"
                detail_reason = "Mercato sostenuto da poche mega cap"

        exposure_map = {
            "BULL": w.get("bull_exposure", 1.0),
            "NARROW_BULL": w.get("narrow_bull_exposure", 0.70),
            "PULLBACK_IN_UPTREND": w.get("pullback_exposure", 0.75),
            "NEUTRAL": w.get("neutral_exposure", 0.60),
            "ROTATION": w.get("rotation_exposure", 0.55),
            "EARLY_RECOVERY": w.get("early_recovery_exposure", 0.55),
            "BEAR": w.get("bear_exposure", 0.35),
            "CRASH": w.get("crash_exposure", 0.0),
        }
        base_exposure = exposure_map.get(market_regime, 0.5)
        exposure_multiplier = exposure_map.get(regime_detail, base_exposure)

        # Regola di prudenza:
        # EARLY_RECOVERY e' l'unico stato che puo' ALZARE l'esposizione, perche'
        # serve proprio a non perdere la svolta dopo una fase debole.
        # Tutti gli altri raffinamenti possono solo ridurla.
        if regime_detail == "EARLY_RECOVERY":
            exposure_multiplier = max(exposure_multiplier, base_exposure)
        else:
            exposure_multiplier = min(exposure_multiplier, base_exposure)

        # ---- Salva lo stato per il prossimo ciclo ----
        await db.agent_state.update_one(
            {"_id": "macro_regime_state"},
            {"$set": {
                "regime": market_regime,
                "regime_detail": regime_detail,
                "confidence_history": history,
                "pending_regime": pending_regime,
                "pending_count": pending_count,
                "raw_confidence": raw_confidence,
                "smoothed_confidence": smoothed_confidence,
                "updated_at": datetime.utcnow(),
            }},
            upsert=True,
        )

        # ============================================
        # 11. CRASH RADAR + SECTOR BOTTOM
        # ============================================
        crash_radar = await self._calc_crash_radar(db, vixy_price)
        if crash_radar["deploy_signal"]:
            print(f"  CRASH RADAR: {crash_radar['crash_level']} "
                  f"(score {crash_radar['crash_risk_score']}, "
                  f"SPY dd {crash_radar['spy_drawdown_pct']}%, VIXY {vixy_price})")

        sector_bottom = await self._calc_sector_bottom_detector(db)
        if sector_bottom["any_bottom"]:
            hot = ", ".join(f"{s['sector']}({s['pct_above_200sma']}%)"
                            for s in sector_bottom["bottom_sectors"])
            print(f"  SECTOR BOTTOM: {hot}")

        # ============================================
        # 12. MARKET CONTEXT
        # ============================================
        market_context = {
            "market_regime": market_regime,
            "regime_detail": regime_detail,
            "regime_detail_reason": detail_reason,
            "regime_confidence": smoothed_confidence,
            "regime_raw_confidence": raw_confidence,
            "regime_changed": regime_changed,
            "regime_pending": pending_regime,
            "regime_pending_count": pending_count,
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
            "crash_radar": crash_radar,
            "sector_bottom": sector_bottom,
            "sector_rankings": sector_rankings,
            "leadership": leadership,
            "timeframes": {
                "short_window": short_w,
                "swing_window": swing_w,
                "structural_window": structural_w,
                "spy": {
                    "intraday": spy_change,
                    "short": spy_r_short,
                    "swing": spy_r_swing,
                    "structural": spy_r_structural,
                },
                "intraday_strong": intraday_strong,
                "short_strong": short_strong,
                "indices": {
                    sym: {
                        "short": data.get("r_short", 0),
                        "swing": data.get("r_swing", 0),
                        "structural": data.get("r_structural", 0),
                    }
                    for sym, data in index_windows.items()
                },
                "indices_bullish_swing": indices_bullish_swing,
                "indices_bullish_short": indices_bullish_short,
                "indices_total": indices_total,
            },
            "details": {
                "spy": {
                    "price": spy_price, "ema20": spy_ema20, "ema50": spy_ema50,
                    "rsi": spy_rsi,
                    "change_pct": spy_change,
                    "return_short": spy_r_short,
                    "return_20d": spy_r_swing,
                    "return_structural": spy_r_structural,
                    "trend_score": round(spy_trend_score, 1),
                    "rsi_score": round(rsi_score, 1),
                },
                "vixy": {"price": vixy_price, "score": round(vol_score, 1), "available": vixy_fresh},
                "indices": {
                    "bullish": indices_bullish_swing,
                    "total": indices_total,
                    "avg_swing": round(avg_swing, 2),
                    "avg_short": round(avg_short, 2),
                    "score": round(alignment_score, 1),
                },
                "crypto": {
                    "btc_change": btc_change, "eth_change": eth_change,
                    "trend_20d": round(crypto_trend, 2),
                    "score": round(crypto_score, 1),
                },
                "dollar": {
                    "uup_swing": uup_swing, "fxe_swing": fxe_swing,
                    "net": dollar_net, "score": round(dollar_score, 1),
                },
                "breadth": {
                    "above_ema50": above_ema50, "total": total_stocks,
                    "pct": breadth_pct,
                    "participation_score": participation_score,
                    "score": round(breadth_score, 1),
                },
                "bonds": {
                    "tlt_swing": tlt_swing, "hyg_swing": hyg_swing,
                    "lqd_swing": lqd_swing, "credit_spread": credit_spread,
                    "signal": bonds_signal, "score": round(bonds_score, 1),
                },
                "commodities": {
                    "gld_swing": gld_swing, "uso_swing": uso_swing,
                    "signal": commodities_signal, "score": round(commodities_score, 1),
                },
                "breadth_div": {
                    "rsp_change": rsp_change,
                    "rsp_rs_swing": rsp_rs_swing,
                    "concentration_swing": leadership.get("concentration_swing", 0),
                    "signal": breadth_divergence,
                    "score": round(breadth_div_score, 1),
                },
                "risk_appetite_detail": {
                    "iwo_swing": iwo_swing, "eem_swing": eem_swing,
                    "iyt_swing": iyt_swing, "signal": risk_appetite,
                    "score": round(risk_appetite_score, 1),
                },
            },
            "analyzed_at": datetime.utcnow().isoformat(),
        }

        # ============================================
        # 13. LLM REASONING
        # ============================================
        from app.services.llm_service import llm_ask, llm_available
        llm_reasoning = None

        if llm_available():
            try:
                sector_line = ", ".join(
                    f"{s['sector']} {s['rs_structural']:+.1f}%"
                    for s in leadership.get("sectors", [])[:5]
                ) or "n/d"

                data_summary = (
                    f"SPY: ${spy_price:.2f} (RSI {spy_rsi:.0f})\n"
                    f"SPY rendimenti: {short_w}g {spy_r_short:+.2f}%, "
                    f"{swing_w}g {spy_r_swing:+.2f}%, {structural_w}g {spy_r_structural:+.2f}%\n"
                    f"Indici {swing_w}g: QQQ {index_windows.get('QQQ', {}).get('r_swing', 0):+.2f}%, "
                    f"IWM {index_windows.get('IWM', {}).get('r_swing', 0):+.2f}%, "
                    f"DIA {index_windows.get('DIA', {}).get('r_swing', 0):+.2f}%, "
                    f"RSP {index_windows.get('RSP', {}).get('r_swing', 0):+.2f}%\n"
                    f"VIXY: ${vixy_price:.1f}\n"
                    f"Breadth: {breadth_pct:.1f}% sopra EMA50\n"
                    f"Leadership: {leadership_state} ({leadership.get('description', '')})\n"
                    f"Concentrazione mega cap: {leadership.get('concentration_swing', 0):+.2f}%\n"
                    f"Settori sopra SPY: {leadership.get('sectors_above_spy', 0)}/"
                    f"{leadership.get('sectors_total', 0)}\n"
                    f"Forza relativa settori {structural_w}g: {sector_line}\n"
                    f"Credito: HYG-LQD {credit_spread:+.2f}%\n"
                    f"Regime calcolato: {market_regime} / {regime_detail} "
                    f"(confidence {smoothed_confidence}, raw {raw_confidence})\n"
                    f"Exposure: {exposure_multiplier}\n"
                    f"--- PROGETTO ALPHA ---\n"
                    f"Crash Radar: {crash_radar['crash_risk_score']}/100 "
                    f"({crash_radar['crash_level']}), SPY drawdown "
                    f"{crash_radar['spy_drawdown_pct']}%\n"
                    f"Settori in capitolazione: "
                    f"{', '.join(s['sector'] for s in sector_bottom['bottom_sectors']) if sector_bottom['any_bottom'] else 'nessuno'}"
                )

                llm_reasoning = llm_ask(
                    system_prompt=(
                        "Sei un analista macro esperto di swing trading. "
                        "Analizza i dati in max 4 frasi in italiano. "
                        "Indica: 1) Il regime e soprattutto CHI sta trainando il mercato "
                        "(mega cap, small cap, difensivi o ciclici), "
                        "2) Se il rialzo e' ampio o stretto e perche' conta, "
                        "3) La differenza tra momentum di breve e trend swing, "
                        "4) Il rischio principale. "
                        "Sii diretto e concreto, senza disclaimer."
                    ),
                    user_prompt=f"Dati di mercato:\n{data_summary}",
                    max_tokens=220,
                    temperature=0.3,
                    agent_name="macro_analyst",
                )
            except Exception as e:
                print(f"  LLM reasoning error: {e}")

        market_context["llm_reasoning"] = llm_reasoning

        await self.log_decision(
            decision_type="regime_assessment",
            data=market_context,
            reasoning=f"Regime={market_regime}/{regime_detail} conf={smoothed_confidence}",
            confidence=smoothed_confidence,
        )

        await db.market_context.update_one(
            {"_id": "latest"},
            {"$set": market_context},
            upsert=True,
        )

        print(f"MacroAnalyst v3.0: {market_regime}/{regime_detail} "
              f"(conf={smoothed_confidence}, raw={raw_confidence}, "
              f"exposure={exposure_multiplier}, breadth={breadth_pct}%, "
              f"leadership={leadership_state})")

        return market_context

    # ==========================================================
    # LEARN
    # ==========================================================

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
        by_regime = {}

        for dec in pending:
            data = dec.get("data", {})
            spy_then = data.get("details", {}).get("spy", {}).get("price", 0)
            regime_then = data.get("market_regime", "NEUTRAL")
            detail_then = data.get("regime_detail", regime_then)

            if spy_then <= 0 or spy_price_now <= 0:
                continue

            actual_return = ((spy_price_now - spy_then) / spy_then) * 100
            total += 1

            was_correct = False

            if detail_then == "EARLY_RECOVERY":
                was_correct = actual_return > 0
            elif detail_then == "NARROW_BULL":
                was_correct = actual_return > -1
            elif detail_then == "PULLBACK_IN_UPTREND":
                was_correct = actual_return > -2
            elif detail_then == "ROTATION":
                was_correct = -3 < actual_return < 3
            elif regime_then == "BULL":
                was_correct = actual_return > 0
            elif regime_then == "BEAR":
                was_correct = actual_return < -1
            elif regime_then == "NEUTRAL":
                was_correct = -2 < actual_return < 3
            elif regime_then == "CRASH":
                was_correct = actual_return < -3

            if was_correct:
                correct += 1

            bucket = by_regime.setdefault(detail_then, {"total": 0, "correct": 0})
            bucket["total"] += 1
            if was_correct:
                bucket["correct"] += 1

            outcome = {
                "correct": was_correct,
                "spy_price_then": spy_then,
                "spy_price_now": spy_price_now,
                "actual_return_pct": round(actual_return, 2),
                "regime_predicted": regime_then,
                "regime_detail_predicted": detail_then,
            }
            await self.record_outcome(str(dec["_id"]), outcome)

        accuracy = (correct / total * 100) if total > 0 else 50

        for name, bucket in by_regime.items():
            bucket["accuracy"] = round(
                bucket["correct"] / bucket["total"] * 100, 1
            ) if bucket["total"] > 0 else 0

        await self.save_params(params)
        await self.save_performance({
            "accuracy": round(accuracy, 1),
            "total_evaluated": total,
            "by_regime": by_regime,
        })

        return {
            "total_evaluated": total,
            "correct": correct,
            "accuracy": round(accuracy, 1),
            "by_regime": by_regime,
        }
