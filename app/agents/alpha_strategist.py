from datetime import datetime
from app.agents.base_agent import BaseAgent
from app.db.mongodb import get_db
from app.services.max_strategy import LIVE_MAX_POLICY, live_cycle_blocked
from app.services.llm_shadow import run_max_strategy_llm_shadow


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
    "max_strategy.live_policy": 1,
    "max_strategy.cycle_gate.upper_block": 1,
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
    "max_strategy.entry_plan.atr14": 1,
    "max_strategy.atr14": 1,
}

# Barre necessarie al calcolo ATR: 14 periodi piu' la chiusura precedente.
ATR_BARS = 20


class AlphaStrategist(BaseAgent):
    """
    AGENTE 2: Alpha Strategist v2.3 — Stock Picker with ML + Leadership + LLM Shadow

    v2.3 — LLM Shadow (log-only) su Max Strategy candidates.
    """

    def __init__(self):
        super().__init__(name="alpha_strategist", version="2.3")
        self.MAX_RAW_CONFLUENCE = 25.0

    def default_params(self) -> dict:
        return {
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
                "ml_winloss": 0.8,
                "trend_predictor": 0.7,
                "mtf_alignment": 1.0,
                "poc_shift": 1.0,
            },
            "min_confluence": 48,
            "max_rsi_entry": 68,
            "min_rsi_entry": 25,
            "min_price": 2.0,
            "max_relative_volume": 3.0,
            "max_per_sector": 2,
            "best_setups": ["pullback_to_poc", "ema_bounce", "breakout", "oversold_reversal"],
            "worst_setups": [],
            "weak_sectors": [],
            "sell_rsi_extreme": 78,
            "sell_score_collapsed": 20,
            "sell_min_pnl_for_rsi_sell": 3.0,
            "ml_winloss_threshold_strong": 0.75,
            "ml_winloss_threshold_medium": 0.60,
            "trend_confidence_threshold_strong": 0.60,
            "trend_confidence_threshold_medium": 0.50,
            "sell_ml_loss_threshold": 0.30,
            "leadership_enabled": True,
            "focus_threshold_discount": 3.0,
            "avoid_threshold_premium": 5.0,
            "outflow_threshold_premium": 3.0,
            "min_confluence_floor": 42,
            "leadership_sort_weight": 2.0,
            "weak_sector_penalty_enabled": False,
            "max_live_enabled": False,
            "max_live_slots": 4,
            "max_stop_cap_pct": 8.0,
            "max_target_atr_mult": 4.0,
            "trend_live_enabled": False,
            "trend_live_slots": 4,
            "trend_max_from_high_pct": 10.0,
            "trend_max_rsi": 80.0,
            "trend_rs_lookback": 126,
        }
