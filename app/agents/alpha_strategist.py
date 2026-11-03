from datetime import datetime
from app.agents.base_agent import BaseAgent
from app.db.mongodb import get_db
from app.services.max_strategy import LIVE_MAX_POLICY, live_cycle_blocked


# ==========================================================
# PROIEZIONI MONGODB — controllo della memoria
# ==========================================================
ASSET_PROJECTION = {
    "price_history": 0,
    "vp_distribution": 0,
    "multi_tf_vp": 0,
    "max_strategy": 0,
    "alpha_snapshot": 0,
    "history": 0,
}

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

ATR_BARS = 20


class AlphaStrategist(BaseAgent):
    """Alpha Strategist — temporary minimal restore. Full file will be restored next."""

    def __init__(self):
        super().__init__(name="alpha_strategist", version="2.3")
        self.MAX_RAW_CONFLUENCE = 25.0

    def default_params(self) -> dict:
        return {
            "min_confluence": 48,
            "max_rsi_entry": 68,
            "min_rsi_entry": 25,
            "min_price": 2.0,
            "max_relative_volume": 3.0,
            "max_per_sector": 2,
            "leadership_enabled": True,
            "min_confluence_floor": 42,
            "max_live_enabled": False,
        }

    async def analyze(self, context: dict) -> dict:
        # Minimal safe stub so pipeline does not crash while full file is restored
        return {
            "buy_candidates": [],
            "sell_signals": [],
            "error": "AlphaStrategist temporarily in safe stub — restore full file",
        }
