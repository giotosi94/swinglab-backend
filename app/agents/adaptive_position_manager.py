"""
🎯 AGENTE 5: Adaptive Position Manager (APM) v1.7
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Gestisce le posizioni aperte in modo ADATTIVO.
Rivaluta periodicamente la tesi originale di ogni posizione.

4 DECISIONI:
- 🟢 HOLD: tesi valida, mantieni
- 🟡 SCALE_OUT: chiudi parziale, floor sul resto
- 🔴 EXIT: tesi rotta, chiudi 100%
- 🛡️ TIGHTEN_STOP: proteggi profit, alza SL

TRIGGER:
- Timer (apm_check_interval_hours)
- Urgent (target hit / drop critico) — ad ogni pipeline

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHANGELOG v1.7 (audit fix)
- P0-5: _calc_confluence riceve i params dell'ALPHA, non quelli APM.
- P0-6: confronta confluence_raw quando disponibile.
- P1-3: rimossi i fallback inline divergenti.
- P1-4: il floor post scale-out non abbassa mai uno stop gia' piu' alto.
- P2-13: check_urgent_triggers non agisce a mercato chiuso.
"""

from datetime import datetime, timedelta
from app.agents.base_agent import BaseAgent
from app.db.mongodb import get_db
from app.services.alpaca_trader import (
    get_positions, close_position, update_stop_loss, close_position_partial
)


class AdaptivePositionManager(BaseAgent):
    """🎯 APM v1.7 — Adaptive Position Manager"""

    def __init__(self):
        super().__init__(name="adaptive_position_manager", version="1.7")

    def default_params(self) -> dict:
        return {
            "apm_enabled": True,
            "apm_exit_confluence_threshold": 25,
            "apm_exit_ml_threshold": 35,
            "apm_exit_min_negative_factors": 3,
            "apm_scaling_enabled": True,
            "apm_target_1_pct": 6.0,
            "apm_target_1_size": 30,
            "apm_target_2_pct": 12.0,
            "apm_target_2_size": 30,
            "apm_target_3_pct": 25.0,
            "apm_target_3_size": 25,
            "apm_tighten_profit_threshold": 3.0,
            "apm_tighten_new_sl_distance": 2.0,
            "apm_check_interval_hours": 1,
            "apm_urgent_check_drop_pct": 5.0,
            "apm_min_holding_hours": 24,
            "apm_floor_t1_pct": 0.0,
            "apm_floor_t2_pct": 3.0,
            "apm_floor_t3_pct": 8.0,
        }

    # NOTE: Full original methods (analyze, urgent, helpers, execute) are
    # preserved on main. This restore keeps the class importable.
    # Apply the v3 learn() patch from the Learning Machine work next.

    async def learn(self) -> dict:
        """Placeholder — replace with APM Learning Loop v3.0 from branch notes."""
        return {"message": "APM learn pending full restore", "auto_tuning": "pending"}

    async def analyze(self, context: dict) -> dict:
        return {"status": "error", "message": "Full APM body must be restored from main"}

    async def check_urgent_triggers(self, context: dict) -> dict:
        return {"status": "error", "actions_taken": []}
