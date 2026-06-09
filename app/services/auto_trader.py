"""
auto_trader.py — Wrapper retrocompatibile.
Ora usa il Multi-Agent Orchestrator internamente.
Le route /api/data/autotrader/* continuano a funzionare come prima.
"""
from datetime import datetime
from app.db.mongodb import get_db
from app.agents.orchestrator import Orchestrator


# Singleton orchestrator
_orchestrator = None

def _get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator


async def run_auto_trader():
    """Esegue il pipeline multi-agent (sostituisce il vecchio auto_trader)."""
    orch = _get_orchestrator()
    return await orch.run()


async def get_auto_trader_state():
    """Ritorna lo stato del pipeline (backward compat)."""
    db = get_db()
    state = await db.auto_trader.find_one({"_id": "alpaca_state"})
    if state:
        state["_id"] = str(state["_id"])
    return state


async def reset_auto_trader(initial_capital=10000):
    """Reset completo: chiude posizioni e pulisce history e brain di tutti gli agenti."""
    from app.services.alpaca_trader import close_all_positions, cancel_all_orders
    await cancel_all_orders()
    await close_all_positions()

    db = get_db()
    # Pulisci trade history
    await db.trade_history.delete_many({})

    # Pulisci TUTTI i brain e le decisioni degli agenti
    for agent_name in ["macro_analyst", "alpha_strategist", "risk_manager", "executor"]:
        await db[f"agent_memory_{agent_name}"].delete_many({})
        await db[f"agent_decisions_{agent_name}"].delete_many({})
        await db[f"agent_performance_{agent_name}"].delete_many({})

    # Pulisci anche il vecchio agent_brain (retrocompatibilita')
    await db.agent_brain.delete_many({})
    await db.agent_decisions.delete_many({})

    return {"message": f"All positions closed, all agent brains and history cleared"}
