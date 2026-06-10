from fastapi import APIRouter
from app.agents.orchestrator import Orchestrator
from app.agents.shared_brain import brain

router = APIRouter()

_orch = None

def _get_orch():
    global _orch
    if _orch is None:
        _orch = Orchestrator()
    return _orch


# ============================================
# FULL PIPELINE (backward compatible)
# ============================================

@router.get("/status")
async def agents_status():
    """Stato completo di tutti gli agenti + shared brain."""
    orch = _get_orch()
    status = await orch.get_status()
    status["shared_brain"] = await brain.get_full_state()
    return status


@router.post("/run")
async def run_pipeline():
    """Pipeline completo (tutti gli agenti in sequenza)."""
    orch = _get_orch()
    return await orch.run()


@router.post("/learn")
async def learn_all():
    """Learning per tutti gli agenti."""
    orch = _get_orch()
    return await orch.learn_all()


# ============================================
# INDIVIDUAL AGENT ENDPOINTS
# ============================================

@router.post("/macro/run")
async def run_macro():
    """Run solo MacroAnalyst → scrive market state nel shared brain."""
    orch = _get_orch()
    try:
        result = await orch.macro.analyze()
        await brain.write_market({
            "regime": result.get("market_regime", "UNKNOWN"),
            "confidence": result.get("regime_confidence", 0),
            "exposure_multiplier": result.get("exposure_multiplier", 0.5),
            "volatility": result.get("volatility_regime", "UNKNOWN"),
            "breadth_pct": result.get("breadth_pct", 0),
            "rotation": result.get("rotation_signal", "unknown"),
            "sector_rankings": result.get("sector_rankings", []),
        })
        return {"status": "ok", "agent": "macro_analyst", "result": result}
    except Exception as e:
        return {"status": "error", "agent": "macro_analyst", "error": str(e)}


@router.post("/alpha/run")
async def run_alpha():
    """Run solo AlphaStrategist → legge market dal brain, scrive candidates."""
    orch = _get_orch()
    try:
        # Legge market state dal brain
        market = await brain.get_market()
        if not market:
            return {"status": "error", "message": "No market data. Run macro first."}

        from app.services.alpaca_trader import get_positions
        positions = await get_positions() or []

        result = await orch.alpha.analyze({
            "market_context": market,
            "positions": positions,
        })

        # Scrive candidates nel brain
        await brain.write_candidates(
            result.get("buy_candidates", []),
            result.get("sell_signals", []),
        )

        return {"status": "ok", "agent": "alpha_strategist", "result": result}
    except Exception as e:
        return {"status": "error", "agent": "alpha_strategist", "error": str(e)}


@router.post("/risk/run")
async def run_risk():
    """Run solo RiskManager → legge market+candidates dal brain, scrive approved."""
    orch = _get_orch()
    try:
        market = await brain.get_market()
        candidates_data = await brain.get_candidates()

        if not candidates_data.get("buy") and not candidates_data.get("sell"):
            return {"status": "ok", "message": "No candidates to evaluate."}

        from app.services.alpaca_trader import get_account, get_positions
        account = await get_account()
        positions = await get_positions() or []

        if not account:
            return {"status": "error", "message": "Alpaca not connected"}

        result = await orch.risk.analyze({
            "market_context": market,
            "buy_candidates": candidates_data.get("buy", []),
            "sell_signals": candidates_data.get("sell", []),
            "account": account,
            "positions": positions,
        })

        # Scrive approved nel brain
        await brain.write_approved(
            result.get("approved_trades", []),
            result.get("approved_sells", []),
            result.get("risk_report", {}),
        )

        return {"status": "ok", "agent": "risk_manager", "result": result}
    except Exception as e:
        return {"status": "error", "agent": "risk_manager", "error": str(e)}


@router.post("/executor/run")
async def run_executor():
    """Run solo Executor → legge approved dal brain, esegue, pulisce."""
    orch = _get_orch()
    try:
        market = await brain.get_market()
        approved = await brain.get_approved()

        approved_trades = approved.get("trades", [])
        approved_sells = approved.get("sells", [])

        if not approved_trades and not approved_sells:
            return {"status": "ok", "message": "No approved trades to execute."}

        result = await orch.executor.analyze({
            "market_context": market,
            "approved_trades": approved_trades,
            "approved_sells": approved_sells,
        })

        # Scrive executions nel brain
        await brain.write_executions(
            result.get("executed_buys", []),
            result.get("executed_sells", []),
            result,
        )

        # Pulisce gli approved dopo l'esecuzione
        await brain.clear_approved()

        return {"status": "ok", "agent": "executor", "result": result}
    except Exception as e:
        return {"status": "error", "agent": "executor", "error": str(e)}


# ============================================
# SHARED BRAIN ENDPOINT
# ============================================

@router.get("/brain")
async def get_brain():
    """Legge lo stato completo del shared brain."""
    return await brain.get_full_state()


# ============================================
# EXISTING ENDPOINTS (unchanged)
# ============================================

@router.get("/{agent_name}/decisions")
async def agent_decisions(agent_name: str, limit: int = 20):
    """Ultime decisioni di un agente specifico."""
    orch = _get_orch()
    agent = orch.agents.get(agent_name)
    if not agent:
        return {"error": f"Agent '{agent_name}' not found"}
    decisions = await agent.get_recent_decisions(limit=limit)
    return {"agent": agent_name, "decisions": decisions}


@router.get("/{agent_name}/performance")
async def agent_performance(agent_name: str, limit: int = 30):
    """Storico performance di un agente."""
    orch = _get_orch()
    agent = orch.agents.get(agent_name)
    if not agent:
        return {"error": f"Agent '{agent_name}' not found"}
    perf = await agent.get_performance_history(limit=limit)
    return {"agent": agent_name, "performance": perf}


@router.get("/{agent_name}/params")
async def agent_params(agent_name: str):
    """Parametri appresi di un agente."""
    orch = _get_orch()
    agent = orch.agents.get(agent_name)
    if not agent:
        return {"error": f"Agent '{agent_name}' not found"}
    params = await agent.get_params()
    return {"agent": agent_name, "params": params}
