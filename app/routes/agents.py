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
            "llm_reasoning": result.get("llm_reasoning"),
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


# ============================================
# 🆕 v4.0 — APM ENDPOINTS
# ============================================

@router.get("/apm/decisions")
async def apm_decisions(limit: int = 30):
    """
    🆕 v4.0 — Ritorna le ultime decisioni APM.
    Include: HOLD, SCALE_OUT, EXIT, TIGHTEN_STOP.
    """
    from app.db.mongodb import get_db
    from datetime import datetime
    
    db = get_db()
    
    # Leggi decisioni da agent_decisions_adaptive_position_manager
    col = db["agent_decisions_adaptive_position_manager"]
    decisions = await col.find().sort("created_at", -1).to_list(limit)
    
    # Serializza + estrai info chiave
    result = []
    for d in decisions:
        data = d.get("data", {})
        action_taken = data.get("action_taken", False)
        
        result.append({
            "id": str(d.get("_id", "")),
            "created_at": d.get("created_at").isoformat() if d.get("created_at") else "",
            "type": d.get("type", "unknown"),  # apm_hold, apm_exit, apm_scale_out, apm_tighten_stop
            "ticker": data.get("ticker", ""),
            "decision": data.get("decision", "UNKNOWN"),
            "reason": data.get("reason", ""),
            "current_pnl_pct": data.get("current_pnl_pct", 0),
            "current_price": data.get("current_price", 0),
            "entry_price": data.get("entry_price", 0),
            "action_taken": action_taken,
            "action_details": data.get("action_details", {}),
            "state_snapshot": data.get("state_snapshot", {}),
            "confidence": d.get("confidence", 0),
        })
    
    return {
        "total": len(result),
        "decisions": result,
        "fetched_at": datetime.utcnow().isoformat(),
    }


@router.get("/apm/status")
async def apm_status():
    """
    🆕 v4.0 — Ritorna stato APM (last run, prossima esecuzione).
    """
    from app.db.mongodb import get_db
    from datetime import datetime, timedelta
    
    db = get_db()
    
    # Ultimo run APM
    last_run_doc = await db.apm_state.find_one({"_id": "last_run"})
    
    # Params APM
    params_doc = await db.agent_memory_adaptive_position_manager.find_one({"_id": "params"})
    
    if not last_run_doc:
        return {
            "status": "never_run",
            "last_run": None,
            "next_check": None,
            "enabled": params_doc.get("apm_enabled", True) if params_doc else True,
            "interval_hours": params_doc.get("apm_check_interval_hours", 3) if params_doc else 3,
        }
    
    last_run = last_run_doc.get("timestamp")
    interval_hours = params_doc.get("apm_check_interval_hours", 3) if params_doc else 3
    
    next_check = last_run + timedelta(hours=interval_hours) if last_run else None
    remaining = None
    if next_check:
        remaining_seconds = (next_check - datetime.utcnow()).total_seconds()
        remaining = round(remaining_seconds / 3600, 2) if remaining_seconds > 0 else 0
    
    return {
        "status": "active",
        "last_run": last_run.isoformat() if last_run else None,
        "last_decisions_count": last_run_doc.get("decisions_count", 0),
        "last_actions_count": last_run_doc.get("actions_count", 0),
        "next_check": next_check.isoformat() if next_check else None,
        "remaining_hours": remaining,
        "enabled": params_doc.get("apm_enabled", True) if params_doc else True,
        "interval_hours": interval_hours,
    }


@router.get("/apm/summary")
async def apm_summary(days: int = 7):
    """
    🆕 v4.0 — Riepilogo statistiche APM negli ultimi N giorni.
    """
    from app.db.mongodb import get_db
    from datetime import datetime, timedelta
    
    db = get_db()
    cutoff = datetime.utcnow() - timedelta(days=days)
    
    col = db["agent_decisions_adaptive_position_manager"]
    decisions = await col.find({
        "created_at": {"$gte": cutoff},
    }).to_list(500)
    
    # Conta per tipo
    counts = {"HOLD": 0, "SCALE_OUT": 0, "EXIT": 0, "TIGHTEN_STOP": 0, "SKIP": 0}
    actions_taken = 0
    total_pnl_managed = 0
    
    for d in decisions:
        data = d.get("data", {})
        decision = data.get("decision", "SKIP")
        counts[decision] = counts.get(decision, 0) + 1
        
        if data.get("action_taken", False):
            actions_taken += 1
        
        total_pnl_managed += abs(data.get("current_pnl_pct", 0))
    
    return {
        "period_days": days,
        "total_decisions": len(decisions),
        "actions_taken": actions_taken,
        "counts": counts,
        "avg_pnl_managed": round(total_pnl_managed / max(len(decisions), 1), 2),
    }
