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

@router.get("/apm-history")
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
    
# ============================================
# 🧬 FASE 3 — APM LEARNING LOOP ENDPOINT
# ============================================

@router.post("/apm/learn")
async def apm_learn():
    """
    🧬 Trigger manuale del Learning Loop APM.
    
    Analizza le decisioni degli ultimi 30 giorni, auto-aggiusta le soglie
    e manda report Telegram.
    
    Configurabile come cron settimanale (domenica 03:00 CEST).
    """
    orch = _get_orch()
    try:
        result = await orch.apm.learn()
        return {
            "status": "ok",
            "learning_result": result,
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }



@router.post("/apm/force-run")
async def apm_force_run():
    """
    🔧 Forza APM a rieseguire immediatamente, ignorando il timer 3h.
    Utile per debug/testing.
    """
    from app.db.mongodb import get_db
    from datetime import datetime
    
    db = get_db()
    await db.apm_state.delete_one({"_id": "last_run"})
    
    return {
        "status": "reset",
        "message": "APM timer reset. Next pipeline will run APM.",
        "reset_at": datetime.utcnow().isoformat(),
    }


# ============================================
# v4.5 — DPS + KELLY ANALYTICS ENDPOINT
# ============================================

@router.get("/dps/status")
async def dps_kelly_status():
    """Ritorna stato completo DPS + Kelly: params, trade stats, sizing history."""
    from app.db.mongodb import get_db
    from datetime import datetime, timedelta
    
    db = get_db()
    
    # 1. Params correnti DPS + Kelly
    risk_params = await db.agent_memory_risk_manager.find_one({"_id": "params"})
    params = risk_params or {}
    
    # 2. Kelly calcolo su trade history
    trades = await db.trade_history.find({
        "side": "sell",
        "pnl_pct": {"$exists": True}
    }).sort("date", -1).limit(100).to_list(100)
    
    n_trades = len(trades)
    kelly_min = params.get("kelly_min_trades", 20)
    kelly_active = n_trades >= kelly_min
    
    wins = [t for t in trades if t.get("pnl_pct", 0) > 0]
    losses = [t for t in trades if t.get("pnl_pct", 0) <= 0]
    
    win_rate = (len(wins) / n_trades * 100) if n_trades > 0 else 0
    avg_win = (sum(t.get("pnl_pct", 0) for t in wins) / len(wins)) if wins else 0
    avg_loss = abs(sum(t.get("pnl_pct", 0) for t in losses) / len(losses)) if losses else 0
    
    kelly_pct = 0
    if avg_win > 0 and avg_loss > 0 and wins and losses:
        wr = win_rate / 100
        lr = 1 - wr
        kelly_pct = (wr * avg_win - lr * avg_loss) / avg_win * 100
    
    fractional_factor = params.get("kelly_fractional_factor", 0.25)
    fractional_kelly = max(0, kelly_pct * fractional_factor)
    
    # 3. Ultimo risk report (per multipliers correnti)
    alpaca_state = await db.auto_trader.find_one({"_id": "alpaca_state"})
    latest_risk = alpaca_state.get("risk_report", {}) if alpaca_state else {}
    
    # 4. Ultime 30 decisions trade_approved con DPS info
    decisions = await db.agent_decisions_risk_manager.find({
        "type": "trade_approved",
        "created_at": {"$gte": datetime.utcnow() - timedelta(days=30)}
    }).sort("created_at", -1).limit(30).to_list(30)
    
    sizing_history = []
    for d in decisions:
        data = d.get("data", {})
        sizing_history.append({
            "ticker": data.get("ticker"),
            "notional_usd": data.get("notional_usd", 0),
            "dps_multiplier": data.get("dps_multiplier", 1.0),
            "kelly_multiplier": data.get("kelly_multiplier", 1.0),
            "confluence": data.get("confluence", 0),
            "risk_reward": data.get("risk_reward", 0),
            "date": d.get("created_at").isoformat() if d.get("created_at") else "",
        })
    
    # 5. Distribution multipliers
    multipliers_dist = {}
    for s in sizing_history:
        mult = s["dps_multiplier"]
        bucket = round(mult * 10) / 10  # 0.1 precision
        multipliers_dist[str(bucket)] = multipliers_dist.get(str(bucket), 0) + 1
    
    avg_dps_mult = sum(s["dps_multiplier"] for s in sizing_history) / len(sizing_history) if sizing_history else 1.0
    avg_kelly_mult = sum(s["kelly_multiplier"] for s in sizing_history) / len(sizing_history) if sizing_history else 1.0
    
    return {
        "params": {
            "dps_enabled": params.get("dps_enabled", True),
            "dps_rr_ideal": params.get("dps_rr_ideal", 2.5),
            "dps_ml_ideal": params.get("dps_ml_ideal", 75.0),
            "dps_conf_ideal": params.get("dps_conf_ideal", 55.0),
            "dps_max_multiplier": params.get("dps_max_multiplier", 1.6),
            "dps_aggressiveness": params.get("dps_aggressiveness", 1.0),
            "kelly_enabled": params.get("kelly_enabled", True),
            "kelly_min_trades": kelly_min,
            "kelly_fractional_factor": fractional_factor,
        },
        "kelly_status": {
            "active": kelly_active,
            "n_trades": n_trades,
            "trades_needed": max(0, kelly_min - n_trades),
            "win_rate": round(win_rate, 1),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "kelly_pct": round(kelly_pct, 2),
            "fractional_kelly": round(fractional_kelly, 2),
            "current_multiplier": latest_risk.get("kelly_multiplier", 1.0),
        },
        "current_risk_report": {
            "kelly_multiplier": latest_risk.get("kelly_multiplier", 1.0),
            "position_size_pct": latest_risk.get("position_size_pct", 12.0),
            "risk_per_trade_usd": latest_risk.get("risk_per_trade_usd", 0),
            "final_multiplier": latest_risk.get("final_multiplier", 0.6),
        },
        "sizing_history": sizing_history,
        "multipliers_distribution": multipliers_dist,
        "stats": {
            "avg_dps_multiplier": round(avg_dps_mult, 3),
            "avg_kelly_multiplier": round(avg_kelly_mult, 3),
            "total_approved_30d": len(sizing_history),
        },
        "generated_at": datetime.utcnow().isoformat(),
    }
