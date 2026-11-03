from fastapi import APIRouter
from app.agents.orchestrator import Orchestrator
from app.agents.shared_brain import brain
from datetime import datetime

router = APIRouter()

_orch = None

def _get_orch():
    global _orch
    if _orch is None:
        _orch = Orchestrator()
    return _orch


@router.get("/status")
async def agents_status():
    orch = _get_orch()
    status = await orch.get_status()
    status["shared_brain"] = await brain.get_full_state()
    return status


@router.post("/run")
async def run_pipeline():
    orch = _get_orch()
    return await orch.run()


LEARN_ALL_ENABLED = True


@router.post("/learn")
async def learn_all(force: bool = False):
    if not LEARN_ALL_ENABLED and not force:
        return {
            "status": "disabled",
            "message": "Learning Loop disabilitato. Usa force=true per forzare.",
        }

    orch = _get_orch()
    results = await orch.learn_all()

    # Guardrail post-learn (Alpha): valuta PRIMA, poi aggiorna baseline
    try:
        alpha = orch.agents.get("alpha_strategist")
        if alpha and hasattr(alpha, "evaluate_guardrail"):
            ares = (results.get("alpha_strategist") or {}).get("result") or {}
            wr = ares.get("win_rate")
            npos = ares.get("total_positions") or ares.get("positions") or ares.get("total_trades")
            if wr is not None:
                guard = await alpha.evaluate_guardrail(
                    current_win_rate=float(wr),
                    current_sample_count=int(npos or 0) if npos is not None else None,
                )
                results.setdefault("alpha_strategist", {}).setdefault("result", {})["guardrail"] = guard
                if guard.get("action") == "rollback":
                    try:
                        from app.services.telegram_bot import send_telegram
                        rb = guard.get("rollback") or {}
                        await send_telegram(
                            "⚠️ <b>ROLLBACK Alpha</b>\n"
                            f"v{rb.get('from_version')} → hist v{rb.get('restored_history_version')} "
                            f"(new v{rb.get('new_param_version')})\n"
                            f"WR {guard.get('current_wr')}% < baseline {guard.get('baseline_wr')}% "
                            f"(drop {guard.get('drop_pts')} pts)"
                        )
                    except Exception as te:
                        print(f"  Guardrail telegram error: {te}")
                elif hasattr(alpha, "set_learn_baseline"):
                    ver = ares.get("param_version")
                    if ver is None:
                        raw = await alpha.get_params_raw()
                        ver = raw.get("param_version")
                    await alpha.set_learn_baseline(
                        win_rate=float(wr),
                        sample_count=int(npos or 0),
                        param_version=ver,
                        extra={"source": "learn_all", "guardrail": guard.get("action")},
                    )
    except Exception as ge:
        print(f"  Guardrail post-learn error: {ge}")
        results["guardrail_error"] = str(ge)

    try:
        from app.services.telegram_bot import send_telegram

        msg = "🧬 <b>Learning Machine Report</b>\n\n"
        msg += f"⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC\n\n"

        for name, res in results.items():
            status = res.get("status", "?")
            if status == "ok":
                r = res.get("result") or {}
                if name == "alpha_strategist":
                    g = r.get("guardrail") or {}
                    msg += (
                        f"🎯 <b>Alpha</b>: WR {r.get('win_rate', '?')}% | "
                        f"min_conf {r.get('min_confluence', '?')} | "
                        f"trades {r.get('total_trades') or r.get('total_positions', 0)}"
                    )
                    if g.get("action"):
                        msg += f" | guardrail={g.get('action')}"
                    msg += "\n"
                elif name == "risk_manager":
                    msg += (
                        f"🛡️ <b>Risk</b>: WR {r.get('win_rate', '?')}% | "
                        f"risk% {r.get('risk_pct_per_trade', '?')} | "
                        f"minRR {r.get('min_risk_reward', '?')}\n"
                    )
                elif name == "adaptive_position_manager":
                    auto = r.get("auto_tuning", "?")
                    adj = r.get("adjustments") or []
                    msg += (
                        f"🧬 <b>APM</b>: auto={auto} | "
                        f"decisions {r.get('total_decisions', 0)} | "
                        f"adj {len(adj)}\n"
                    )
                elif name == "macro_analyst":
                    msg += (
                        f"🌍 <b>Macro</b>: accuracy {r.get('accuracy', '?')}% | "
                        f"eval {r.get('total_evaluated', 0)}\n"
                    )
                elif name == "executor":
                    msg += (
                        f"⚡ <b>Executor</b>: fill_rate {r.get('fill_rate', '?')}% | "
                        f"buffer {r.get('buffer', '?')}%\n"
                    )
                else:
                    msg += f"• <b>{name}</b>: ok\n"
            else:
                msg += f"• <b>{name}</b>: ❌ {res.get('error', 'error')}\n"

        msg += "\n✅ Learning completato. Parametri versionati in history."
        await send_telegram(msg)
    except Exception as e:
        print(f"  Learning Telegram summary error: {e}")

    return {
        "status": "ok",
        "learning_enabled": True,
        "results": results,
        "learned_at": datetime.utcnow().isoformat(),
    }


@router.post("/professional-report")
async def generate_desk_brief():
    try:
        from app.services.professional_report import (
            build_context_from_brain,
            generate_professional_report,
            save_professional_report,
        )
        ctx = await build_context_from_brain()
        text = generate_professional_report(ctx)
        if not text:
            return {
                "status": "skipped",
                "message": "LLM non disponibile o risposta vuota",
                "context_preview": {
                    "regime": ctx.get("regime"),
                    "focus": ctx.get("focus_sectors"),
                    "avoid": ctx.get("avoid_sectors"),
                },
            }
        await save_professional_report(text, meta={
            "regime": ctx.get("regime"),
            "source": "on_demand",
        })
        return {
            "status": "ok",
            "professional_report": text,
            "generated_at": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.post("/macro/run")
async def run_macro():
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
    orch = _get_orch()
    try:
        market = await brain.get_market()
        if not market:
            return {"status": "error", "message": "No market data. Run macro first."}
        from app.services.alpaca_trader import get_positions
        positions = await get_positions() or []
        result = await orch.alpha.analyze({
            "market_context": {
                "market_regime": market.get("regime"),
                "regime_detail": market.get("regime_detail"),
                "regime_confidence": market.get("confidence"),
                "exposure_multiplier": market.get("exposure_multiplier"),
                "volatility_regime": market.get("volatility"),
                "breadth_pct": market.get("breadth_pct"),
                "rotation_signal": market.get("rotation"),
                "sector_rankings": market.get("sector_rankings", []),
                "llm_reasoning": market.get("llm_reasoning"),
            },
            "positions": positions,
        })
        await brain.write_candidates(
            result.get("buy_candidates", []),
            result.get("sell_signals", []),
        )
        return {"status": "ok", "agent": "alpha_strategist", "result": result}
    except Exception as e:
        return {"status": "error", "agent": "alpha_strategist", "error": str(e)}


@router.post("/risk/run")
async def run_risk():
    orch = _get_orch()
    try:
        market = await brain.get_market()
        candidates = await brain.get_candidates()
        from app.services.alpaca_trader import get_account, get_positions
        account = await get_account()
        positions = await get_positions() or []
        result = await orch.risk.analyze({
            "market_context": market or {},
            "buy_candidates": (candidates or {}).get("buy", []),
            "sell_signals": (candidates or {}).get("sell", []),
            "account": account,
            "positions": positions,
        })
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
    orch = _get_orch()
    try:
        market = await brain.get_market()
        approved = await brain.get_approved()
        result = await orch.executor.analyze({
            "market_context": market or {},
            "approved_trades": (approved or {}).get("trades", []),
            "approved_sells": (approved or {}).get("sells", []),
        })
        await brain.write_executions(
            executed_buys=result.get("executed_buys", []),
            executed_sells=result.get("executed_sells", []),
            details={
                "failed_orders": result.get("failed_orders", []),
                "cancelled_stale": result.get("cancelled_stale", 0),
                "market_status": result.get("market_status", {}),
            },
        )
        await brain.clear_approved()
        return {"status": "ok", "agent": "executor", "result": result}
    except Exception as e:
        return {"status": "error", "agent": "executor", "error": str(e)}


@router.get("/brain")
async def get_brain():
    return await brain.get_full_state()


@router.get("/{agent_name}/decisions")
async def agent_decisions(agent_name: str, limit: int = 20):
    orch = _get_orch()
    agent = orch.agents.get(agent_name)
    if not agent:
        return {"error": f"Agent '{agent_name}' not found"}
    decisions = await agent.get_recent_decisions(limit=limit)
    return {"agent": agent_name, "decisions": decisions}


@router.get("/{agent_name}/performance")
async def agent_performance(agent_name: str, limit: int = 30):
    orch = _get_orch()
    agent = orch.agents.get(agent_name)
    if not agent:
        return {"error": f"Agent '{agent_name}' not found"}
    perf = await agent.get_performance_history(limit=limit)
    return {"agent": agent_name, "performance": perf}


@router.get("/{agent_name}/params")
async def agent_params(agent_name: str):
    orch = _get_orch()
    agent = orch.agents.get(agent_name)
    if not agent:
        return {"error": f"Agent '{agent_name}' not found"}
    params = await agent.get_params()
    return {"agent": agent_name, "params": params}


@router.get("/{agent_name}/guardrail")
async def agent_guardrail_status(agent_name: str):
    orch = _get_orch()
    agent = orch.agents.get(agent_name)
    if not agent:
        return {"error": f"Agent '{agent_name}' not found"}
    baseline = await agent.get_learn_baseline() if hasattr(agent, "get_learn_baseline") else {}
    perf = await agent.get_performance_history(limit=1) if hasattr(agent, "get_performance_history") else []
    current_wr = None
    if perf:
        m = (perf[0] or {}).get("metrics") or {}
        current_wr = m.get("win_rate")
    status = {
        "agent": agent_name,
        "baseline": baseline,
        "latest_performance": perf[0] if perf else None,
        "thresholds": {
            "min_days": getattr(agent, "guardrail_min_days", 7),
            "min_new_positions": getattr(agent, "guardrail_min_new_positions", 5),
            "wr_drop_pts": getattr(agent, "guardrail_wr_drop_pts", 12.0),
        },
    }
    if current_wr is not None and baseline.get("win_rate") is not None:
        drop = round(float(baseline["win_rate"]) - float(current_wr), 1)
        status["drop_pts"] = drop
        status["would_rollback"] = drop >= getattr(agent, "guardrail_wr_drop_pts", 12.0)
    return status


@router.post("/{agent_name}/rollback")
async def agent_rollback(agent_name: str):
    orch = _get_orch()
    agent = orch.agents.get(agent_name)
    if not agent:
        return {"error": f"Agent '{agent_name}' not found"}
    if not hasattr(agent, "rollback_to_previous_version"):
        return {"error": "Agent does not support rollback"}
    result = await agent.rollback_to_previous_version(reason="manual_rollback")
    try:
        from app.services.telegram_bot import send_telegram
        if result.get("rolled_back"):
            await send_telegram(
                f"↩️ <b>Rollback manuale {agent_name}</b>\n"
                f"v{result.get('from_version')} → hist v{result.get('restored_history_version')} "
                f"(new v{result.get('new_param_version')})"
            )
    except Exception:
        pass
    return {"agent": agent_name, **result}


@router.get("/{agent_name}/param-history")
async def agent_param_history(agent_name: str, limit: int = 20):
    orch = _get_orch()
    agent = orch.agents.get(agent_name)
    if not agent:
        return {"error": f"Agent '{agent_name}' not found"}
    if not hasattr(agent, "get_param_history"):
        return {"error": "Agent does not support param history"}
    history = await agent.get_param_history(limit=limit)
    return {"agent": agent_name, "history": history}


@router.get("/apm-history")
async def apm_decisions(limit: int = 30):
    from app.db.mongodb import get_db
    db = get_db()
    col = db["agent_decisions_adaptive_position_manager"]
    decisions = await col.find().sort("created_at", -1).to_list(limit)
    for d in decisions:
        d["_id"] = str(d["_id"])
    return {"decisions": decisions}


@router.get("/apm/status")
async def apm_status():
    from app.db.mongodb import get_db
    db = get_db()
    params_doc = await db.agent_memory_adaptive_position_manager.find_one({"_id": "params"})
    return {
        "enabled": params_doc.get("apm_enabled", True) if params_doc else True,
        "interval_hours": params_doc.get("apm_check_interval_hours", 3) if params_doc else 3,
        "params": params_doc or {},
    }


@router.get("/apm/summary")
async def apm_summary(days: int = 7):
    from app.db.mongodb import get_db
    from datetime import timedelta
    db = get_db()
    col = db["agent_decisions_adaptive_position_manager"]
    cutoff = datetime.utcnow() - timedelta(days=days)
    decisions = await col.find({"created_at": {"$gte": cutoff}}).to_list(500)
    actions = [d for d in decisions if d.get("decision") and d.get("decision") != "HOLD"]
    return {
        "days": days,
        "total_decisions": len(decisions),
        "actions_taken": len(actions),
    }


@router.post("/apm/learn")
async def apm_learn():
    orch = _get_orch()
    try:
        result = await orch.apm.learn()
        return {"status": "ok", "learning_result": result}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.post("/apm/force-run")
async def apm_force_run():
    orch = _get_orch()
    try:
        from app.services.alpaca_trader import get_positions
        positions = await get_positions() or []
        market = await brain.get_market() or {}
        result = await orch.apm.analyze({
            "market_context": market,
            "positions": positions,
            "force": True,
        })
        return {"status": "ok", "result": result}
    except Exception as e:
        return {"status": "error", "error": str(e)}
