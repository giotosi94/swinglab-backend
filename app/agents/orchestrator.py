from datetime import datetime
import gc
import time

from app.agents.macro_analyst import MacroAnalyst
from app.agents.alpha_strategist import AlphaStrategist
from app.agents.risk_manager import RiskManager
from app.agents.executor import Executor
from app.agents.adaptive_position_manager import AdaptivePositionManager
from app.agents.shared_brain import brain
from app.services.alpaca_trader import get_account, get_positions
from app.db.mongodb import get_db


APM_ASSET_PROJECTION = {
    "price_history": 0,
    "vp_distribution": 0,
    "multi_tf_vp": 0,
    "max_strategy": 0,
    "alpha_snapshot": 0,
    "llm_analysis": 0,
    "history": 0,
}


class Orchestrator:
    """ORCHESTRATOR v2.2 — SharedBrain + Professional Report"""

    def __init__(self):
        self.macro = MacroAnalyst()
        self.alpha = AlphaStrategist()
        self.risk = RiskManager()
        self.executor = Executor()
        self.apm = AdaptivePositionManager()

        self.agents = {
            "macro_analyst": self.macro,
            "alpha_strategist": self.alpha,
            "risk_manager": self.risk,
            "executor": self.executor,
            "adaptive_position_manager": self.apm,
        }

    async def run(self) -> dict:
        db = get_db()
        pipeline_start = time.time()
        report = {"steps": {}, "errors": [], "timing": {}}

        print("=" * 60)
        print("SWINGLAB MULTI-AGENT PIPELINE v2.2 (Professional Report)")
        print("=" * 60)

        # STEP 0: Alpaca
        t0 = time.time()
        try:
            account = await get_account()
            positions = await get_positions() or []
            if not account:
                return {"error": "Alpaca not connected", "steps": {}}
            equity = float(account.get("equity", 0))
            cash = float(account.get("cash", 0))
            print(f"Account: equity=${equity:.2f}, cash=${cash:.2f}, positions={len(positions)}")
        except Exception as e:
            return {"error": f"Alpaca error: {str(e)}", "steps": {}}
        report["timing"]["alpaca_fetch"] = round(time.time() - t0, 2)

        # STEP 1: Macro
        t1 = time.time()
        try:
            market_context = await self.macro.analyze()
            report["steps"]["macro_analyst"] = {
                "status": "ok",
                "regime": market_context.get("market_regime"),
                "regime_detail": market_context.get("regime_detail"),
                "confidence": market_context.get("regime_confidence"),
                "exposure": market_context.get("exposure_multiplier"),
                "breadth": market_context.get("breadth_pct"),
                "volatility": market_context.get("volatility_regime"),
                "leadership": market_context.get("leadership", {}).get("state") if isinstance(market_context.get("leadership"), dict) else market_context.get("leadership"),
            }
            try:
                await brain.write_market({
                    "regime": market_context.get("market_regime", "UNKNOWN"),
                    "regime_detail": market_context.get("regime_detail", "UNKNOWN"),
                    "confidence": market_context.get("regime_confidence", 0),
                    "exposure_multiplier": market_context.get("exposure_multiplier", 0.5),
                    "volatility": market_context.get("volatility_regime", "UNKNOWN"),
                    "breadth_pct": market_context.get("breadth_pct", 0),
                    "rotation": market_context.get("rotation_signal", "unknown"),
                    "leadership": market_context.get("leadership", {}).get("state", "UNKNOWN") if isinstance(market_context.get("leadership"), dict) else market_context.get("leadership", "UNKNOWN"),
                    "sector_rankings": market_context.get("sector_rankings", []),
                    "llm_reasoning": market_context.get("llm_reasoning"),
                })
                print("  Brain: market state written")
            except Exception as be:
                print(f"  Brain write error (market): {be}")
        except Exception as e:
            report["errors"].append(f"MacroAnalyst: {str(e)}")
            report["steps"]["macro_analyst"] = {"status": "error", "error": str(e)}
            print(f"MacroAnalyst ERROR: {e}")
            market_context = {
                "market_regime": "NEUTRAL",
                "regime_detail": "NEUTRAL",
                "regime_confidence": 50,
                "exposure_multiplier": 0.5,
                "sector_rankings": [],
            }
        report["timing"]["macro_analyst"] = round(time.time() - t1, 2)

        # STEP 2: Alpha
        t2 = time.time()
        alpha_result = {}
        try:
            alpha_result = await self.alpha.analyze({
                "market_context": market_context,
                "positions": positions,
            })
            buy_candidates = alpha_result.get("buy_candidates", [])
            sell_signals = alpha_result.get("sell_signals", [])
            report["steps"]["alpha_strategist"] = {
                "status": "ok",
                "buy_candidates": len(buy_candidates),
                "sell_signals": len(sell_signals),
                "top_picks": [c["ticker"] for c in buy_candidates[:5]],
                "sells": [s["ticker"] for s in sell_signals],
                "summary": alpha_result.get("summary", {}),
            }
            try:
                await brain.write_candidates(buy_candidates, sell_signals)
                print(f"  Brain: {len(buy_candidates)} candidates + {len(sell_signals)} sells written")
            except Exception as be:
                print(f"  Brain write error (candidates): {be}")
        except Exception as e:
            report["errors"].append(f"AlphaStrategist: {str(e)}")
            report["steps"]["alpha_strategist"] = {"status": "error", "error": str(e)}
            buy_candidates = []
            sell_signals = []
            print(f"AlphaStrategist ERROR: {e}")
        report["timing"]["alpha_strategist"] = round(time.time() - t2, 2)

        # STEP 3: Risk
        t3 = time.time()
        try:
            risk_result = await self.risk.analyze({
                "market_context": market_context,
                "buy_candidates": buy_candidates,
                "sell_signals": sell_signals,
                "account": account,
                "positions": positions,
            })
            approved_trades = risk_result.get("approved_trades", [])
            approved_sells = risk_result.get("approved_sells", [])
            rejected_trades = risk_result.get("rejected_trades", [])
            risk_report = risk_result.get("risk_report", {})
            report["steps"]["risk_manager"] = {
                "status": "ok",
                "approved_trades": len(approved_trades),
                "rejected_trades": len(rejected_trades),
                "approved_sells": len(approved_sells),
                "approved_tickers": [t["ticker"] for t in approved_trades],
                "rejected_reasons": [{"ticker": t["ticker"], "reason": t["reason"]} for t in rejected_trades[:10]],
                "risk_report": risk_report,
            }
            try:
                await brain.write_approved(approved_trades, approved_sells, risk_report)
                print(f"  Brain: {len(approved_trades)} approved + {len(approved_sells)} sells written")
            except Exception as be:
                print(f"  Brain write error (approved): {be}")
        except Exception as e:
            report["errors"].append(f"RiskManager: {str(e)}")
            report["steps"]["risk_manager"] = {"status": "error", "error": str(e)}
            approved_trades = []
            approved_sells = []
            rejected_trades = []
            risk_report = {}
            print(f"RiskManager ERROR: {e}")
        report["timing"]["risk_manager"] = round(time.time() - t3, 2)

        # STEP 3.4: APM Urgent
        t_urgent = time.time()
        try:
            urgent_result = await self.apm.check_urgent_triggers({"positions": positions})
            urgent_actions = len(urgent_result.get("actions_taken", []))
            if urgent_actions > 0:
                print(f"  APM URGENT: {urgent_actions} actions triggered")
                positions = await get_positions() or []
            report["steps"]["apm_urgent"] = {"status": "ok", "actions_taken": urgent_actions}
        except Exception as e:
            report["errors"].append(f"APM Urgent: {str(e)}")
            report["steps"]["apm_urgent"] = {"status": "error", "error": str(e)}
            print(f"APM URGENT ERROR: {e}")
        report["timing"]["apm_urgent"] = round(time.time() - t_urgent, 2)

        # STEP 3.5: APM full
        t_apm = time.time()
        apm_result = {}
        try:
            apm_ml_map = alpha_result.get("ml_map") if isinstance(alpha_result, dict) else None
            if apm_ml_map:
                print(f"  APM: ml_map riusata da Alpha ({len(apm_ml_map)} ticker)")
            else:
                apm_ml_map = {}
                apm_assets = None
                try:
                    apm_assets = await db.assets.find({}, APM_ASSET_PROJECTION).to_list(400)
                    apm_ml_map = await self.alpha._load_ml_predictions(db, apm_assets, market_context)
                except Exception as e:
                    print(f"  APM ml_map load error: {e}")
                finally:
                    if apm_assets is not None:
                        del apm_assets
                    gc.collect()
            apm_result = await self.apm.analyze({
                "market_context": market_context,
                "positions": positions,
                "ml_map": apm_ml_map,
            })
            apm_status = apm_result.get("status", "unknown")
            actions_taken = len(apm_result.get("actions_taken", []))
            report["steps"]["adaptive_position_manager"] = {
                "status": "ok",
                "apm_status": apm_status,
                "actions_taken": actions_taken,
                "summary": apm_result.get("summary", {}),
            }
            if apm_status == "ok":
                print(f"  APM: {actions_taken} actions taken")
            elif apm_status == "skipped_timer":
                print(f"  APM skipped: {apm_result.get('message', 'timer not elapsed')}")
            if actions_taken > 0:
                positions = await get_positions() or []
            del apm_ml_map
        except Exception as e:
            report["errors"].append(f"APM: {str(e)}")
            report["steps"]["adaptive_position_manager"] = {"status": "error", "error": str(e)}
            print(f"APM ERROR: {e}")
        report["timing"]["adaptive_position_manager"] = round(time.time() - t_apm, 2)

        # STEP 4: Executor
        t4 = time.time()
        try:
            exec_result = await self.executor.analyze({
                "market_context": market_context,
                "approved_trades": approved_trades,
                "approved_sells": approved_sells,
            })
            report["steps"]["executor"] = {
                "status": "ok",
                "executed_buys": len(exec_result.get("executed_buys", [])),
                "executed_sells": len(exec_result.get("executed_sells", [])),
                "failed_orders": len(exec_result.get("failed_orders", [])),
                "cancelled_stale": exec_result.get("cancelled_stale", 0),
                "market_status": exec_result.get("market_status", {}),
                "message": exec_result.get("message"),
                "details": exec_result,
            }
            try:
                await brain.write_executions(
                    executed_buys=exec_result.get("executed_buys", []),
                    executed_sells=exec_result.get("executed_sells", []),
                    details={
                        "failed_orders": exec_result.get("failed_orders", []),
                        "cancelled_stale": exec_result.get("cancelled_stale", 0),
                        "market_status": exec_result.get("market_status", {}),
                        "llm_reasoning": exec_result.get("llm_reasoning"),
                    },
                )
                await brain.clear_approved()
                print("  Brain: executions written + approved cleared")
            except Exception as be:
                print(f"  Brain write error (executions): {be}")
        except Exception as e:
            report["errors"].append(f"Executor: {str(e)}")
            report["steps"]["executor"] = {"status": "error", "error": str(e)}
            exec_result = {}
            print(f"Executor ERROR: {e}")
        report["timing"]["executor"] = round(time.time() - t4, 2)

        # Crash Deploy
        try:
            from app.services.crash_deploy_live import check_and_deploy
            deploy_res = await check_and_deploy()
            report["steps"]["crash_deploy"] = deploy_res
        except Exception as e:
            report["errors"].append(f"CrashDeploy: {str(e)}")
            report["steps"]["crash_deploy"] = {"status": "error", "error": str(e)}

        # Save pipeline state
        total_time = round(time.time() - pipeline_start, 2)
        report["timing"]["total"] = total_time
        state = {
            "last_run": datetime.utcnow().isoformat(),
            "equity": equity,
            "cash": cash,
            "positions": len(positions),
            "actions": (
                [{"action": "BUY", **t} for t in exec_result.get("executed_buys", [])] +
                [{"action": "SELL", **s} for s in exec_result.get("executed_sells", [])]
            ),
            "market": {
                "regime": market_context.get("market_regime", "UNKNOWN"),
                "regime_detail": market_context.get("regime_detail", "UNKNOWN"),
                "confidence": market_context.get("regime_confidence", 0),
                "exposure_multiplier": market_context.get("exposure_multiplier", 0.5),
                "volatility": market_context.get("volatility_regime", "UNKNOWN"),
                "breadth_pct": market_context.get("breadth_pct", 0),
            },
            "risk_report": risk_report,
            "updated_at": datetime.utcnow(),
        }
        await db.auto_trader.update_one({"_id": "alpaca_state"}, {"$set": state}, upsert=True)

        print(f"\n{'=' * 60}")
        print(f"PIPELINE COMPLETE in {total_time}s")
        print(f"   Regime: {market_context.get('market_regime')}/{market_context.get('regime_detail')}")
        print(f"   Buys: {len(exec_result.get('executed_buys', []))} | Sells: {len(exec_result.get('executed_sells', []))}")
        if report["errors"]:
            print(f"   Errors: {report['errors']}")
        print(f"{'=' * 60}")

        # Professional Trader Report (genera solo se contesto cambiato o stale)
        try:
            from app.services.professional_report import generate_and_save_if_needed
            leadership_obj = market_context.get("leadership") or {}
            leadership_state = leadership_obj.get("state") if isinstance(leadership_obj, dict) else leadership_obj
            apm_summary = {}
            if isinstance(apm_result, dict):
                apm_summary = apm_result.get("summary") or {
                    "status": apm_result.get("status"),
                    "actions_taken": len(apm_result.get("actions_taken", [])),
                }
            risk_llm = risk_report.get("llm_reasoning") if isinstance(risk_report, dict) else None
            report_ctx = {
                "regime": market_context.get("market_regime"),
                "regime_detail": market_context.get("regime_detail"),
                "confidence": market_context.get("regime_confidence"),
                "exposure_multiplier": market_context.get("exposure_multiplier"),
                "breadth_pct": market_context.get("breadth_pct"),
                "volatility": market_context.get("volatility_regime"),
                "leadership": leadership_state,
                "rotation": market_context.get("rotation_signal") or market_context.get("rotation"),
                "focus_sectors": market_context.get("focus_sectors") or [],
                "avoid_sectors": market_context.get("avoid_sectors") or [],
                "macro_llm": market_context.get("llm_reasoning"),
                "risk_llm": risk_llm,
                "apm_summary": apm_summary,
                "top_picks": report.get("steps", {}).get("alpha_strategist", {}).get("top_picks", []),
                "sells": report.get("steps", {}).get("alpha_strategist", {}).get("sells", []),
                "approved_count": len(approved_trades),
                "rejected_count": len(rejected_trades),
                "executed_buys": len(exec_result.get("executed_buys", [])),
                "executed_sells": len(exec_result.get("executed_sells", [])),
            }
            pro_res = await generate_and_save_if_needed(report_ctx, force=False)
            report["professional_report_status"] = pro_res.get("status")
            if pro_res.get("text"):
                report["professional_report"] = pro_res["text"][:120] + "..."
        except Exception as e:
            print(f"  Professional report error: {e}")
            report["errors"].append(f"ProfessionalReport: {str(e)}")

        alpha_result = None
        gc.collect()
        return report

    async def learn_all(self) -> dict:
        print("\nLEARNING LOOP — All Agents")
        results = {}
        for name, agent in self.agents.items():
            try:
                result = await agent.learn()
                results[name] = {"status": "ok", "result": result}
            except Exception as e:
                results[name] = {"status": "error", "error": str(e)}
                print(f"  {name} learn error: {e}")
        gc.collect()
        return results

    async def get_status(self) -> dict:
        db = get_db()
        status = {}
        for name, agent in self.agents.items():
            params = await agent.get_params()
            recent = await agent.get_recent_decisions(limit=5)
            perf = await agent.get_performance_history(limit=5)
            status[name] = {
                "params": params,
                "recent_decisions": recent,
                "performance": perf,
            }
        state = await db.auto_trader.find_one({"_id": "alpaca_state"})
        if state:
            state["_id"] = str(state["_id"])
        status["pipeline_state"] = state
        return status
