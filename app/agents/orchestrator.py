from datetime import datetime
import time
from app.agents.macro_analyst import MacroAnalyst
from app.agents.alpha_strategist import AlphaStrategist
from app.agents.risk_manager import RiskManager
from app.agents.executor import Executor
from app.agents.adaptive_position_manager import AdaptivePositionManager
from app.agents.shared_brain import brain
from app.services.alpaca_trader import get_account, get_positions
from app.db.mongodb import get_db


class Orchestrator:
    """
    🎯 ORCHESTRATOR v2.0 — con SharedBrain integrato
    Coordina il pipeline dei 4 agenti in sequenza:
    MacroAnalyst → AlphaStrategist → RiskManager → Executor

    Ogni agente riceve l'output del precedente come contesto
    e scrive lo stato sul SharedBrain MongoDB.
    Questo permette agli LLM reasoning di leggere il contesto
    degli altri agenti per analisi più intelligenti.
    """

    def __init__(self):
        self.macro = MacroAnalyst()
        self.alpha = AlphaStrategist()
        self.risk = RiskManager()
        self.executor = Executor()
        # 🆕 v4.0 — APM (Adaptive Position Manager)
        self.apm = AdaptivePositionManager()
        self.agents = {
            "macro_analyst": self.macro,
            "alpha_strategist": self.alpha,
            "risk_manager": self.risk,
            "executor": self.executor,
            "adaptive_position_manager": self.apm,  # 🆕 v4.0
        }

    async def run(self) -> dict:
        """
        Esegue il pipeline completo dei 4 agenti.
        Scrive su SharedBrain dopo ogni step.
        Ritorna il report completo con il risultato di ogni agente.
        """
        db = get_db()
        pipeline_start = time.time()
        report = {"steps": {}, "errors": [], "timing": {}}

        print("=" * 60)
        print("🤖 SWINGLAB MULTI-AGENT PIPELINE v2.0 (with SharedBrain)")
        print("=" * 60)

        # ============================================
        # STEP 0: Fetch Alpaca account & positions
        # ============================================
        t0 = time.time()
        try:
            account = await get_account()
            positions = await get_positions() or []
            if not account:
                return {"error": "Alpaca not connected", "steps": {}}
            equity = float(account.get("equity", 0))
            cash = float(account.get("cash", 0))
            print(f"💰 Account: equity=${equity:.2f}, cash=${cash:.2f}, "
                  f"positions={len(positions)}")
        except Exception as e:
            return {"error": f"Alpaca error: {str(e)}", "steps": {}}
        report["timing"]["alpaca_fetch"] = round(time.time() - t0, 2)

        # ============================================
        # STEP 1: 🌍 MacroAnalyst → SharedBrain
        # ============================================
        t1 = time.time()
        try:
            market_context = await self.macro.analyze()
            report["steps"]["macro_analyst"] = {
                "status": "ok",
                "regime": market_context.get("market_regime"),
                "confidence": market_context.get("regime_confidence"),
                "exposure": market_context.get("exposure_multiplier"),
                "breadth": market_context.get("breadth_pct"),
                "volatility": market_context.get("volatility_regime"),
            }

            # 🆕 Scrive sul SharedBrain
            try:
                await brain.write_market({
                    "regime": market_context.get("market_regime", "UNKNOWN"),
                    "confidence": market_context.get("regime_confidence", 0),
                    "exposure_multiplier": market_context.get("exposure_multiplier", 0.5),
                    "volatility": market_context.get("volatility_regime", "UNKNOWN"),
                    "breadth_pct": market_context.get("breadth_pct", 0),
                    "rotation": market_context.get("rotation_signal", "unknown"),
                    "sector_rankings": market_context.get("sector_rankings", []),
                    "llm_reasoning": market_context.get("llm_reasoning"),
                })
                print("  🧠 Brain: market state written")
            except Exception as be:
                print(f"  ⚠️ Brain write error (market): {be}")

        except Exception as e:
            report["errors"].append(f"MacroAnalyst: {str(e)}")
            report["steps"]["macro_analyst"] = {"status": "error", "error": str(e)}
            print(f"❌ MacroAnalyst ERROR: {e}")
            # Fallback: usa context minimo
            market_context = {
                "market_regime": "NEUTRAL",
                "regime_confidence": 50,
                "exposure_multiplier": 0.5,
                "sector_rankings": [],
            }
        report["timing"]["macro_analyst"] = round(time.time() - t1, 2)

        # ============================================
        # STEP 2: 🎯 AlphaStrategist → SharedBrain
        # ============================================
        t2 = time.time()
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

            # 🆕 Scrive sul SharedBrain
            try:
                await brain.write_candidates(buy_candidates, sell_signals)
                print(f"  🧠 Brain: {len(buy_candidates)} candidates + {len(sell_signals)} sells written")
            except Exception as be:
                print(f"  ⚠️ Brain write error (candidates): {be}")

        except Exception as e:
            report["errors"].append(f"AlphaStrategist: {str(e)}")
            report["steps"]["alpha_strategist"] = {"status": "error", "error": str(e)}
            buy_candidates = []
            sell_signals = []
            print(f"❌ AlphaStrategist ERROR: {e}")
        report["timing"]["alpha_strategist"] = round(time.time() - t2, 2)

        # ============================================
        # STEP 3: 🛡️ RiskManager → SharedBrain
        # ============================================
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
                "rejected_reasons": [
                    {"ticker": t["ticker"], "reason": t["reason"]}
                    for t in rejected_trades[:10]
                ],
                "risk_report": risk_report,
            }

            # 🆕 Scrive sul SharedBrain
            try:
                await brain.write_approved(approved_trades, approved_sells, risk_report)
                print(f"  🧠 Brain: {len(approved_trades)} approved + {len(approved_sells)} sells written")
            except Exception as be:
                print(f"  ⚠️ Brain write error (approved): {be}")

        except Exception as e:
            report["errors"].append(f"RiskManager: {str(e)}")
            report["steps"]["risk_manager"] = {"status": "error", "error": str(e)}
            approved_trades = []
            approved_sells = []
            rejected_trades = []
            risk_report = {}
            print(f"❌ RiskManager ERROR: {e}")
        report["timing"]["risk_manager"] = round(time.time() - t3, 2)

        # ============================================
        # 🆕 v4.0 — STEP 3.5: 🎯 APM (Adaptive Position Manager)
        # Rivaluta posizioni aperte ogni 3h e decide HOLD/SCALE/EXIT/TIGHTEN
        # ============================================
        t_apm = time.time()
        apm_result = {}
        try:
            # Prepara ml_map per APM (riuso quello di Alpha)
            from app.agents.alpha_strategist import AlphaStrategist
            apm_ml_map = {}
            try:
                apm_alpha = self.alpha
                # Reload ml_map se disponibile
                apm_assets = await db.assets.find({}, {"ticker": 1}).to_list(300)
                apm_ml_map = await apm_alpha._load_ml_predictions(db, apm_assets, market_context)
            except Exception as e:
                print(f"  ⚠️ APM ml_map load error: {e}")
            
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
                print(f"  🎯 APM: {actions_taken} actions taken")
            elif apm_status == "skipped_timer":
                print(f"  ⏳ APM skipped: {apm_result.get('message', 'timer not elapsed')}")
            elif apm_status == "disabled":
                print(f"  ⏸️ APM disabled in settings")
            
            # Ricarica positions se APM ha fatto azioni (potrebbe aver chiuso qualcosa)
            if actions_taken > 0:
                positions = await get_positions() or []
                print(f"  📊 Reloaded positions: {len(positions)} still open")
        except Exception as e:
            report["errors"].append(f"APM: {str(e)}")
            report["steps"]["adaptive_position_manager"] = {"status": "error", "error": str(e)}
            print(f"❌ APM ERROR: {e}")
        report["timing"]["adaptive_position_manager"] = round(time.time() - t_apm, 2)

        # ============================================
        # STEP 4: ⚡ Executor → SharedBrain
        # ============================================
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
                "details": exec_result,
            }

            # 🆕 Scrive sul SharedBrain
            try:
                await brain.write_executions(
                    executed_buys=exec_result.get("executed_buys", []),
                    executed_sells=exec_result.get("executed_sells", []),
                    details={
                        "failed_orders": exec_result.get("failed_orders", []),
                        "cancelled_stale": exec_result.get("cancelled_stale", 0),
                        "trailing_adjustments": exec_result.get("trailing_adjustments", []),
                        "synced_trades": exec_result.get("synced_trades", 0),
                        "market_status": exec_result.get("market_status", {}),
                        "llm_reasoning": exec_result.get("llm_reasoning"),
                    },
                )
                # Dopo l'esecuzione, pulisce gli approved dal brain
                # (così al prossimo run non vengono riusati per errore)
                await brain.clear_approved()
                print(f"  🧠 Brain: executions written + approved cleared")
            except Exception as be:
                print(f"  ⚠️ Brain write error (executions): {be}")

        except Exception as e:
            report["errors"].append(f"Executor: {str(e)}")
            report["steps"]["executor"] = {"status": "error", "error": str(e)}
            exec_result = {}
            print(f"❌ Executor ERROR: {e}")
        report["timing"]["executor"] = round(time.time() - t4, 2)

        # ============================================
        # STEP 5: Save pipeline state (backward compat con auto_trader)
        # ============================================
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
                "confidence": market_context.get("regime_confidence", 0),
                "exposure_multiplier": market_context.get("exposure_multiplier", 0.5),
                "volatility": market_context.get("volatility_regime", "UNKNOWN"),
                "breadth_pct": market_context.get("breadth_pct", 0),
                "rotation": market_context.get("rotation_signal", "unknown"),
            },
            "pipeline": {
                "steps": {k: v.get("status") for k, v in report["steps"].items()},
                "timing": report["timing"],
                "errors": report["errors"],
            },
            "risk_report": risk_report,
            "updated_at": datetime.utcnow(),
        }

        await db.auto_trader.update_one(
            {"_id": "alpaca_state"}, {"$set": state}, upsert=True
        )

        print(f"\n{'=' * 60}")
        print(f"🏁 PIPELINE COMPLETE in {total_time}s")
        print(f"   Regime: {market_context.get('market_regime')} "
              f"(conf={market_context.get('regime_confidence')})")
        print(f"   Buys: {len(exec_result.get('executed_buys', []))} | "
              f"Sells: {len(exec_result.get('executed_sells', []))} | "
              f"Rejected: {len(rejected_trades)}")
        if report["errors"]:
            print(f"   ⚠️ Errors: {report['errors']}")
        print(f"{'=' * 60}")

        return report

    async def learn_all(self) -> dict:
        """Esegue il learning loop per tutti gli agenti."""
        print("\n🧬 LEARNING LOOP — All Agents")
        results = {}
        for name, agent in self.agents.items():
            try:
                result = await agent.learn()
                results[name] = {"status": "ok", "result": result}
            except Exception as e:
                results[name] = {"status": "error", "error": str(e)}
                print(f"  ❌ {name} learn error: {e}")
        return results

    async def get_status(self) -> dict:
        """Ritorna lo stato di tutti gli agenti."""
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

        # Pipeline state
        pipeline_state = await db.auto_trader.find_one({"_id": "alpaca_state"})
        if pipeline_state:
            pipeline_state["_id"] = str(pipeline_state["_id"])

        # 🆕 Aggiunge anche shared brain
        brain_state = await brain.get_full_state()

        return {
            "agents": status,
            "pipeline_state": pipeline_state,
            "shared_brain": brain_state,
        }
