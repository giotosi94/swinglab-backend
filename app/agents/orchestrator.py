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


# Campi esclusi dal reload asset per l'APM.
# max_strategy e' diventato il blocco piu' pesante del documento
# (weekly_context, daily_confirmation, execution_4h, structural_base,
# entry_plan, profiles POC): su 300 asset erano oltre 10 MB inutili,
# perche' il modello ML non li usa.
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
    """
    ORCHESTRATOR v2.1 — con SharedBrain integrato

    Coordina il pipeline degli agenti in sequenza:
    MacroAnalyst -> AlphaStrategist -> RiskManager -> APM -> Executor

    Ogni agente riceve l'output del precedente come contesto
    e scrive lo stato sul SharedBrain MongoDB.

    v2.0 -> v2.1: ottimizzazione memoria.
      - Il reload degli asset per l'APM escludeva solo tre campi e
        riportava in RAM 300 documenti completi, max_strategy incluso.
      - Se l'AlphaStrategist espone gia' la propria ml_map, viene riusata
        invece di ricalcolarla.
      - Gli oggetti grandi vengono liberati esplicitamente a fine step.
    """

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
        """
        Esegue il pipeline completo.
        Scrive su SharedBrain dopo ogni step.
        Ritorna il report con il risultato di ogni agente.
        """
        db = get_db()
        pipeline_start = time.time()
        report = {"steps": {}, "errors": [], "timing": {}}

        print("=" * 60)
        print("SWINGLAB MULTI-AGENT PIPELINE v2.1 (with SharedBrain)")
        print("=" * 60)

        # ============================================
        # STEP 0: Alpaca account e posizioni
        # ============================================
        t0 = time.time()
        try:
            account = await get_account()
            positions = await get_positions() or []

            if not account:
                return {"error": "Alpaca not connected", "steps": {}}

            equity = float(account.get("equity", 0))
            cash = float(account.get("cash", 0))

            print(f"Account: equity=${equity:.2f}, cash=${cash:.2f}, "
                  f"positions={len(positions)}")

        except Exception as e:
            return {"error": f"Alpaca error: {str(e)}", "steps": {}}

        report["timing"]["alpaca_fetch"] = round(time.time() - t0, 2)

        # ============================================
        # STEP 1: MacroAnalyst -> SharedBrain
        # ============================================
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
                "leadership": market_context.get("leadership", {}).get("state"),
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
                    "leadership": market_context.get("leadership", {}).get("state", "UNKNOWN"),
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

        # ============================================
        # STEP 2: AlphaStrategist -> SharedBrain
        # ============================================
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
                print(f"  Brain: {len(buy_candidates)} candidates + "
                      f"{len(sell_signals)} sells written")
            except Exception as be:
                print(f"  Brain write error (candidates): {be}")

        except Exception as e:
            report["errors"].append(f"AlphaStrategist: {str(e)}")
            report["steps"]["alpha_strategist"] = {"status": "error", "error": str(e)}
            buy_candidates = []
            sell_signals = []
            print(f"AlphaStrategist ERROR: {e}")

        report["timing"]["alpha_strategist"] = round(time.time() - t2, 2)

        # ============================================
        # STEP 3: RiskManager -> SharedBrain
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

            try:
                await brain.write_approved(approved_trades, approved_sells, risk_report)
                print(f"  Brain: {len(approved_trades)} approved + "
                      f"{len(approved_sells)} sells written")
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

        # ============================================
        # STEP 3.4: APM URGENT TRIGGERS
        # Check veloce solo su target hit o drop critico.
        # Bypassa il timer orario: gira ad ogni pipeline.
        # ============================================
        t_urgent = time.time()
        urgent_result = {}

        try:
            urgent_result = await self.apm.check_urgent_triggers({
                "positions": positions,
            })

            urgent_actions = len(urgent_result.get("actions_taken", []))

            if urgent_actions > 0:
                print(f"  APM URGENT: {urgent_actions} actions triggered")
                positions = await get_positions() or []

            report["steps"]["apm_urgent"] = {
                "status": "ok",
                "actions_taken": urgent_actions,
            }

        except Exception as e:
            report["errors"].append(f"APM Urgent: {str(e)}")
            report["steps"]["apm_urgent"] = {"status": "error", "error": str(e)}
            print(f"APM URGENT ERROR: {e}")

        report["timing"]["apm_urgent"] = round(time.time() - t_urgent, 2)

        # ============================================
        # STEP 3.5: APM (Adaptive Position Manager)
        # Rivaluta le posizioni aperte e decide
        # HOLD / SCALE / EXIT / TIGHTEN
        # ============================================
        t_apm = time.time()
        apm_result = {}

        try:
            # L'APM ha bisogno delle predizioni ML sugli asset.
            # Se l'AlphaStrategist le ha gia' calcolate e le espone,
            # le riusiamo: ricalcolarle significa rileggere 300 documenti.
            apm_ml_map = alpha_result.get("ml_map") if isinstance(alpha_result, dict) else None

            if apm_ml_map:
                print(f"  APM: ml_map riusata da Alpha ({len(apm_ml_map)} ticker)")
            else:
                apm_ml_map = {}
                apm_assets = None
                try:
                    # Proiezione stretta: il modello ML usa indicatori tecnici,
                    # non i blocchi strutturali di Max Strategy.
                    apm_assets = await db.assets.find(
                        {}, APM_ASSET_PROJECTION
                    ).to_list(400)

                    apm_ml_map = await self.alpha._load_ml_predictions(
                        db, apm_assets, market_context
                    )
                except Exception as e:
                    print(f"  APM ml_map load error: {e}")
                finally:
                    # Gli asset servivano solo a produrre la ml_map
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
            elif apm_status == "disabled":
                print("  APM disabled in settings")

            if actions_taken > 0:
                positions = await get_positions() or []
                print(f"  Reloaded positions: {len(positions)} still open")

            del apm_ml_map

        except Exception as e:
            report["errors"].append(f"APM: {str(e)}")
            report["steps"]["adaptive_position_manager"] = {"status": "error", "error": str(e)}
            print(f"APM ERROR: {e}")

        report["timing"]["adaptive_position_manager"] = round(time.time() - t_apm, 2)

        # ============================================
        # STEP 4: Executor -> SharedBrain
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
                "deferred_buys": exec_result.get("deferred_buys", []),
                "deferred_sells": exec_result.get("deferred_sells", []),
                "requires_revalidation": exec_result.get("requires_revalidation", False),
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
                        "trailing_adjustments": exec_result.get("trailing_adjustments", []),
                        "synced_trades": exec_result.get("synced_trades", 0),
                        "market_status": exec_result.get("market_status", {}),
                        "llm_reasoning": exec_result.get("llm_reasoning"),
                    },
                )

                # Dopo l'esecuzione pulisce gli approved dal brain,
                # cosi' al prossimo run non vengono riusati per errore
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

        # ============================================
        # CRASH DEPLOY LIVE (Progetto Alpha)
        # Tripla sicurezza: flag master + dry_run + gate regime.
        # Gira ad ogni ciclo ma il gate interno lo blocca in bull
        # o se il flag e' disattivato.
        # ============================================
        try:
            from app.services.crash_deploy_live import check_and_deploy

            deploy_res = await check_and_deploy()
            report["steps"]["crash_deploy"] = deploy_res

            if deploy_res.get("actions"):
                print(f"  Crash Deploy: {deploy_res['status']} — {deploy_res['actions']}")
            else:
                print(f"  Crash Deploy: {deploy_res.get('status')} "
                      f"(regime {deploy_res.get('regime', '-')})")

        except Exception as e:
            report["errors"].append(f"CrashDeploy: {str(e)}")
            report["steps"]["crash_deploy"] = {"status": "error", "error": str(e)}
            print(f"Crash Deploy ERROR: {e}")

        # ============================================
        # STEP 5: salva lo stato della pipeline
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
                "regime_detail": market_context.get("regime_detail", "UNKNOWN"),
                "confidence": market_context.get("regime_confidence", 0),
                "exposure_multiplier": market_context.get("exposure_multiplier", 0.5),
                "volatility": market_context.get("volatility_regime", "UNKNOWN"),
                "breadth_pct": market_context.get("breadth_pct", 0),
                "rotation": market_context.get("rotation_signal", "unknown"),
                "leadership": market_context.get("leadership", {}).get("state", "UNKNOWN"),
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
        print(f"PIPELINE COMPLETE in {total_time}s")
        print(f"   Regime: {market_context.get('market_regime')}/"
              f"{market_context.get('regime_detail')} "
              f"(conf={market_context.get('regime_confidence')})")
        print(f"   Buys: {len(exec_result.get('executed_buys', []))} | "
              f"Sells: {len(exec_result.get('executed_sells', []))} | "
              f"Rejected: {len(rejected_trades)}")

        if report["errors"]:
            print(f"   Errors: {report['errors']}")

        print(f"{'=' * 60}")

        # Libera le strutture grandi prima di restituire il report:
        # su Render la memoria non torna sempre al sistema operativo,
        # ma evitiamo che restino agganciate fino al ciclo successivo.
        alpha_result = None
        gc.collect()

        return report

    async def learn_all(self) -> dict:
        """Esegue il learning loop per tutti gli agenti."""
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

        pipeline_state = await db.auto_trader.find_one({"_id": "alpaca_state"})
        if pipeline_state:
            pipeline_state["_id"] = str(pipeline_state["_id"])

        brain_state = await brain.get_full_state()

        return {
            "agents": status,
            "pipeline_state": pipeline_state,
            "shared_brain": brain_state,
        }
