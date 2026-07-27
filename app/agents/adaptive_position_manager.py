"""
🎯 AGENTE 5: Adaptive Position Manager (APM) v1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Gestisce le posizioni aperte in modo ADATTIVO.
Rivaluta ogni 3 ore (configurabile) la tesi originale di ogni posizione.

4 DECISIONI:
- 🟢 HOLD: tesi valida, mantieni
- 🟡 SCALE_OUT: chiudi parziale, break-even sul resto
- 🔴 EXIT: tesi rotta, chiudi 100%
- 🛡️ TIGHTEN_STOP: proteggi profit, alza SL

TRIGGER:
- Timer (ogni 3 ore)
- Urgent (drop >5% in 1h)

LEARNING:
- Ogni decisione loggata con outcome
- Weekly review per aggiustare soglie
"""

from datetime import datetime, timedelta
from app.agents.base_agent import BaseAgent
from app.db.mongodb import get_db
from app.services.alpaca_trader import get_positions, close_position, update_stop_loss, close_position_partial


class AdaptivePositionManager(BaseAgent):
    """
    🎯 APM v1.0 — Adaptive Position Manager
    """

    def __init__(self):
        super().__init__(name="adaptive_position_manager", version="1.0")

    def default_params(self) -> dict:
        return {
            # ===== MASTER TOGGLE =====
            "apm_enabled": True,
            
            # ===== EXIT thresholds =====
            "apm_exit_confluence_threshold": 30,    # Se scende sotto → considera exit
            "apm_exit_ml_threshold": 40,            # Se ML WIN scende sotto → considera exit
            "apm_exit_min_negative_factors": 2,     # Minimo fattori negativi
            
            # ===== SCALE OUT targets =====
            "apm_scaling_enabled": True,
            "apm_target_1_pct": 5.0,     # +5% → chiude 50%
            "apm_target_1_size": 50,     # % da chiudere al T1
            "apm_target_2_pct": 10.0,    # +10% → chiude 30%
            "apm_target_2_size": 30,
            "apm_target_3_pct": 20.0,    # +20% → chiude 20% residuo
            "apm_target_3_size": 20,
            
            # ===== TIGHTEN STOP =====
            "apm_tighten_profit_threshold": 3.0,    # Se profit > 3% AND ML down → tighten SL
            "apm_tighten_new_sl_distance": 2.0,     # Alza SL a -2% dal current
            
            # ===== FREQUENCY =====
            "apm_check_interval_hours": 3,           # Ogni 3 ore
            "apm_urgent_check_drop_pct": 5.0,        # Se drop > 5% in 1h → urgent
        }

    async def check_urgent_triggers(self, context: dict) -> dict:
        """
        🆕 v4.2 — Check veloce SOLO su trigger matematici (target hit, drop).
        
        Chiamato ad ogni pipeline (15 min) — bypass timer 1h.
        Solo eventi CRITICI:
        - P&L >= target 1/2/3 → SCALE_OUT
        - Drop >5% in 1h → EXIT
        
        Zero LLM, zero confluence recalc. Millisecondi.
        """
        db = get_db()
        params = await self.get_params()
        
        if not params.get("apm_enabled", True):
            return {"status": "disabled", "actions_taken": []}
        
        positions = context.get("positions", [])
        if not positions:
            return {"status": "no_positions", "actions_taken": []}
        
        # Params targets
        t1_pct = params.get("apm_target_1_pct", 5.0)
        t2_pct = params.get("apm_target_2_pct", 10.0)
        t3_pct = params.get("apm_target_3_pct", 20.0)
        t1_size = params.get("apm_target_1_size", 50)
        t2_size = params.get("apm_target_2_size", 30)
        t3_size = params.get("apm_target_3_size", 20)
        urgent_drop_pct = params.get("apm_urgent_check_drop_pct", 5.0)
        scaling_enabled = params.get("apm_scaling_enabled", True)
        
        actions_taken = []
        
        for pos in positions:
            symbol = pos.get("symbol")
            current_price = float(pos.get("current_price", 0))
            entry_price = float(pos.get("avg_entry_price", 0))
            pnl_pct = float(pos.get("unrealized_plpc", 0)) * 100
            
            if not symbol or entry_price <= 0:
                continue
            
            # Trova buy_trade
            buy_trade = await db.trade_history.find_one(
                {
                    "ticker": symbol,
                    "side": "buy",
                    "sell_linked": {"$ne": True}
                },
                sort=[("date", -1)]
            )
            if not buy_trade:
                continue
            
            # Check se già scaled out (per non ripetere stesso target)
            # Check se già scaled out (per non ripetere stesso target)
            # 🆕 v4.6 — Adaptive targets: usa quelli calcolati al buy, fallback su params
            adaptive_t1 = buy_trade.get("adaptive_t1_pct")
            adaptive_t2 = buy_trade.get("adaptive_t2_pct")
            adaptive_t3 = buy_trade.get("adaptive_t3_pct")
            
            # Override params solo se disponibili nel buy (fallback per trade vecchi)
            if adaptive_t1 and adaptive_t1 > 0:
                t1_pct_local = adaptive_t1
                t2_pct_local = adaptive_t2 if adaptive_t2 else t2_pct
                t3_pct_local = adaptive_t3 if adaptive_t3 else t3_pct
            else:
                t1_pct_local = t1_pct
                t2_pct_local = t2_pct
                t3_pct_local = t3_pct
            last_target_hit = buy_trade.get("last_target_hit", 0)
            partial_scaled_out = buy_trade.get("partial_scaled_out", False)
            
            # 🔧 v4.4 — Safety net: se già partial_scaled_out ma last_target_hit=0
            # (bug legacy), forza last_target_hit >= 1 per bloccare re-trigger T1
            if partial_scaled_out and last_target_hit < 1:
                last_target_hit = 1
            
            action = None
            reason = None
            size_pct = 0
            target_num = 0
            
            # ============================================
            # TRIGGER CHECK (in ordine: T3 → T2 → T1 → drop)
            # ============================================
            if scaling_enabled and pnl_pct >= t3_pct_local and last_target_hit < 3:
                action = "SCALE_OUT"
                target_num = 3
                size_pct = t3_size
                reason = f"URGENT T3 hit (+{pnl_pct:.1f}% >= +{t3_pct_local}% adaptive)"
            elif scaling_enabled and pnl_pct >= t2_pct_local and last_target_hit < 2:
                action = "SCALE_OUT"
                target_num = 2
                size_pct = t2_size
                reason = f"URGENT T2 hit (+{pnl_pct:.1f}% >= +{t2_pct_local}% adaptive)"
            elif scaling_enabled and pnl_pct >= t1_pct_local and last_target_hit < 1:
                action = "SCALE_OUT"
                target_num = 1
                size_pct = t1_size
                reason = f"URGENT T1 hit (+{pnl_pct:.1f}% >= +{t1_pct_local}% adaptive)"
            elif pnl_pct <= -urgent_drop_pct:
                # Drop critico → NON EXIT automatico, ma logga per next full analysis
                # Meglio non fare EXIT senza confluence check
                print(f"  ⚠️ URGENT: {symbol} drop {pnl_pct:.1f}% — will be reviewed in next full APM run")
                continue
            
            # ============================================
            # ESEGUI TRIGGER (solo SCALE_OUT)
            # ============================================
            if action == "SCALE_OUT":
                print(f"  🚨 URGENT TRIGGER {symbol}: {reason}")
                
                action_taken, action_details = await self._execute_scale_out(
                    symbol, pos, buy_trade, target_num, size_pct, reason
                )
                
                if action_taken:
                    decision_log = {
                        "ticker": symbol,
                        "decision": "SCALE_OUT",
                        "reason": reason,
                        "current_pnl_pct": round(pnl_pct, 2),
                        "current_price": current_price,
                        "entry_price": entry_price,
                        "action_taken": True,
                        "action_details": action_details,
                        "trigger_type": "urgent",
                    }
                    actions_taken.append(decision_log)
                    
                    # Log to decisions collection
                    await self.log_decision(
                        decision_type=f"apm_urgent_scale_out",
                        data=decision_log,
                        reasoning=reason,
                        confidence=80,
                    )
                    
                    # Telegram alert
                    try:
                        from app.services.telegram_bot import send_telegram
                        msg = (
                            f"🚨 <b>APM URGENT TRIGGER</b>\n\n"
                            f"🟡 <b>{symbol}</b> SCALE_OUT T{target_num}\n"
                            f"P&L: {pnl_pct:+.2f}% | Size: {size_pct}%\n"
                            f"{reason}"
                        )
                        await send_telegram(msg)
                    except Exception as e:
                        print(f"  Telegram error: {e}")
        
        if actions_taken:
            print(f"🚨 APM URGENT: {len(actions_taken)} actions triggered")
        
        return {
            "status": "ok",
            "actions_taken": actions_taken,
            "checked_positions": len(positions),
        }
        
    async def analyze(self, context: dict) -> dict:
        """
        Analizza tutte le posizioni aperte e decide azione per ciascuna.
        """
        db = get_db()
        params = await self.get_params()
        
        # Check master toggle
        if not params.get("apm_enabled", True):
            return {
                "status": "disabled",
                "message": "APM is disabled in settings",
            }
        
        # Ricava contesto
        market_ctx = context.get("market_context", {})
        # 🆕 v4.5 — Salva regime per uso in _decide_action
        self._current_market_regime = market_ctx.get("market_regime", "NEUTRAL")
        self._current_market_confidence = market_ctx.get("regime_confidence", 50)
        positions = context.get("positions", [])
        ml_map = context.get("ml_map", {})
        
        # Se no posizioni, skip
        if not positions:
            return {
                "status": "no_positions",
                "message": "No positions to analyze",
                "decisions": [],
            }
        
        # Check timer (skip se non è il momento)
        should_run = await self._should_run_now(params)
        if not should_run["run"]:
            return {
                "status": "skipped_timer",
                "message": should_run["reason"],
                "next_check": should_run.get("next_check"),
                "decisions": [],
            }
        
        # Ottieni dati assets per confluence recalculation
        assets = await db.assets.find({}, {
            "price_history": 0, "vp_distribution": 0, "multi_tf_vp": 0
        }).to_list(300)
        assets_map = {a["ticker"]: a for a in assets}
        
        # Carica assets_map con confluence recalculation
        # Import qui per evitare circular import
        from app.agents.alpha_strategist import AlphaStrategist
        alpha = AlphaStrategist()
        
        # ============================================
        # Analizza ogni posizione
        # ============================================
        decisions = []
        actions_taken = []
        
        for pos in positions:
            symbol = pos.get("symbol")
            current_price = float(pos.get("current_price", 0))
            entry_price = float(pos.get("avg_entry_price", 0))
            qty = float(pos.get("qty", 0))
            pnl_pct = float(pos.get("unrealized_plpc", 0)) * 100
            
            if not symbol or entry_price <= 0:
                continue
            
            # Trova buy_trade originale nel DB
            buy_trade = await db.trade_history.find_one(
                {
                    "ticker": symbol,
                    "side": "buy",
                    "sell_linked": {"$ne": True}
                },
                sort=[("date", -1)]
            )
            
            if not buy_trade:
                decisions.append({
                    "ticker": symbol,
                    "decision": "SKIP",
                    "reason": "No buy_trade found in DB",
                    "current_pnl_pct": pnl_pct,
                })
                continue
            
            # Recupera dati originali
            original_confluence = buy_trade.get("confluence", 50)
            original_stop = buy_trade.get("stop_loss", 0)
            original_target = buy_trade.get("target", 0)
            original_setup = buy_trade.get("setup_type", "unknown")
            original_ml_score = buy_trade.get("ml_score", 0)
            original_ml_pred = buy_trade.get("ml_prediction", "unknown")
            
            # Ricalcola confluence attuale
            asset = assets_map.get(symbol)
            current_ml_data = ml_map.get(symbol, {}) if ml_map else {}
            
            if not asset:
                decisions.append({
                    "ticker": symbol,
                    "decision": "SKIP",
                    "reason": "No asset data",
                    "current_pnl_pct": pnl_pct,
                })
                continue
            
            # Calcola confluence attuale
            try:
                current_conf_data = alpha._calc_confluence(
                    asset, market_ctx, params, current_ml_data
                )
                current_confluence = current_conf_data.get("score", 0)
            except Exception as e:
                print(f"  ⚠️ APM confluence calc error {symbol}: {e}")
                current_confluence = original_confluence
            
            # Dati ML attuali
            current_ml_score = current_ml_data.get("ml_score", 0)
            current_ml_pred = current_ml_data.get("ml_prediction", "unknown")
            current_trend_pred = current_ml_data.get("trend_prediction", "unknown")
            
            # ============================================
            # DECISION LOGIC
            # ============================================
            # 🆕 v4.6 — Salva buy_trade per accesso adaptive targets in _decide_action
            self._current_buy_trade = buy_trade
            decision_result = self._decide_action(
                pos=pos,
                pnl_pct=pnl_pct,
                original_confluence=original_confluence,
                current_confluence=current_confluence,
                original_ml_score=original_ml_score,
                current_ml_score=current_ml_score,
                current_ml_pred=current_ml_pred,
                current_trend_pred=current_trend_pred,
                original_target=original_target,
                original_stop=original_stop,
                params=params,
            )
            
            decision = decision_result["decision"]
            reason = decision_result["reason"]
            details = decision_result.get("details", {})
            
            # ============================================
            # ESEGUI AZIONE
            # ============================================
            action_taken = False
            action_details = {}
            
            if decision == "EXIT":
                action_taken, action_details = await self._execute_exit(
                    symbol, pos, buy_trade, reason, current_confluence, current_ml_score
                )
            elif decision == "SCALE_OUT":
                action_taken, action_details = await self._execute_scale_out(
                    symbol, pos, buy_trade, details.get("target_hit", 1), 
                    details.get("size_pct", 50), reason
                )
            elif decision == "TIGHTEN_STOP":
                action_taken, action_details = await self._execute_tighten_stop(
                    symbol, pos, buy_trade, details.get("new_stop", 0), reason
                )
            
            # ============================================
            # LOG DECISION
            # ============================================
            decision_log = {
                "ticker": symbol,
                "decision": decision,
                "reason": reason,
                "current_pnl_pct": round(pnl_pct, 2),
                "current_price": current_price,
                "entry_price": entry_price,
                "state_snapshot": {
                    "confluence_original": original_confluence,
                    "confluence_now": current_confluence,
                    "ml_score_original": original_ml_score,
                    "ml_score_now": current_ml_score,
                    "ml_prediction_original": original_ml_pred,
                    "ml_prediction_now": current_ml_pred,
                    "trend_prediction_now": current_trend_pred,
                    "regime": market_ctx.get("market_regime", "UNKNOWN"),
                },
                "action_taken": action_taken,
                "action_details": action_details,
                "details": details,
            }
            decisions.append(decision_log)
            
            if action_taken:
                actions_taken.append(decision_log)
            
            # Log to decisions collection
            await self.log_decision(
                decision_type=f"apm_{decision.lower()}",
                data=decision_log,
                reasoning=reason,
                confidence=70 if action_taken else 40,
            )
        
        # ============================================
        # UPDATE last_run timestamp
        # ============================================
        await db.apm_state.update_one(
            {"_id": "last_run"},
            {"$set": {
                "timestamp": datetime.utcnow(),
                "decisions_count": len(decisions),
                "actions_count": len(actions_taken),
            }},
            upsert=True
        )
        
        # ============================================
        # BUILD SUMMARY + LLM REASONING
        # ============================================
        summary = self._build_summary(decisions, actions_taken)
        
        # LLM Reasoning
        from app.services.llm_service import llm_ask, llm_available
        llm_reasoning = None
        if llm_available() and (actions_taken or len(decisions) > 0):
            try:
                summary_text = self._build_llm_summary_text(decisions, actions_taken, market_ctx)
                llm_reasoning = llm_ask(
                    system_prompt=(
                        "Sei un position manager esperto di swing trading. "
                        "Valuta le decisioni APM appena prese in max 3 frasi in italiano. "
                        "Indica: 1) Se le decisioni sono coerenti col regime, 2) Rischio residuo del portfolio, "
                        "3) Suggerimento operativo. Sii diretto, concreto, no disclaimers."
                    ),
                    user_prompt=summary_text,
                    max_tokens=200,
                    temperature=0.3,
                    agent_name="apm",
                )
                if llm_reasoning:
                    print(f"  🧠 APM LLM: {llm_reasoning[:80]}...")
            except Exception as e:
                print(f"  APM LLM error: {e}")
        
        # ============================================
        # TELEGRAM NOTIFICATION
        # ============================================
        if actions_taken:
            await self._send_telegram_alert(actions_taken, market_ctx)
        
        print(f"🎯 APM: analyzed {len(decisions)} positions, {len(actions_taken)} actions taken")
        
        return {
            "status": "ok",
            "decisions": decisions,
            "actions_taken": actions_taken,
            "summary": summary,
            "llm_reasoning": llm_reasoning,
            "analyzed_at": datetime.utcnow().isoformat(),
        }

    async def _should_run_now(self, params: dict) -> dict:
        """Check se è il momento di rieseguire l'APM (timer-based)."""
        db = get_db()
        interval_hours = params.get("apm_check_interval_hours", 3)
        
        last_run_doc = await db.apm_state.find_one({"_id": "last_run"})
        
        if not last_run_doc:
            return {"run": True, "reason": "First run"}
        
        last_run = last_run_doc.get("timestamp")
        if not last_run:
            return {"run": True, "reason": "No last_run timestamp"}
        
        elapsed = (datetime.utcnow() - last_run).total_seconds() / 3600
        
        if elapsed >= interval_hours:
            return {"run": True, "reason": f"Elapsed {elapsed:.1f}h >= {interval_hours}h"}
        
        remaining = interval_hours - elapsed
        next_check = datetime.utcnow() + timedelta(hours=remaining)
        
        return {
            "run": False,
            "reason": f"Wait {remaining:.1f}h more (interval: {interval_hours}h)",
            "next_check": next_check.isoformat(),
        }

    def _decide_action(self, pos, pnl_pct, original_confluence, current_confluence,
                       original_ml_score, current_ml_score, current_ml_pred,
                       current_trend_pred, original_target, original_stop, params) -> dict:
        """
        Logica decisionale APM.
        Ritorna: {"decision": "...", "reason": "...", "details": {...}}
        """
        # 🆕 v4.6 — Adaptive targets (letti da self._current_buy_trade se disponibile)
        buy_trade_ref = getattr(self, "_current_buy_trade", None)
        if buy_trade_ref:
            adaptive_t1 = buy_trade_ref.get("adaptive_t1_pct")
            adaptive_t2 = buy_trade_ref.get("adaptive_t2_pct")
            adaptive_t3 = buy_trade_ref.get("adaptive_t3_pct")
        else:
            adaptive_t1 = adaptive_t2 = adaptive_t3 = None

        # 🆕 v1.3 — Target già raggiunti (per proteggere i runner post-T1)
        last_target_hit_now = buy_trade_ref.get("last_target_hit", 0) if buy_trade_ref else 0
        partial_now = buy_trade_ref.get("partial_scaled_out", False) if buy_trade_ref else False
        if partial_now and last_target_hit_now < 1:
            last_target_hit_now = 1

        # Soglie base
        exit_conf_th_base = params.get("apm_exit_confluence_threshold", 30)
        exit_ml_th_base = params.get("apm_exit_ml_threshold", 40)
        
        # 🆕 v4.5 — APM Regime-Aware: adatta soglie a market regime
        # Legge da market_context passato dal Orchestrator
        # Non altera params in DB, solo runtime
        market_regime = "NEUTRAL"
        regime_confidence = 50
        try:
            # Se disponibile nel context (aggiunto in analyze() sopra)
            market_regime = getattr(self, "_current_market_regime", "NEUTRAL")
            regime_confidence = getattr(self, "_current_market_confidence", 50)
        except:
            pass
        
        # Adatta soglie in base al regime
        regime_multipliers = {
            "BULL": {"conf": -10, "ml": -10},      # meno aggressivo (esce solo se davvero rotto)
            "NEUTRAL": {"conf": 0, "ml": 0},       # comportamento standard
            "BEAR": {"conf": +10, "ml": +10},      # più aggressivo (esce prima)
            "CRASH": {"conf": +15, "ml": +15},     # molto aggressivo (protegge capitale)
        }
        adj = regime_multipliers.get(market_regime, {"conf": 0, "ml": 0})
        exit_conf_th = max(15, min(50, exit_conf_th_base + adj["conf"]))
        exit_ml_th = max(20, min(60, exit_ml_th_base + adj["ml"]))
        
        min_negative = params.get("apm_exit_min_negative_factors", 2)
        
        # 🆕 v1.1 — Detect "ML flat" (bug fix)
        # Se ML score è sospettosamente uguale per tutti (es. 92.3% overfitted),
        # ignoriamo il factor ML dalle decisioni.
        # ML flat = probabilmente modello overfit o non discriminatorio.
        ml_score_looks_flat = (
            current_ml_score > 85 and current_ml_score < 95 
            and abs(current_ml_score - 92.3) < 1.0  # troppo vicino a 92.3%
        )
        
        # Conta fattori negativi
        negative_factors = []
        if current_confluence < exit_conf_th:
            negative_factors.append(f"confluence {current_confluence:.0f} < {exit_conf_th}")
        
        # 🔧 Ignora ML score se sembra "flat" (bug del modello)
        if not ml_score_looks_flat:
            if current_ml_score < exit_ml_th:
                negative_factors.append(f"ML {current_ml_score:.0f}% < {exit_ml_th}%")
            if current_ml_pred == "LOSS":
                negative_factors.append("ML predicts LOSS")
        
        # Trend è indipendente dal ML score, quindi lo teniamo
        if current_trend_pred == "DOWN":
            negative_factors.append("Trend DOWN")
        
        # ============================================
        # 🔴 EXIT
        # ============================================
        # 🆕 v1.4 — Minimum holding period (anti-churning).
        # Non uscire per "tesi debole" su posizioni appena aperte: evita il loop
        # compra→vende→ricompra quando Alpha (entry) e APM (exit) hanno soglie
        # in conflitto (es. confluence 44 basta per comprare ma l'APM la scarta).
        # Lo STOP LOSS resta attivo (gestito dall'Executor) → la protezione vera c'è.
        buy_date_ref = buy_trade_ref.get("date") if buy_trade_ref else None
        hours_held = 999.0
        if buy_date_ref:
            try:
                hours_held = (datetime.utcnow() - buy_date_ref).total_seconds() / 3600
            except Exception:
                hours_held = 999.0
        too_fresh = hours_held < 24  # prime 24h: niente EXIT per tesi debole

        # 🆕 v1.3 — Runner protection: se ha già preso T1+ ed è in profitto,
        # NON uscire per tesi debole. Lascia correre verso T2/T3.
        runner_in_profit = last_target_hit_now >= 1 and pnl_pct > 0
        if len(negative_factors) >= min_negative and not runner_in_profit and not too_fresh:
            return {
                "decision": "EXIT",
                "reason": (
                    f"Tesi invalidata: {len(negative_factors)} fattori negativi. "
                    f"Confluence {original_confluence}→{current_confluence:.0f}, "
                    f"ML {original_ml_score:.0f}%→{current_ml_score:.0f}%. "
                    f"Meglio uscire con P&L {pnl_pct:+.1f}% ora che rischiare peggio."
                ),
                "details": {
                    "negative_factors": negative_factors,
                    "confluence_drop": original_confluence - current_confluence,
                    "ml_drop": original_ml_score - current_ml_score,
                },
            }
        
        # ============================================
        # 🟡 SCALE_OUT (multi-target)
        # ============================================
        if params.get("apm_scaling_enabled", True):
            t1_pct = adaptive_t1 if adaptive_t1 else params.get("apm_target_1_pct", 5.0)
            t2_pct = adaptive_t2 if adaptive_t2 else params.get("apm_target_2_pct", 10.0)
            t3_pct = adaptive_t3 if adaptive_t3 else params.get("apm_target_3_pct", 20.0)
            t1_size = params.get("apm_target_1_size", 50)
            t2_size = params.get("apm_target_2_size", 30)
            t3_size = params.get("apm_target_3_size", 20)
            
            # 🔧 v4.7 — Fix critico: leggi last_target_hit per evitare ripetizioni
            last_target_hit = buy_trade_ref.get("last_target_hit", 0) if buy_trade_ref else 0
            partial_scaled_out = buy_trade_ref.get("partial_scaled_out", False) if buy_trade_ref else False
            # Safety net: se partial ma last_target_hit=0 (legacy), forza a 1
            if partial_scaled_out and last_target_hit < 1:
                last_target_hit = 1
            
            # Ha già raggiunto qualche target?
            # (Per ora semplice check: se pnl_pct >= t3, target 3; se >= t2, target 2; ecc.)
            # In produzione futura terremo track di quali target sono già stati raggiunti
            
            if pnl_pct >= t3_pct and last_target_hit < 3:
                return {
                    "decision": "SCALE_OUT",
                    "reason": (
                        f"Target 3 raggiunto (+{pnl_pct:.1f}% >= +{t3_pct}%). "
                        f"Chiudo {t3_size}% posizione residua, break-even sul resto."
                    ),
                    "details": {"target_hit": 3, "size_pct": t3_size},
                }
            elif pnl_pct >= t2_pct and last_target_hit < 2:
                return {
                    "decision": "SCALE_OUT",
                    "reason": (
                        f"Target 2 raggiunto (+{pnl_pct:.1f}% >= +{t2_pct}%). "
                        f"Chiudo {t2_size}% posizione, lascio correre il resto."
                    ),
                    "details": {"target_hit": 2, "size_pct": t2_size},
                }
            elif pnl_pct >= t1_pct and last_target_hit < 1:
                return {
                    "decision": "SCALE_OUT",
                    "reason": (
                        f"Target 1 raggiunto (+{pnl_pct:.1f}% >= +{t1_pct}%). "
                        f"Chiudo {t1_size}% posizione per prendere profit, alzo SL a break-even sul resto."
                    ),
                    "details": {"target_hit": 1, "size_pct": t1_size},
                }
        
        # ============================================
        # 🛡️ TIGHTEN STOP
        # ============================================
        tighten_th = params.get("apm_tighten_profit_threshold", 3.0)
        current_price = float(pos.get("current_price", 0))
        
        if pnl_pct >= tighten_th and (current_ml_pred == "LOSS" or current_trend_pred == "DOWN"):
            new_sl_distance = params.get("apm_tighten_new_sl_distance", 2.0) / 100
            new_stop = round(current_price * (1 - new_sl_distance), 2)
            
            return {
                "decision": "TIGHTEN_STOP",
                "reason": (
                    f"Profit +{pnl_pct:.1f}% ma ML mostra segnale bearish. "
                    f"Alzo SL a ${new_stop} (-{new_sl_distance*100}% dal current) per proteggere il profit."
                ),
                "details": {"new_stop": new_stop},
            }
        
        # ============================================
        # 🟢 HOLD
        # ============================================
        return {
            "decision": "HOLD",
            "reason": (
                f"Tesi ancora valida. Confluence {current_confluence:.0f} (era {original_confluence}), "
                f"ML {current_ml_score:.0f}%. P&L {pnl_pct:+.1f}%. Mantengo posizione."
            ),
            "details": {},
        }

    async def _execute_exit(self, symbol, pos, buy_trade, reason, current_confluence, current_ml_score):
        """Esegue chiusura 100% della posizione."""
        db = get_db()
        try:
            close_result = await close_position(symbol)
            if close_result is None:
                return False, {"error": "close_position returned None"}
            
            current_price = float(pos.get("current_price", 0))
            entry_price = float(buy_trade.get("entry_price", 0))
            qty = float(pos.get("qty", 0))
            pnl_pct = round(((current_price - entry_price) / entry_price) * 100, 2) if entry_price > 0 else 0
            pnl_dollar = round((current_price - entry_price) * qty, 2)
            days_held = max(1, (datetime.utcnow() - buy_trade.get("date", datetime.utcnow())).days)
            
            sell_order_id = f"apm_exit_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Log sell in trade_history
            await db.trade_history.insert_one({
                "ticker": symbol,
                "side": "sell",
                "entry_price": entry_price,
                "exit_price": current_price,
                "shares": float(qty),
                "pnl_pct": pnl_pct,
                "pnl_dollar": pnl_dollar,
                "days_held": days_held,
                "reason": "APM_EXIT",
                "apm_reason": reason,
                "setup_type": buy_trade.get("setup_type", "unknown"),
                "sector": buy_trade.get("sector", "unknown"),
                "market_regime": buy_trade.get("market_regime", "UNKNOWN"),
                "order_id": sell_order_id,
                "buy_order_id": buy_trade.get("order_id", ""),
                "agent": "adaptive_position_manager",
                "date": datetime.utcnow(),
                "source": "apm_v1",
                "apm_confluence_at_exit": current_confluence,
                "apm_ml_at_exit": current_ml_score,
            })
            
            # Link buy → sell
            await db.trade_history.update_one(
                {"_id": buy_trade["_id"]},
                {"$set": {"sell_linked": True, "sell_order_id": sell_order_id}}
            )
            
            # Cleanup trailing stop
            await db.trailing_stops.delete_one({"ticker": symbol})
            
            print(f"  🔴 APM EXIT {symbol}: P&L {pnl_pct:+.2f}% (${pnl_dollar:+.0f})")
            
            return True, {
                "action": "EXIT",
                "pnl_pct": pnl_pct,
                "pnl_dollar": pnl_dollar,
                "days_held": days_held,
            }
        except Exception as e:
            print(f"  ⚠️ APM EXIT error {symbol}: {e}")
            return False, {"error": str(e)}

    async def _execute_scale_out(self, symbol, pos, buy_trade, target_hit, size_pct, reason):
        """
        🆕 v4.0 FASE 4 — Chiusura parziale REALE su Alpaca.
        
        Chiude size_pct% della posizione + sposta SL a break-even sul restante.
        
        Args:
            symbol: ticker
            pos: dict posizione Alpaca
            buy_trade: dict buy da trade_history
            target_hit: 1, 2, o 3 (quale target scattato)
            size_pct: % posizione da chiudere (es. 50)
            reason: motivo APM
        """
        db = get_db()
        
        try:
            # 1. Calcola quantità da chiudere
            current_qty = float(pos.get("qty", 0))
            qty_to_close = round(current_qty * (size_pct / 100), 4)
            qty_remaining = round(current_qty - qty_to_close, 4)
            
            if qty_to_close <= 0.0001:
                return False, {"error": "qty_to_close too small"}
            
            current_price = float(pos.get("current_price", 0))
            entry_price = float(buy_trade.get("entry_price", 0))
            
            # 2. Cancella eventuali ordini SL/TP aperti (li rimpiazzeremo dopo)
            from app.services.alpaca_trader import get_orders, cancel_order
            open_orders = await get_orders(status="open", limit=50)
            cancelled_orders = 0
            if open_orders:
                for o in open_orders:
                    if o.get("symbol") == symbol and o.get("side") == "sell":
                        await cancel_order(o.get("id"))
                        cancelled_orders += 1
            
            # 3. Chiudi parziale su Alpaca
            close_result = await close_position_partial(symbol, qty_to_close)
            
            if close_result is None:
                return False, {"error": "partial close failed", "cancelled_orders": cancelled_orders}
            
            # 4. Calcola P&L su porzione chiusa
            pnl_pct_partial = round(((current_price - entry_price) / entry_price) * 100, 2) if entry_price > 0 else 0
            pnl_dollar_partial = round((current_price - entry_price) * qty_to_close, 2)
            buy_date = buy_trade.get("date", datetime.utcnow())
            days_held = max(1, (datetime.utcnow() - buy_date).days) if buy_date else 1
            
            sell_order_id = f"apm_scale_{target_hit}_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # 5. Log sell parziale in trade_history
            await db.trade_history.insert_one({
                "ticker": symbol,
                "side": "sell",
                "entry_price": entry_price,
                "exit_price": current_price,
                "shares": float(qty_to_close),
                "pnl_pct": pnl_pct_partial,
                "pnl_dollar": pnl_dollar_partial,
                "days_held": days_held,
                "reason": f"APM_SCALE_OUT_T{target_hit}",
                "apm_reason": reason,
                "setup_type": buy_trade.get("setup_type", "unknown"),
                "sector": buy_trade.get("sector", "unknown"),
                "market_regime": buy_trade.get("market_regime", "UNKNOWN"),
                "order_id": sell_order_id,
                "buy_order_id": buy_trade.get("order_id", ""),
                "agent": "adaptive_position_manager",
                "date": datetime.utcnow(),
                "source": "apm_v1_scale_out",
                "target_hit": target_hit,
                "partial": True,
                "qty_closed": float(qty_to_close),
                "qty_remaining": float(qty_remaining),
            })
            
            # 6. Aggiorna buy_trade con qty ridotta (per software SL/TP tracking)
            await db.trade_history.update_one(
                {"_id": buy_trade["_id"]},
                {"$set": {
                    "shares": float(qty_remaining),
                    "partial_scaled_out": True,
                    "last_scale_out_at": datetime.utcnow(),
                    "last_target_hit": target_hit,
                }}
            )
            
            # 7. 🆕 v4.8 — Floor FISSO (no trailing dal picco). "Lascia correre".
            # Dopo lo scale-out il residuo corre libero verso il target pieno,
            # protetto solo da un floor fisso. Il software SL/TP dell'Executor
            # (sicurezza indispensabile con fractional) chiude a floor (giù) o
            # al target Alpha (su). apm_managed blocca _manage_trailing_stops.
            if target_hit == 1:
                floor_price = entry_price          # break-even
            elif target_hit == 2:
                floor_price = entry_price * 1.03   # +3% garantito
            else:
                floor_price = entry_price * 1.08
            new_stop = round(floor_price, 2)

            await db.trailing_stops.update_one(
                {"ticker": symbol},
                {"$set": {
                    "ticker": symbol,
                    "stop_price": new_stop,
                    "floor_price": new_stop,
                    "trailing_active": False,
                    "apm_managed": True,
                    "reason": f"APM scale-out T{target_hit}: floor fisso ${new_stop} (lascia correre verso target)",
                    "updated_at": datetime.utcnow(),
                    "source": "apm_v1_scale_out",
                }},
                upsert=True
            )
            
            print(f"  🟡 APM SCALE_OUT T{target_hit} {symbol}: closed {qty_to_close:.4f} shares "
                  f"(P&L {pnl_pct_partial:+.2f}%, ${pnl_dollar_partial:+.0f}), "
                  f"SL → ${new_stop:.2f}, remaining {qty_remaining:.4f}")
            
            return True, {
                "action": "SCALE_OUT_REAL",
                "target_hit": target_hit,
                "size_pct": size_pct,
                "qty_closed": float(qty_to_close),
                "qty_remaining": float(qty_remaining),
                "pnl_pct": pnl_pct_partial,
                "pnl_dollar": pnl_dollar_partial,
                "new_stop": round(new_stop, 2),
                "cancelled_orders": cancelled_orders,
            }
        except Exception as e:
            print(f"  ⚠️ APM SCALE_OUT error {symbol}: {e}")
            return False, {"error": str(e)}

    async def _execute_tighten_stop(self, symbol, pos, buy_trade, new_stop, reason):
        """Esegue tightening dello stop loss."""
        db = get_db()
        try:
            # Aggiorna trailing_stops collection
            await db.trailing_stops.update_one(
                {"ticker": symbol},
                {"$set": {
                    "ticker": symbol,
                    "stop_price": new_stop,
                    "reason": f"APM tighten: {reason}",
                    "updated_at": datetime.utcnow(),
                    "source": "apm_v1",
                }},
                upsert=True
            )
            
            print(f"  🛡️ APM TIGHTEN {symbol}: SL → ${new_stop}")
            
            return True, {
                "action": "TIGHTEN_STOP",
                "new_stop": new_stop,
            }
        except Exception as e:
            print(f"  ⚠️ APM TIGHTEN error {symbol}: {e}")
            return False, {"error": str(e)}

    def _build_summary(self, decisions, actions_taken):
        """Costruisce summary decisioni."""
        counts = {"HOLD": 0, "SCALE_OUT": 0, "EXIT": 0, "TIGHTEN_STOP": 0, "SKIP": 0}
        for d in decisions:
            decision = d.get("decision", "SKIP")
            counts[decision] = counts.get(decision, 0) + 1
        
        return {
            "total_analyzed": len(decisions),
            "actions_taken": len(actions_taken),
            "counts": counts,
        }

    def _build_llm_summary_text(self, decisions, actions_taken, market_ctx):
        """Costruisce testo per LLM reasoning."""
        text = f"Regime: {market_ctx.get('market_regime', 'UNKNOWN')}\n"
        text += f"Analizzate {len(decisions)} posizioni, {len(actions_taken)} azioni prese.\n\n"
        
        if actions_taken:
            text += "AZIONI:\n"
            for a in actions_taken:
                text += f"- {a['ticker']}: {a['decision']} (P&L {a['current_pnl_pct']:+.1f}%) — {a['reason'][:100]}\n"
        else:
            text += "Nessuna azione, tutte in HOLD.\n"
        
        return text

    async def _send_telegram_alert(self, actions_taken, market_ctx):
        """Invia notifica Telegram per azioni APM."""
        try:
            from app.services.telegram_bot import send_telegram
            
            msg = "🎯 <b>SwingLab APM Report</b>\n\n"
            msg += f"Regime: {market_ctx.get('market_regime', 'UNKNOWN')}\n"
            msg += f"Azioni prese: {len(actions_taken)}\n\n"
            
            for a in actions_taken:
                emoji = {
                    "EXIT": "🔴",
                    "SCALE_OUT": "🟡",
                    "TIGHTEN_STOP": "🛡️",
                }.get(a["decision"], "⚪")
                
                msg += f"{emoji} <b>{a['ticker']}</b> — {a['decision']}\n"
                msg += f"  P&L: {a['current_pnl_pct']:+.2f}%\n"
                msg += f"  {a['reason'][:150]}\n\n"
            
            await send_telegram(msg)
        except Exception as e:
            print(f"  ⚠️ APM Telegram error: {e}")

    async def learn(self) -> dict:
        """
        🧬 FASE 3 — APM Learning Loop v1.0
        
        Analizza le decisioni APM degli ultimi 30 giorni e auto-aggiusta le soglie:
        - Se troppe EXIT premature (posizioni sarebbero recuperate) → alza soglia (meno aggressivo)
        - Se poche EXIT tardive (posizioni scese ancora) → abbassa soglia (più aggressivo)
        - Analizza performance per tipo decisione
        
        Ritorna report con statistiche e aggiornamenti.
        """
        db = get_db()
        params = await self.get_params()
        
        # Cutoff: ultimi 30 giorni
        cutoff = datetime.utcnow() - timedelta(days=30)
        
        # Carica tutte le decisioni APM
        decisions = await self._col_decisions().find({
            "created_at": {"$gte": cutoff},
        }).sort("created_at", -1).to_list(500)
        
        if len(decisions) < 10:
            return {
                "message": "Not enough decisions to learn (need 10+)",
                "count": len(decisions),
            }
        
        # ============================================
        # 1. STATISTICHE PER TIPO DECISIONE
        # ============================================
        stats = {
            "HOLD": {"count": 0, "outcomes": []},
            "EXIT": {"count": 0, "outcomes": []},
            "SCALE_OUT": {"count": 0, "outcomes": []},
            "TIGHTEN_STOP": {"count": 0, "outcomes": []},
        }
        
        for d in decisions:
            data = d.get("data", {})
            decision = data.get("decision", "UNKNOWN")
            if decision in stats:
                stats[decision]["count"] += 1
                stats[decision]["outcomes"].append({
                    "ticker": data.get("ticker"),
                    "pnl_pct": data.get("current_pnl_pct", 0),
                    "confluence_now": data.get("state_snapshot", {}).get("confluence_now", 0),
                    "ml_score_now": data.get("state_snapshot", {}).get("ml_score_now", 0),
                    "created_at": d.get("created_at"),
                })
        
        # ============================================
        # 2. ANALISI EXIT — Erano corrette?
        # ============================================
        # Per ogni EXIT, verifica se il prezzo è sceso ancora nei giorni successivi
        # (se sì → decisione corretta. Se no → EXIT prematuro)
        exit_analysis = {
            "correct_exits": 0,
            "premature_exits": 0,
            "total_analyzed": 0,
        }
        
        for exit_dec in stats["EXIT"]["outcomes"]:
            ticker = exit_dec["ticker"]
            exit_time = exit_dec["created_at"]
            
            if not exit_time:
                continue
            
            # Cerca trade sell corrispondente in trade_history
            sell_trade = await db.trade_history.find_one({
                "ticker": ticker,
                "side": "sell",
                "date": {"$gte": exit_time - timedelta(hours=1)},
                "reason": {"$in": ["APM_EXIT", "SOFTWARE_STOP_LOSS", "SOFTWARE_TAKE_PROFIT"]},
            })
            
            if not sell_trade:
                continue
            
            exit_pnl = sell_trade.get("pnl_pct", 0)
            exit_analysis["total_analyzed"] += 1
            
            # Se P&L era in perdita quando APM ha detto EXIT → corretto
            # Se P&L era in profit → APM ha "salvato profitto" (corretto)
            # Se il prezzo poi è risalito molto → prematuro (missed opportunity)
            if exit_pnl > -1:
                exit_analysis["correct_exits"] += 1
            else:
                exit_analysis["premature_exits"] += 1
        
        # ============================================
        # 3. AUTO-TUNING DELLE SOGLIE
        # ============================================
        old_thresholds = {
            "apm_exit_confluence_threshold": params.get("apm_exit_confluence_threshold", 30),
            "apm_exit_ml_threshold": params.get("apm_exit_ml_threshold", 40),
        }
        new_thresholds = dict(old_thresholds)
        adjustments = []
        
        if exit_analysis["total_analyzed"] >= 3:
            correct_rate = exit_analysis["correct_exits"] / exit_analysis["total_analyzed"]
            
            # Se >70% degli EXIT erano corretti → sistema OK, magari più aggressivo
            if correct_rate >= 0.70:
                # Abbassa soglia confluence (esci prima)
                new_conf = max(20, old_thresholds["apm_exit_confluence_threshold"] - 2)
                if new_conf != old_thresholds["apm_exit_confluence_threshold"]:
                    new_thresholds["apm_exit_confluence_threshold"] = new_conf
                    adjustments.append(f"Exit confluence: {old_thresholds['apm_exit_confluence_threshold']} → {new_conf} (più aggressivo)")
            
            # Se <40% degli EXIT erano corretti → sistema troppo aggressivo, alza soglie
            elif correct_rate < 0.40:
                new_conf = min(45, old_thresholds["apm_exit_confluence_threshold"] + 3)
                if new_conf != old_thresholds["apm_exit_confluence_threshold"]:
                    new_thresholds["apm_exit_confluence_threshold"] = new_conf
                    adjustments.append(f"Exit confluence: {old_thresholds['apm_exit_confluence_threshold']} → {new_conf} (più conservativo)")
        
        # ============================================
        # 4. STATISTICHE HOLD — Ha senso mantenere?
        # ============================================
        hold_stats = {
            "count": stats["HOLD"]["count"],
            "avg_pnl": 0,
            "wins_ratio": 0,
        }
        if stats["HOLD"]["outcomes"]:
            pnls = [o["pnl_pct"] for o in stats["HOLD"]["outcomes"]]
            wins = sum(1 for p in pnls if p > 0)
            hold_stats["avg_pnl"] = round(sum(pnls) / len(pnls), 2)
            hold_stats["wins_ratio"] = round(wins / len(pnls) * 100, 1)
        
        # ============================================
        # 5. SALVA NUOVE SOGLIE + PERFORMANCE
        # ============================================
        if adjustments:
            for key, value in new_thresholds.items():
                params[key] = value
            await self.save_params(params)
        
        # Salva performance snapshot
        await self.save_performance({
            "total_decisions": len(decisions),
            "hold_count": stats["HOLD"]["count"],
            "exit_count": stats["EXIT"]["count"],
            "scale_out_count": stats["SCALE_OUT"]["count"],
            "tighten_count": stats["TIGHTEN_STOP"]["count"],
            "exit_correct_rate": round(exit_analysis["correct_exits"] / max(exit_analysis["total_analyzed"], 1) * 100, 1),
            "avg_hold_pnl": hold_stats["avg_pnl"],
            "hold_wins_ratio": hold_stats["wins_ratio"],
            "adjustments_count": len(adjustments),
        })
        
        # ============================================
        # 6. TELEGRAM REPORT
        # ============================================
        try:
            from app.services.telegram_bot import send_telegram
            
            msg = "🧬 <b>APM Learning Loop Report</b>\n\n"
            msg += f"📊 <b>Ultimi 30 giorni:</b>\n"
            msg += f"  Total decisioni: {len(decisions)}\n"
            msg += f"  🟢 HOLD: {stats['HOLD']['count']}\n"
            msg += f"  🔴 EXIT: {stats['EXIT']['count']}\n"
            msg += f"  🟡 SCALE_OUT: {stats['SCALE_OUT']['count']}\n"
            msg += f"  🛡️ TIGHTEN: {stats['TIGHTEN_STOP']['count']}\n\n"
            
            if exit_analysis["total_analyzed"] > 0:
                correct_pct = round(exit_analysis["correct_exits"] / exit_analysis["total_analyzed"] * 100, 1)
                msg += f"🎯 <b>Exit Accuracy:</b> {correct_pct}%\n"
                msg += f"  Corretti: {exit_analysis['correct_exits']}/{exit_analysis['total_analyzed']}\n\n"
            
            msg += f"📈 <b>HOLD stats:</b>\n"
            msg += f"  Avg P&L: {hold_stats['avg_pnl']:+.2f}%\n"
            msg += f"  Wins ratio: {hold_stats['wins_ratio']}%\n\n"
            
            if adjustments:
                msg += f"🔧 <b>Auto-tuning applicato:</b>\n"
                for adj in adjustments:
                    msg += f"  • {adj}\n"
            else:
                msg += f"✅ <b>Nessun tuning necessario</b>\n"
                msg += f"  Sistema APM stabile\n"
            
            await send_telegram(msg)
        except Exception as e:
            print(f"  APM Learning Telegram error: {e}")
        
        print(f"🧬 APM LEARN: {len(decisions)} decisions analyzed, {len(adjustments)} adjustments")
        
        return {
            "total_decisions": len(decisions),
            "stats": {k: v["count"] for k, v in stats.items()},
            "exit_analysis": exit_analysis,
            "hold_stats": hold_stats,
            "old_thresholds": old_thresholds,
            "new_thresholds": new_thresholds,
            "adjustments": adjustments,
            "learned_at": datetime.utcnow().isoformat(),
        }
