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
from app.services.alpaca_trader import get_positions, close_position, update_stop_loss


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
        # Soglie
        exit_conf_th = params.get("apm_exit_confluence_threshold", 30)
        exit_ml_th = params.get("apm_exit_ml_threshold", 40)
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
        if len(negative_factors) >= min_negative:
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
            t1_pct = params.get("apm_target_1_pct", 5.0)
            t2_pct = params.get("apm_target_2_pct", 10.0)
            t3_pct = params.get("apm_target_3_pct", 20.0)
            t1_size = params.get("apm_target_1_size", 50)
            t2_size = params.get("apm_target_2_size", 30)
            t3_size = params.get("apm_target_3_size", 20)
            
            # Ha già raggiunto qualche target?
            # (Per ora semplice check: se pnl_pct >= t3, target 3; se >= t2, target 2; ecc.)
            # In produzione futura terremo track di quali target sono già stati raggiunti
            
            if pnl_pct >= t3_pct:
                return {
                    "decision": "SCALE_OUT",
                    "reason": (
                        f"Target 3 raggiunto (+{pnl_pct:.1f}% >= +{t3_pct}%). "
                        f"Chiudo {t3_size}% posizione residua, break-even sul resto."
                    ),
                    "details": {"target_hit": 3, "size_pct": t3_size},
                }
            elif pnl_pct >= t2_pct:
                return {
                    "decision": "SCALE_OUT",
                    "reason": (
                        f"Target 2 raggiunto (+{pnl_pct:.1f}% >= +{t2_pct}%). "
                        f"Chiudo {t2_size}% posizione, lascio correre il resto."
                    ),
                    "details": {"target_hit": 2, "size_pct": t2_size},
                }
            elif pnl_pct >= t1_pct:
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
        """Esegue chiusura parziale."""
        # NOTA: Alpaca supporta chiusura parziale via close_position con qty specifica
        # Per ora, semplice log — implementeremo chiusura parziale nella FASE 4 (weekend prox)
        print(f"  🟡 APM SCALE_OUT {symbol}: Target {target_hit}, size {size_pct}%")
        print(f"     ⚠️ Scale-out implementation planned for FASE 4 (weekend 19-20 luglio)")
        
        return True, {
            "action": "SCALE_OUT_LOGGED",
            "target_hit": target_hit,
            "size_pct": size_pct,
            "note": "Full implementation in FASE 4",
        }

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
        Learning loop APM.
        Analizza outcome delle decisioni passate per aggiustare soglie.
        """
        db = get_db()
        
        # Per la FASE 1 (base), il learning è minimo
        # Verrà espanso nella FASE 3 (weekend 19-20 luglio)
        
        # Recupera decisioni ultimi 30 giorni
        cutoff = datetime.utcnow() - timedelta(days=30)
        decisions = await self._col_decisions().find({
            "created_at": {"$gte": cutoff},
        }).to_list(500)
        
        if len(decisions) < 5:
            return {"message": "Not enough APM decisions to learn", "count": len(decisions)}
        
        # Conteggi
        exits = [d for d in decisions if d.get("type") == "apm_exit"]
        scales = [d for d in decisions if d.get("type") == "apm_scale_out"]
        tightens = [d for d in decisions if d.get("type") == "apm_tighten_stop"]
        holds = [d for d in decisions if d.get("type") == "apm_hold"]
        
        return {
            "total_decisions": len(decisions),
            "exits": len(exits),
            "scale_outs": len(scales),
            "tightens": len(tightens),
            "holds": len(holds),
            "message": "APM learning loop v1.0 — full analysis in FASE 3",
        }
