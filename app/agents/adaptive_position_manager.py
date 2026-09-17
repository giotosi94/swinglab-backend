"""
🎯 AGENTE 5: Adaptive Position Manager (APM) v1.7
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Gestisce le posizioni aperte in modo ADATTIVO.
Rivaluta periodicamente la tesi originale di ogni posizione.

4 DECISIONI:
- 🟢 HOLD: tesi valida, mantieni
- 🟡 SCALE_OUT: chiudi parziale, floor sul resto
- 🔴 EXIT: tesi rotta, chiudi 100%
- 🛡️ TIGHTEN_STOP: proteggi profit, alza SL

TRIGGER:
- Timer (apm_check_interval_hours)
- Urgent (target hit / drop critico) — ad ogni pipeline

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHANGELOG v1.7 (audit fix)
- P0-5: _calc_confluence riceve i params dell'ALPHA, non quelli APM.
        Prima factor_weights era vuoto → tutti i pesi a 1.0 → la
        confluence "now" era su una scala diversa da quella del buy,
        rendendo la logica relativa v1.5 insensata.
- P0-6: confronta confluence_raw (pre sector_penalty e sentiment_adj)
        quando disponibile nel buy_trade. Fallback su confluence.
- P1-3: rimossi i fallback inline divergenti. get_params() ora fa il
        merge con default_params(), quindi i default vivono in un posto solo.
- P1-4: il floor post scale-out non abbassa mai uno stop gia' piu' alto.
- P2-13: check_urgent_triggers non agisce a mercato chiuso (evitava
         di marcare last_target_hit senza che la vendita avvenisse).
- default_params allineati ai valori decisi: 30/30/25, min_negative 3,
  interval 1h, soglie profilo Aggressivo.
"""

from datetime import datetime, timedelta
from app.agents.base_agent import BaseAgent
from app.db.mongodb import get_db
from app.services.alpaca_trader import (
    get_positions, close_position, update_stop_loss, close_position_partial
)


class AdaptivePositionManager(BaseAgent):
    """🎯 APM v1.7 — Adaptive Position Manager"""

    def __init__(self):
        super().__init__(name="adaptive_position_manager", version="1.7")

    def default_params(self) -> dict:
        return {
            # ===== MASTER TOGGLE =====
            "apm_enabled": True,

            # ===== EXIT thresholds =====
            "apm_exit_confluence_threshold": 25,
            "apm_exit_ml_threshold": 35,
            "apm_exit_min_negative_factors": 3,

            # ===== SCALE OUT targets =====
            "apm_scaling_enabled": True,
            "apm_target_1_pct": 6.0,
            "apm_target_1_size": 30,
            "apm_target_2_pct": 12.0,
            "apm_target_2_size": 30,
            "apm_target_3_pct": 25.0,
            "apm_target_3_size": 25,

            # ===== TIGHTEN STOP =====
            "apm_tighten_profit_threshold": 3.0,
            "apm_tighten_new_sl_distance": 2.0,

            # ===== FREQUENCY =====
            "apm_check_interval_hours": 1,
            "apm_urgent_check_drop_pct": 5.0,

            # ===== ANTI-CHURNING =====
            "apm_min_holding_hours": 24,

            # ===== FLOOR post scale-out (profit lock) =====
            "apm_floor_t1_pct": 0.0,
            "apm_floor_t2_pct": 3.0,
            "apm_floor_t3_pct": 8.0,
        }

    # ==========================================
    # HELPERS
    # ==========================================

    @staticmethod
    def _entry_confluence(buy_trade) -> float:
        """
        🔧 P0-6 — Confluence d'ingresso confrontabile con il ricalcolo.

        La 'confluence' salvata nel buy include sector_penalty (-5) e
        sentiment_adj (+5 / -8 / -15 earnings). Il ricalcolo APM non li
        applica: confrontarli direttamente introduce un bias sistematico
        (su un'entry con earnings la 'now' risulta ~15 punti piu' alta
        per pura aritmetica, e il fattore non scatta mai).

        Usa confluence_raw se presente, altrimenti scorpora l'aggiustamento
        sentiment noto, altrimenti ripiega su confluence.
        """
        if not buy_trade:
            return 50.0

        raw = buy_trade.get("confluence_raw")
        if raw is not None:
            try:
                return float(raw)
            except (TypeError, ValueError):
                pass

        conf = float(buy_trade.get("confluence", 50) or 50)
        adj = buy_trade.get("sentiment_adj")
        if adj is not None:
            try:
                conf = conf - float(adj)
            except (TypeError, ValueError):
                pass
        return conf

    @staticmethod
    def _is_market_open() -> bool:
        """🔧 P2-13 — evita azioni a mercato chiuso."""
        try:
            from app.agents.executor import Executor
            return bool(Executor.is_market_open().get("is_open"))
        except Exception:
            return True

    def _floor_for_target(self, entry_price: float, target_hit: int, params: dict) -> float:
        """Floor fisso post scale-out, configurabile."""
        if target_hit == 1:
            pct = params.get("apm_floor_t1_pct", 0.0)
        elif target_hit == 2:
            pct = params.get("apm_floor_t2_pct", 3.0)
        else:
            pct = params.get("apm_floor_t3_pct", 8.0)
        return entry_price * (1 + pct / 100)

    @staticmethod
    def _resolve_targets(buy_trade, params):
        """Adaptive targets dal buy, fallback sui params."""
        t1 = params.get("apm_target_1_pct")
        t2 = params.get("apm_target_2_pct")
        t3 = params.get("apm_target_3_pct")

        if buy_trade:
            a1 = buy_trade.get("adaptive_t1_pct")
            a2 = buy_trade.get("adaptive_t2_pct")
            a3 = buy_trade.get("adaptive_t3_pct")
            if a1 and a1 > 0:
                t1 = a1
                t2 = a2 if a2 else t2
                t3 = a3 if a3 else t3
        return t1, t2, t3

    @staticmethod
    def _last_target_hit(buy_trade) -> int:
        """last_target_hit con safety net sui record legacy."""
        if not buy_trade:
            return 0
        last = buy_trade.get("last_target_hit", 0) or 0
        if buy_trade.get("partial_scaled_out") and last < 1:
            last = 1
        return int(last)

    # ==========================================
    # URGENT TRIGGERS (ad ogni pipeline)
    # ==========================================

    async def check_urgent_triggers(self, context: dict) -> dict:
        """
        Check veloce SOLO su trigger matematici (target hit, drop).
        Zero LLM, zero confluence recalc.
        """
        db = get_db()
        params = await self.get_params()

        if not params.get("apm_enabled", True):
            return {"status": "disabled", "actions_taken": []}

        positions = context.get("positions", [])
        if not positions:
            return {"status": "no_positions", "actions_taken": []}

        # 🔧 P2-13 — a mercato chiuso il partial close non viene eseguito,
        # ma last_target_hit verrebbe comunque scritto a DB.
        if not self._is_market_open():
            return {"status": "market_closed", "actions_taken": []}

        t1_size = params.get("apm_target_1_size")
        t2_size = params.get("apm_target_2_size")
        t3_size = params.get("apm_target_3_size")
        urgent_drop_pct = params.get("apm_urgent_check_drop_pct")
        scaling_enabled = params.get("apm_scaling_enabled", True)

        actions_taken = []

        for pos in positions:
            symbol = pos.get("symbol")
            current_price = float(pos.get("current_price", 0))
            entry_price = float(pos.get("avg_entry_price", 0))
            pnl_pct = float(pos.get("unrealized_plpc", 0)) * 100

            if not symbol or entry_price <= 0:
                continue

            buy_trade = await db.trade_history.find_one(
                {"ticker": symbol, "side": "buy", "sell_linked": {"$ne": True}},
                sort=[("date", -1)]
            )
            if not buy_trade:
                continue

            t1_pct, t2_pct, t3_pct = self._resolve_targets(buy_trade, params)
            last_target_hit = self._last_target_hit(buy_trade)

            action = None
            reason = None
            size_pct = 0
            target_num = 0

            if scaling_enabled and pnl_pct >= t3_pct and last_target_hit < 3:
                action = "SCALE_OUT"
                target_num = 3
                size_pct = t3_size
                reason = f"URGENT T3 hit (+{pnl_pct:.1f}% >= +{t3_pct}%)"
            elif scaling_enabled and pnl_pct >= t2_pct and last_target_hit < 2:
                action = "SCALE_OUT"
                target_num = 2
                size_pct = t2_size
                reason = f"URGENT T2 hit (+{pnl_pct:.1f}% >= +{t2_pct}%)"
            elif scaling_enabled and pnl_pct >= t1_pct and last_target_hit < 1:
                action = "SCALE_OUT"
                target_num = 1
                size_pct = t1_size
                reason = f"URGENT T1 hit (+{pnl_pct:.1f}% >= +{t1_pct}%)"
            elif pnl_pct <= -urgent_drop_pct:
                print(f"  ⚠️ URGENT: {symbol} drop {pnl_pct:.1f}% — review nel prossimo run APM completo")
                continue

            if action == "SCALE_OUT":
                print(f"  🚨 URGENT TRIGGER {symbol}: {reason}")

                action_taken, action_details = await self._execute_scale_out(
                    symbol, pos, buy_trade, target_num, size_pct, reason, params
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

                    await self.log_decision(
                        decision_type="apm_urgent_scale_out",
                        data=decision_log,
                        reasoning=reason,
                        confidence=80,
                    )

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

    # ==========================================
    # ANALYZE (timer-based, completo)
    # ==========================================

    async def analyze(self, context: dict) -> dict:
        """Analizza tutte le posizioni aperte e decide azione per ciascuna."""
        db = get_db()
        params = await self.get_params()

        if not params.get("apm_enabled", True):
            return {"status": "disabled", "message": "APM is disabled in settings"}

        market_ctx = context.get("market_context", {})
        self._current_market_regime = market_ctx.get("market_regime", "NEUTRAL")
        self._current_market_confidence = market_ctx.get("regime_confidence", 50)

        positions = context.get("positions", [])
        ml_map = context.get("ml_map", {})

        if not positions:
            return {"status": "no_positions", "message": "No positions to analyze", "decisions": []}

        should_run = await self._should_run_now(params)
        if not should_run["run"]:
            return {
                "status": "skipped_timer",
                "message": should_run["reason"],
                "next_check": should_run.get("next_check"),
                "decisions": [],
            }

        assets = await db.assets.find({}, {
            "price_history": 0, "vp_distribution": 0, "multi_tf_vp": 0
        }).to_list(300)
        assets_map = {a["ticker"]: a for a in assets}

        # 🔧 P0-5 — CRITICO: il ricalcolo confluence deve usare i params
        # dell'AlphaStrategist (factor_weights, soglie ML/trend), NON quelli
        # dell'APM. Con i params APM, factor_weights era {} → tutti i pesi
        # tornavano a 1.0 e la scala non era confrontabile con quella del buy.
        from app.agents.alpha_strategist import AlphaStrategist
        alpha = AlphaStrategist()
        try:
            alpha_params = await alpha.get_params()
        except Exception as e:
            print(f"  ⚠️ APM: alpha.get_params() error: {e} — uso i default Alpha")
            alpha_params = alpha.default_params()

        decisions = []
        actions_taken = []

        for pos in positions:
            symbol = pos.get("symbol")
            current_price = float(pos.get("current_price", 0))
            entry_price = float(pos.get("avg_entry_price", 0))
            pnl_pct = float(pos.get("unrealized_plpc", 0)) * 100

            if not symbol or entry_price <= 0:
                continue

            buy_trade = await db.trade_history.find_one(
                {"ticker": symbol, "side": "buy", "sell_linked": {"$ne": True}},
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

            # 🔧 P0-6 — confluence d'ingresso confrontabile
            original_confluence = self._entry_confluence(buy_trade)
            original_stop = buy_trade.get("stop_loss", 0)
            original_target = buy_trade.get("target", 0)
            original_ml_score = buy_trade.get("ml_score", 0)
            original_ml_pred = buy_trade.get("ml_prediction", "unknown")

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

            try:
                current_conf_data = alpha._calc_confluence(
                    asset, market_ctx, alpha_params, current_ml_data
                )
                current_confluence = current_conf_data.get("score", 0)
            except Exception as e:
                print(f"  ⚠️ APM confluence calc error {symbol}: {e}")
                current_confluence = original_confluence

            current_ml_score = current_ml_data.get("ml_score", 0)
            current_ml_pred = current_ml_data.get("ml_prediction", "unknown")
            current_trend_pred = current_ml_data.get("trend_prediction", "unknown")

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

            action_taken = False
            action_details = {}

            if decision == "EXIT":
                action_taken, action_details = await self._execute_exit(
                    symbol, pos, buy_trade, reason, current_confluence, current_ml_score
                )
            elif decision == "SCALE_OUT":
                action_taken, action_details = await self._execute_scale_out(
                    symbol, pos, buy_trade,
                    details.get("target_hit", 1),
                    details.get("size_pct", params.get("apm_target_1_size")),
                    reason, params
                )
            elif decision == "TIGHTEN_STOP":
                action_taken, action_details = await self._execute_tighten_stop(
                    symbol, pos, buy_trade, details.get("new_stop", 0), reason
                )

            decision_log = {
                "ticker": symbol,
                "decision": decision,
                "reason": reason,
                "current_pnl_pct": round(pnl_pct, 2),
                "current_price": current_price,
                "entry_price": entry_price,
                "state_snapshot": {
                    "confluence_original": round(original_confluence, 1),
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

            await self.log_decision(
                decision_type=f"apm_{decision.lower()}",
                data=decision_log,
                reasoning=reason,
                confidence=70 if action_taken else 40,
            )

        await db.apm_state.update_one(
            {"_id": "last_run"},
            {"$set": {
                "timestamp": datetime.utcnow(),
                "decisions_count": len(decisions),
                "actions_count": len(actions_taken),
            }},
            upsert=True
        )

        summary = self._build_summary(decisions, actions_taken)

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
        """Check timer."""
        db = get_db()
        interval_hours = params.get("apm_check_interval_hours")

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

    # ==========================================
    # DECISION LOGIC
    # ==========================================

    def _decide_action(self, pos, pnl_pct, original_confluence, current_confluence,
                       original_ml_score, current_ml_score, current_ml_pred,
                       current_trend_pred, original_target, original_stop, params) -> dict:
        """Ritorna: {"decision": "...", "reason": "...", "details": {...}}"""

        buy_trade_ref = getattr(self, "_current_buy_trade", None)

        t1_pct, t2_pct, t3_pct = self._resolve_targets(buy_trade_ref, params)
        last_target_hit_now = self._last_target_hit(buy_trade_ref)

        exit_conf_th_base = params.get("apm_exit_confluence_threshold")
        exit_ml_th_base = params.get("apm_exit_ml_threshold")

        market_regime = getattr(self, "_current_market_regime", "NEUTRAL")

        regime_multipliers = {
            "BULL": {"conf": -10, "ml": -10},
            "NEUTRAL": {"conf": 0, "ml": 0},
            "BEAR": {"conf": +10, "ml": +10},
            "CRASH": {"conf": +15, "ml": +15},
        }
        adj = regime_multipliers.get(market_regime, {"conf": 0, "ml": 0})
        exit_conf_th = max(15, min(50, exit_conf_th_base + adj["conf"]))
        exit_ml_th = max(20, min(60, exit_ml_th_base + adj["ml"]))

        min_negative = params.get("apm_exit_min_negative_factors")

        # Detect "ML flat" (output degenerato del modello)
        ml_score_looks_flat = (
            85 < current_ml_score < 95
            and abs(current_ml_score - 92.3) < 1.0
        )

        # 🆕 v1.5 — Logica RELATIVA alla tesi d'ingresso (anti-churning).
        # Esce se la tesi si e' ROTTA rispetto al buy, non se un numero
        # e' basso in assoluto.
        confluence_drop = original_confluence - current_confluence
        negative_factors = []

        if confluence_drop >= 10 and current_confluence < exit_conf_th:
            negative_factors.append(
                f"confluence crollata {original_confluence:.0f}→{current_confluence:.0f}"
            )

        if not ml_score_looks_flat:
            if current_ml_pred == "LOSS" and current_ml_score < exit_ml_th:
                negative_factors.append(f"ML LOSS {current_ml_score:.0f}% < {exit_ml_th}%")

        if current_trend_pred == "DOWN":
            negative_factors.append("Trend DOWN")

        # ============================================
        # 🔴 EXIT
        # ============================================
        min_hold_h = params.get("apm_min_holding_hours", 24)
        buy_date_ref = buy_trade_ref.get("date") if buy_trade_ref else None
        hours_held = 999.0
        if buy_date_ref:
            try:
                hours_held = (datetime.utcnow() - buy_date_ref).total_seconds() / 3600
            except Exception:
                hours_held = 999.0
        too_fresh = hours_held < min_hold_h

        runner_in_profit = last_target_hit_now >= 1 and pnl_pct > 0

        if len(negative_factors) >= min_negative and not runner_in_profit and not too_fresh:
            return {
                "decision": "EXIT",
                "reason": (
                    f"Tesi invalidata: {len(negative_factors)} fattori negativi. "
                    f"Confluence {original_confluence:.0f}→{current_confluence:.0f}, "
                    f"ML {original_ml_score:.0f}%→{current_ml_score:.0f}%. "
                    f"Meglio uscire con P&L {pnl_pct:+.1f}% ora che rischiare peggio."
                ),
                "details": {
                    "negative_factors": negative_factors,
                    "confluence_drop": round(confluence_drop, 1),
                    "ml_drop": original_ml_score - current_ml_score,
                    "hours_held": round(hours_held, 1),
                },
            }

        # ============================================
        # 🟡 SCALE_OUT (multi-target)
        # ============================================
        if params.get("apm_scaling_enabled", True):
            t1_size = params.get("apm_target_1_size")
            t2_size = params.get("apm_target_2_size")
            t3_size = params.get("apm_target_3_size")

            if pnl_pct >= t3_pct and last_target_hit_now < 3:
                return {
                    "decision": "SCALE_OUT",
                    "reason": (
                        f"Target 3 raggiunto (+{pnl_pct:.1f}% >= +{t3_pct}%). "
                        f"Chiudo {t3_size}% della posizione residua."
                    ),
                    "details": {"target_hit": 3, "size_pct": t3_size},
                }
            elif pnl_pct >= t2_pct and last_target_hit_now < 2:
                return {
                    "decision": "SCALE_OUT",
                    "reason": (
                        f"Target 2 raggiunto (+{pnl_pct:.1f}% >= +{t2_pct}%). "
                        f"Chiudo {t2_size}%, lascio correre il resto."
                    ),
                    "details": {"target_hit": 2, "size_pct": t2_size},
                }
            elif pnl_pct >= t1_pct and last_target_hit_now < 1:
                return {
                    "decision": "SCALE_OUT",
                    "reason": (
                        f"Target 1 raggiunto (+{pnl_pct:.1f}% >= +{t1_pct}%). "
                        f"Chiudo {t1_size}% per prendere profit, floor a break-even sul resto."
                    ),
                    "details": {"target_hit": 1, "size_pct": t1_size},
                }

        # ============================================
        # 🛡️ TIGHTEN STOP
        # ============================================
        # v1.6 — NON toccare i runner post-T1: il floor "lascia correre"
        # gestisce gia' quelle posizioni.
        tighten_th = params.get("apm_tighten_profit_threshold")
        current_price = float(pos.get("current_price", 0))

        if (last_target_hit_now == 0
                and pnl_pct >= tighten_th
                and (current_ml_pred == "LOSS" or current_trend_pred == "DOWN")):
            new_sl_distance = params.get("apm_tighten_new_sl_distance") / 100
            new_stop = round(current_price * (1 - new_sl_distance), 2)

            return {
                "decision": "TIGHTEN_STOP",
                "reason": (
                    f"Profit +{pnl_pct:.1f}% ma segnale bearish. "
                    f"Alzo SL a ${new_stop} (-{new_sl_distance*100:.0f}% dal current)."
                ),
                "details": {"new_stop": new_stop},
            }

        # ============================================
        # 🟢 HOLD
        # ============================================
        return {
            "decision": "HOLD",
            "reason": (
                f"Tesi ancora valida. Confluence {current_confluence:.0f} "
                f"(era {original_confluence:.0f}), ML {current_ml_score:.0f}%. "
                f"P&L {pnl_pct:+.1f}%. Mantengo posizione."
            ),
            "details": {"negative_factors": negative_factors},
        }

    # ==========================================
    # EXECUTION
    # ==========================================

    async def _execute_exit(self, symbol, pos, buy_trade, reason, current_confluence, current_ml_score):
        """Chiusura 100% della posizione."""
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

            # 🔧 P1-6 — propaga i dati del buy sul record di sell,
            # altrimenti il learning dell'Alpha resta cieco.
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
                "confluence": buy_trade.get("confluence", 0),
                "confluence_raw": buy_trade.get("confluence_raw"),
                "rsi_at_entry": buy_trade.get("rsi_at_entry", 50),
                "ml_score": buy_trade.get("ml_score", 0),
                "risk_reward": buy_trade.get("risk_reward", 0),
                "order_id": sell_order_id,
                "buy_order_id": buy_trade.get("order_id", ""),
                "agent": "adaptive_position_manager",
                "date": datetime.utcnow(),
                "source": "apm_v1",
                "apm_confluence_at_exit": current_confluence,
                "apm_ml_at_exit": current_ml_score,
            })

            await db.trade_history.update_one(
                {"_id": buy_trade["_id"]},
                {"$set": {"sell_linked": True, "sell_order_id": sell_order_id}}
            )

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

    async def _execute_scale_out(self, symbol, pos, buy_trade, target_hit, size_pct, reason, params=None):
        """
        Chiusura parziale REALE su Alpaca + floor fisso sul residuo.

        NOTA ARCHITETTURALE:
        Con il sizing notional le posizioni sono frazionarie e Alpaca NON
        accetta ordini stop frazionari (errore 42210000). La protezione e'
        quindi software: il floor viene scritto su db.trailing_stops e
        l'Executor lo applica in _check_software_sl_tp. apm_managed=True
        impedisce a _manage_trailing_stops di sovrascriverlo.
        """
        db = get_db()
        if params is None:
            params = await self.get_params()

        try:
            current_qty = float(pos.get("qty", 0))
            qty_to_close = round(current_qty * (size_pct / 100), 4)
            qty_remaining = round(current_qty - qty_to_close, 4)

            if qty_to_close <= 0.0001:
                return False, {"error": "qty_to_close too small"}

            current_price = float(pos.get("current_price", 0))
            entry_price = float(buy_trade.get("entry_price", 0))

            # Cancella eventuali ordini sell aperti sul ticker (se esistono)
            from app.services.alpaca_trader import get_orders, cancel_order
            open_orders = await get_orders(status="open", limit=50)
            cancelled_orders = 0
            if open_orders:
                for o in open_orders:
                    if o.get("symbol") == symbol and o.get("side") == "sell":
                        await cancel_order(o.get("id"))
                        cancelled_orders += 1

            close_result = await close_position_partial(symbol, qty_to_close)

            if close_result is None:
                return False, {"error": "partial close failed", "cancelled_orders": cancelled_orders}

            pnl_pct_partial = round(((current_price - entry_price) / entry_price) * 100, 2) if entry_price > 0 else 0
            pnl_dollar_partial = round((current_price - entry_price) * qty_to_close, 2)
            buy_date = buy_trade.get("date", datetime.utcnow())
            days_held = max(1, (datetime.utcnow() - buy_date).days) if buy_date else 1

            sell_order_id = f"apm_scale_{target_hit}_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

            # 🔧 P1-6 — dati del buy propagati anche sulle tranche
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
                "confluence": buy_trade.get("confluence", 0),
                "confluence_raw": buy_trade.get("confluence_raw"),
                "rsi_at_entry": buy_trade.get("rsi_at_entry", 50),
                "ml_score": buy_trade.get("ml_score", 0),
                "risk_reward": buy_trade.get("risk_reward", 0),
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

            await db.trade_history.update_one(
                {"_id": buy_trade["_id"]},
                {"$set": {
                    "shares": float(qty_remaining),
                    "partial_scaled_out": True,
                    "last_scale_out_at": datetime.utcnow(),
                    "last_target_hit": target_hit,
                }}
            )

            # 🔧 P1-4 — Il floor non deve MAI abbassare uno stop gia' piu' alto.
            # Scenario reale: TIGHTEN porta lo stop a entry+10%, poi scatta T1
            # e il floor break-even lo riportava indietro di 10 punti.
            floor_price = self._floor_for_target(entry_price, target_hit, params)
            new_stop = round(floor_price, 2)

            existing = await db.trailing_stops.find_one({"ticker": symbol})
            existing_stop = float(existing.get("stop_price", 0) or 0) if existing else 0
            if existing_stop > new_stop:
                print(f"  🛡️ {symbol}: mantengo stop esistente ${existing_stop:.2f} > floor ${new_stop:.2f}")
                new_stop = existing_stop

            await db.trailing_stops.update_one(
                {"ticker": symbol},
                {"$set": {
                    "ticker": symbol,
                    "stop_price": new_stop,
                    "floor_price": new_stop,
                    "trailing_active": False,
                    "apm_managed": True,
                    "last_target_hit": target_hit,
                    "reason": f"APM scale-out T{target_hit}: floor fisso ${new_stop} (lascia correre verso target)",
                    "updated_at": datetime.utcnow(),
                    "source": "apm_v1_scale_out",
                }},
                upsert=True
            )

            # Best-effort: se per qualche motivo esiste un ordine stop sul broker
            # (posizione a shares intere), prova ad allinearlo. Con le frazionarie
            # fallisce: e' previsto, la protezione resta software.
            try:
                await update_stop_loss(symbol, new_stop)
            except Exception:
                pass

            print(f"  🟡 APM SCALE_OUT T{target_hit} {symbol}: closed {qty_to_close:.4f} shares "
                  f"(P&L {pnl_pct_partial:+.2f}%, ${pnl_dollar_partial:+.0f}), "
                  f"floor → ${new_stop:.2f}, remaining {qty_remaining:.4f}")

            return True, {
                "action": "SCALE_OUT_REAL",
                "target_hit": target_hit,
                "size_pct": size_pct,
                "qty_closed": float(qty_to_close),
                "qty_remaining": float(qty_remaining),
                "pnl_pct": pnl_pct_partial,
                "pnl_dollar": pnl_dollar_partial,
                "new_stop": new_stop,
                "cancelled_orders": cancelled_orders,
            }

        except Exception as e:
            print(f"  ⚠️ APM SCALE_OUT error {symbol}: {e}")
            return False, {"error": str(e)}

    async def _execute_tighten_stop(self, symbol, pos, buy_trade, new_stop, reason):
        """Tightening dello stop loss (mono-direzionale, idempotente)."""
        db = get_db()
        try:
            existing = await db.trailing_stops.find_one({"ticker": symbol})
            current_stop = float(existing.get("stop_price", 0) or 0) if existing else 0

            if current_stop > 0 and new_stop <= current_stop * 1.003:
                return False, {"skipped": "stop già adeguato", "current_stop": current_stop}

            await db.trailing_stops.update_one(
                {"ticker": symbol},
                {"$set": {
                    "ticker": symbol,
                    "stop_price": new_stop,
                    "reason": f"APM tighten: {reason}",
                    "updated_at": datetime.utcnow(),
                    "source": "apm_v1",
                    "apm_managed": True,
                    "tighten": True,
                }},
                upsert=True
            )

            try:
                await update_stop_loss(symbol, new_stop)
            except Exception:
                pass

            print(f"  🛡️ APM TIGHTEN {symbol}: SL → ${new_stop}")

            return True, {"action": "TIGHTEN_STOP", "new_stop": new_stop}
        except Exception as e:
            print(f"  ⚠️ APM TIGHTEN error {symbol}: {e}")
            return False, {"error": str(e)}

    # ==========================================
    # SUMMARY / NOTIFICHE
    # ==========================================

    def _build_summary(self, decisions, actions_taken):
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

    # ==========================================
    # LEARNING
    # ==========================================

    async def learn(self) -> dict:
        """
        🧬 APM Learning Loop

        ⚠️ AUTO-TUNING DISABILITATO (audit P2-8).
        La metrica precedente (exit_pnl > -1 = "corretto") non misurava
        cosa succede DOPO l'uscita, quindi non poteva rilevare le uscite
        premature — che sono esattamente il problema osservato sui 193
        trade (+1.13% medio dopo l'exit). Peggio: con correct_rate >= 0.70
        ABBASSAVA la soglia rendendo l'APM piu' aggressivo.

        Ora il loop produce solo statistiche e report, senza toccare i
        parametri. Riattivare dopo aver riscritto la metrica confrontando
        il prezzo a +5/+10 giorni dall'uscita.
        """
        db = get_db()
        params = await self.get_params()

        cutoff = datetime.utcnow() - timedelta(days=30)

        decisions = await self._col_decisions().find({
            "created_at": {"$gte": cutoff},
        }).sort("created_at", -1).to_list(500)

        if len(decisions) < 10:
            return {
                "message": "Not enough decisions to learn (need 10+)",
                "count": len(decisions),
                "auto_tuning": "disabled",
            }

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

        # ---- Analisi EXIT: cosa e' successo DOPO l'uscita ----
        exit_analysis = {
            "total_analyzed": 0,
            "price_higher_after": 0,
            "price_lower_after": 0,
            "avg_move_after_pct": 0.0,
        }

        moves_after = []

        for exit_dec in stats["EXIT"]["outcomes"]:
            ticker = exit_dec["ticker"]
            exit_time = exit_dec["created_at"]
            if not ticker or not exit_time:
                continue

            sell_trade = await db.trade_history.find_one({
                "ticker": ticker,
                "side": "sell",
                "reason": "APM_EXIT",
                "date": {"$gte": exit_time - timedelta(hours=2),
                         "$lte": exit_time + timedelta(hours=2)},
            })
            if not sell_trade:
                continue

            exit_price = float(sell_trade.get("exit_price", 0) or 0)
            if exit_price <= 0:
                continue

            bars_doc = await db.stock_bars.find_one({"ticker": ticker}, {"bars": 1})
            bars = (bars_doc or {}).get("bars", [])
            if not bars:
                continue

            exit_day = exit_time.strftime("%Y-%m-%d")
            idx = next((i for i, b in enumerate(bars) if b.get("date", "") >= exit_day), None)
            if idx is None:
                continue

            fwd_idx = min(idx + 5, len(bars) - 1)
            if fwd_idx <= idx:
                continue

            price_after = float(bars[fwd_idx].get("c", 0) or 0)
            if price_after <= 0:
                continue

            move_pct = (price_after - exit_price) / exit_price * 100
            moves_after.append(move_pct)
            exit_analysis["total_analyzed"] += 1
            if move_pct > 0:
                exit_analysis["price_higher_after"] += 1
            else:
                exit_analysis["price_lower_after"] += 1

        if moves_after:
            exit_analysis["avg_move_after_pct"] = round(sum(moves_after) / len(moves_after), 2)

        hold_stats = {"count": stats["HOLD"]["count"], "avg_pnl": 0, "wins_ratio": 0}
        if stats["HOLD"]["outcomes"]:
            pnls = [o["pnl_pct"] for o in stats["HOLD"]["outcomes"]]
            wins = sum(1 for p in pnls if p > 0)
            hold_stats["avg_pnl"] = round(sum(pnls) / len(pnls), 2)
            hold_stats["wins_ratio"] = round(wins / len(pnls) * 100, 1)

        await self.save_performance({
            "total_decisions": len(decisions),
            "hold_count": stats["HOLD"]["count"],
            "exit_count": stats["EXIT"]["count"],
            "scale_out_count": stats["SCALE_OUT"]["count"],
            "tighten_count": stats["TIGHTEN_STOP"]["count"],
            "exit_avg_move_after_pct": exit_analysis["avg_move_after_pct"],
            "exit_premature_count": exit_analysis["price_higher_after"],
            "avg_hold_pnl": hold_stats["avg_pnl"],
            "hold_wins_ratio": hold_stats["wins_ratio"],
            "auto_tuning": "disabled",
        })

        try:
            from app.services.telegram_bot import send_telegram

            msg = "🧬 <b>APM Learning Report</b>\n\n"
            msg += "📊 <b>Ultimi 30 giorni:</b>\n"
            msg += f"  Total decisioni: {len(decisions)}\n"
            msg += f"  🟢 HOLD: {stats['HOLD']['count']}\n"
            msg += f"  🔴 EXIT: {stats['EXIT']['count']}\n"
            msg += f"  🟡 SCALE_OUT: {stats['SCALE_OUT']['count']}\n"
            msg += f"  🛡️ TIGHTEN: {stats['TIGHTEN_STOP']['count']}\n\n"

            if exit_analysis["total_analyzed"] > 0:
                msg += "🚪 <b>Cosa e' successo dopo le EXIT (5g):</b>\n"
                msg += f"  Analizzate: {exit_analysis['total_analyzed']}\n"
                msg += f"  Prezzo SALITO dopo: {exit_analysis['price_higher_after']} (uscite premature)\n"
                msg += f"  Prezzo SCESO dopo: {exit_analysis['price_lower_after']} (uscite corrette)\n"
                msg += f"  Movimento medio: {exit_analysis['avg_move_after_pct']:+.2f}%\n\n"

            msg += "📈 <b>HOLD stats:</b>\n"
            msg += f"  Avg P&L: {hold_stats['avg_pnl']:+.2f}%\n"
            msg += f"  Wins ratio: {hold_stats['wins_ratio']}%\n\n"
            msg += "🔒 <b>Auto-tuning disabilitato</b> (protocollo 50 trade)\n"

            await send_telegram(msg)
        except Exception as e:
            print(f"  APM Learning Telegram error: {e}")

        print(f"🧬 APM LEARN: {len(decisions)} decisions analyzed, auto-tuning OFF")

        return {
            "total_decisions": len(decisions),
            "stats": {k: v["count"] for k, v in stats.items()},
            "exit_analysis": exit_analysis,
            "hold_stats": hold_stats,
            "current_thresholds": {
                "apm_exit_confluence_threshold": params.get("apm_exit_confluence_threshold"),
                "apm_exit_ml_threshold": params.get("apm_exit_ml_threshold"),
                "apm_exit_min_negative_factors": params.get("apm_exit_min_negative_factors"),
            },
            "auto_tuning": "disabled",
            "adjustments": [],
            "learned_at": datetime.utcnow().isoformat(),
        }
