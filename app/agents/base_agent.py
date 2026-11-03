from datetime import datetime, timedelta
from app.db.mongodb import get_db
import math
from copy import deepcopy
from typing import Optional


class BaseAgent:
    """
    Classe base per tutti gli agenti SwingLab.

    Ogni agente ha:
    - Un nome univoco (usato per le collection MongoDB)
    - Parametri appresi (memory) che si aggiornano nel tempo
    - Log di ogni decisione con contesto completo
    - Sistema di learning con time decay
    - Versioning dei parametri + history (v3.0)
    - Guardrail / rollback leggero post-learn (v3.1)

    v2.0 — get_params() ora fa il MERGE con default_params().
    v3.0 — Learning Machine core:
        - Param history append-only (agent_param_history_{name})
        - update_params_safe() con max relative delta
        - Helper get_position_trades() per aggregare scale-out/tranche
    v3.1 — Guardrail:
        - baseline WR dopo learn
        - rollback a versione precedente se WR scende nettamente
    """

    def __init__(self, name: str, version: str = "1.0"):
        self.name = name
        self.version = version
        self.decay_days = 60
        self.min_decisions_to_learn = 5
        self.max_param_delta_pct = 0.25
        self.max_history = 50
        self.guardrail_min_days = 7
        self.guardrail_min_new_positions = 5
        self.guardrail_wr_drop_pts = 12.0

    def _col_memory(self):
        return get_db()[f"agent_memory_{self.name}"]

    def _col_param_history(self):
        return get_db()[f"agent_param_history_{self.name}"]

    def _col_decisions(self):
        return get_db()[f"agent_decisions_{self.name}"]

    def _col_performance(self):
        return get_db()[f"agent_performance_{self.name}"]

    async def get_params(self) -> dict:
        params = self.default_params()
        doc = await self._col_memory().find_one({"_id": "params"})
        if doc:
            doc.pop("_id", None)
            doc.pop("updated_at", None)
            doc.pop("agent_version", None)
            doc.pop("param_version", None)
            params.update(doc)
        return params

    async def get_params_raw(self) -> dict:
        doc = await self._col_memory().find_one({"_id": "params"})
        if doc:
            doc.pop("_id", None)
            return doc
        return {}

    async def save_params(self, params: dict, reason: str = "manual"):
        params = dict(params)
        params.pop("_id", None)
        current = await self.get_params_raw()
        now = datetime.utcnow()
        params["updated_at"] = now
        params["agent_version"] = self.version
        prev_version = current.get("param_version", 0)
        new_version = int(prev_version) + 1
        params["param_version"] = new_version
        await self._col_memory().update_one(
            {"_id": "params"}, {"$set": params}, upsert=True
        )
        history_doc = {
            "param_version": new_version,
            "agent_version": self.version,
            "reason": reason,
            "params": {k: v for k, v in params.items()
                       if k not in ("updated_at", "agent_version", "param_version")},
            "created_at": now,
        }
        await self._col_param_history().insert_one(history_doc)
        count = await self._col_param_history().count_documents({})
        if count > self.max_history:
            oldest = await self._col_param_history().find().sort(
                "created_at", 1
            ).limit(count - self.max_history).to_list(count - self.max_history)
            ids = [d["_id"] for d in oldest]
            if ids:
                await self._col_param_history().delete_many({"_id": {"$in": ids}})
        return new_version

    async def update_params_safe(self, new_params: dict, reason: str = "learn") -> dict:
        current = await self.get_params()
        applied = deepcopy(current)
        changes = {}
        for key, new_val in new_params.items():
            if key in ("updated_at", "agent_version", "param_version", "_id"):
                continue
            old_val = current.get(key)
            if not isinstance(new_val, (int, float)) or not isinstance(old_val, (int, float)):
                if new_val != old_val:
                    applied[key] = new_val
                    changes[key] = {"from": old_val, "to": new_val, "clamped": False}
                continue
            if old_val == 0:
                delta = new_val - old_val
                max_abs = 1.0
                if abs(delta) > max_abs:
                    new_val = old_val + (max_abs if delta > 0 else -max_abs)
                    clamped = True
                else:
                    clamped = False
            else:
                rel = (new_val - old_val) / abs(old_val)
                if abs(rel) > self.max_param_delta_pct:
                    new_val = old_val * (1 + self.max_param_delta_pct * (1 if rel > 0 else -1))
                    clamped = True
                else:
                    clamped = False
            if isinstance(old_val, int) and not isinstance(old_val, bool):
                new_val = int(round(new_val))
            if new_val != old_val:
                applied[key] = new_val
                changes[key] = {"from": old_val, "to": new_val, "clamped": clamped}
        if changes:
            version = await self.save_params(applied, reason=reason)
            return {
                "updated": True,
                "param_version": version,
                "changes": changes,
                "params": applied,
            }
        return {
            "updated": False,
            "param_version": current.get("param_version", 0),
            "changes": {},
            "params": applied,
        }

    async def get_param_history(self, limit: int = 20) -> list:
        docs = await self._col_param_history().find().sort(
            "created_at", -1
        ).to_list(limit)
        for d in docs:
            d["_id"] = str(d["_id"])
        return docs

    async def set_learn_baseline(self, win_rate: float, sample_count: int,
                                 param_version: int = None, extra: dict = None) -> dict:
        params = await self.get_params_raw()
        version = param_version if param_version is not None else params.get("param_version", 0)
        doc = {
            "param_version": int(version or 0),
            "win_rate": float(win_rate),
            "sample_count": int(sample_count),
            "created_at": datetime.utcnow(),
            "extra": extra or {},
        }
        await self._col_memory().update_one(
            {"_id": "learn_baseline"}, {"$set": doc}, upsert=True
        )
        return doc

    async def get_learn_baseline(self) -> dict:
        doc = await self._col_memory().find_one({"_id": "learn_baseline"}) or {}
        doc.pop("_id", None)
        return doc

    async def rollback_to_previous_version(self, reason: str = "guardrail_rollback") -> dict:
        current = await self.get_params_raw()
        cur_ver = int(current.get("param_version") or 0)
        if cur_ver <= 1:
            return {"rolled_back": False, "reason": "no_previous_version", "param_version": cur_ver}
        hist = await self._col_param_history().find_one({"param_version": cur_ver - 1})
        if not hist:
            docs = await self._col_param_history().find().sort("created_at", -1).to_list(8)
            hist = next((d for d in docs if int(d.get("param_version") or 0) < cur_ver), None)
            if not hist:
                return {"rolled_back": False, "reason": "history_missing", "param_version": cur_ver}
        restored = dict(hist.get("params") or {})
        new_ver = await self.save_params(
            restored, reason="%s_from_v%s" % (reason, hist.get("param_version"))
        )
        return {
            "rolled_back": True,
            "from_version": cur_ver,
            "restored_history_version": hist.get("param_version"),
            "new_param_version": new_ver,
            "reason": reason,
        }

    async def evaluate_guardrail(self, current_win_rate: float = None,
                                 current_sample_count: int = None) -> dict:
        baseline = await self.get_learn_baseline()
        if not baseline or baseline.get("win_rate") is None:
            return {"action": "none", "reason": "no_baseline"}

        created = baseline.get("created_at")
        age_days = 0.0
        if created:
            try:
                if isinstance(created, str):
                    created = datetime.fromisoformat(created.replace("Z", ""))
                age_days = (datetime.utcnow() - created).total_seconds() / 86400.0
            except Exception:
                age_days = 0.0

        base_wr = float(baseline.get("win_rate") or 0)
        base_n = int(baseline.get("sample_count") or 0)
        cur_n = int(current_sample_count if current_sample_count is not None else base_n)
        new_samples = max(0, cur_n - base_n)

        if age_days < self.guardrail_min_days and new_samples < self.guardrail_min_new_positions:
            return {
                "action": "wait",
                "reason": "insufficient_evidence",
                "age_days": round(age_days, 1),
                "new_samples": new_samples,
                "baseline_wr": base_wr,
                "baseline_version": baseline.get("param_version"),
            }

        if current_win_rate is None:
            return {
                "action": "wait",
                "reason": "no_current_metric",
                "baseline_wr": base_wr,
            }

        cur_wr = float(current_win_rate)
        drop = base_wr - cur_wr

        if drop >= self.guardrail_wr_drop_pts:
            rb = await self.rollback_to_previous_version(reason="guardrail_wr_drop")
            if rb.get("rolled_back"):
                await self.set_learn_baseline(
                    win_rate=cur_wr,
                    sample_count=cur_n,
                    param_version=rb.get("new_param_version"),
                    extra={"after_rollback": True, "drop_pts": drop},
                )
            return {
                "action": "rollback",
                "reason": "wr_drop",
                "baseline_wr": base_wr,
                "current_wr": cur_wr,
                "drop_pts": round(drop, 1),
                "rollback": rb,
            }

        if drop <= 0 or age_days >= self.guardrail_min_days * 2:
            await self.set_learn_baseline(
                win_rate=cur_wr,
                sample_count=cur_n,
                param_version=(await self.get_params_raw()).get("param_version"),
                extra={"refreshed": True},
            )

        return {
            "action": "ok",
            "reason": "within_tolerance" if drop > 0 else "stable_or_better",
            "baseline_wr": base_wr,
            "current_wr": cur_wr,
            "drop_pts": round(drop, 1),
            "age_days": round(age_days, 1),
        }

    def default_params(self) -> dict:
        return {}

    async def log_decision(self, decision_type: str, data: dict,
                           reasoning: str = "", confidence: float = 50.0):
        doc = {
            "agent": self.name,
            "type": decision_type,
            "data": data,
            "reasoning": reasoning,
            "confidence": confidence,
            "outcome": None,
            "outcome_date": None,
            "created_at": datetime.utcnow(),
        }
        result = await self._col_decisions().insert_one(doc)
        return str(result.inserted_id)

    def calc_weight(self, created_at: datetime) -> float:
        if not created_at:
            return 0.1
        days_old = (datetime.utcnow() - created_at).days
        if days_old <= 0:
            return 1.0
        lam = math.log(2) / self.decay_days
        return math.exp(-lam * days_old)

    async def get_position_trades(self, lookback_days: int = 180) -> list:
        db = get_db()
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)
        sells = await db.trade_history.find({
            "side": "sell",
            "date": {"$gte": cutoff},
            "pnl_pct": {"$exists": True},
        }).to_list(2000)
        positions = {}
        for t in sells:
            boid = t.get("buy_order_id") or t.get("order_id") or str(t.get("_id"))
            if not boid:
                continue
            if boid not in positions:
                positions[boid] = {
                    "buy_order_id": boid,
                    "ticker": t.get("ticker"),
                    "entry_date": t.get("buy_date") or t.get("date"),
                    "exit_date": t.get("date"),
                    "total_pnl_dollar": 0.0,
                    "total_notional": 0.0,
                    "n_exits": 0,
                    "sample_trade": t,
                    "is_closed": True,
                }
            pos = positions[boid]
            pnl_d = float(t.get("pnl_dollar") or 0)
            notional = abs(float(t.get("notional_usd") or
                                (t.get("entry_price", 0) or 0) * (t.get("shares", 0) or 0) or 0))
            pos["total_pnl_dollar"] += pnl_d
            pos["total_notional"] += notional
            pos["n_exits"] += 1
            if t.get("date") and (pos["exit_date"] is None or t["date"] > pos["exit_date"]):
                pos["exit_date"] = t["date"]
            if t.get("rsi_at_entry") or t.get("confluence"):
                pos["sample_trade"] = t
        result = []
        for pos in positions.values():
            if pos["total_notional"] > 0:
                pos["total_pnl_pct"] = (pos["total_pnl_dollar"] / pos["total_notional"]) * 100
            else:
                pos["total_pnl_pct"] = 0.0
            entry = pos.get("entry_date") or pos.get("exit_date")
            pos["weight"] = self.calc_weight(entry) if entry else 0.5
            result.append(pos)
        return result

    async def evaluate_past_decisions(self, lookback_days: int = 90) -> list:
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)
        decisions = await self._col_decisions().find({
            "agent": self.name,
            "outcome": {"$ne": None},
            "created_at": {"$gte": cutoff},
        }).sort("created_at", -1).to_list(500)
        for d in decisions:
            d["_id"] = str(d["_id"])
            d["weight"] = self.calc_weight(d.get("created_at"))
        return decisions

    async def get_recent_decisions(self, limit: int = 20) -> list:
        decisions = await self._col_decisions().find(
            {"agent": self.name}
        ).sort("created_at", -1).to_list(limit)
        for d in decisions:
            d["_id"] = str(d["_id"])
        return decisions

    async def record_outcome(self, decision_id: str, outcome: dict):
        from bson import ObjectId
        await self._col_decisions().update_one(
            {"_id": ObjectId(decision_id)},
            {"$set": {
                "outcome": outcome,
                "outcome_date": datetime.utcnow(),
            }}
        )

    async def save_performance(self, metrics: dict):
        doc = {
            "agent": self.name,
            "metrics": metrics,
            "created_at": datetime.utcnow(),
        }
        await self._col_performance().insert_one(doc)

    async def get_performance_history(self, limit: int = 30) -> list:
        docs = await self._col_performance().find(
            {"agent": self.name}
        ).sort("created_at", -1).to_list(limit)
        for d in docs:
            d["_id"] = str(d["_id"])
        return docs

    async def analyze(self, context: dict) -> dict:
        raise NotImplementedError

    async def learn(self) -> dict:
        raise NotImplementedError
