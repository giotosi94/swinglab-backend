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

    v2.0 — get_params() ora fa il MERGE con default_params().
    v3.0 — Learning Machine core:
        - Param history append-only (agent_param_history_{name})
        - update_params_safe() con max relative delta
        - Helper get_position_trades() per aggregare scale-out/tranche
        - record_outcome e evaluate migliorati
    """

    def __init__(self, name: str, version: str = "1.0"):
        self.name = name
        self.version = version
        self.decay_days = 60  # Dopo 60 giorni un trade pesa 50%
        self.min_decisions_to_learn = 5  # Minimo decisioni per attivare learning
        # v3.0 — limiti di sicurezza per il learning
        self.max_param_delta_pct = 0.25  # max 25% di variazione relativa per ciclo
        self.max_history = 50  # quante versioni tenere in history

    def _col_memory(self):
        """Collection MongoDB per i parametri appresi (current)"""
        return get_db()[f"agent_memory_{self.name}"]

    def _col_param_history(self):
        """Collection MongoDB per lo storico dei set di parametri"""
        return get_db()[f"agent_param_history_{self.name}"]

    def _col_decisions(self):
        """Collection MongoDB per il log decisioni"""
        return get_db()[f"agent_decisions_{self.name}"]

    def _col_performance(self):
        """Collection MongoDB per le metriche di performance"""
        return get_db()[f"agent_performance_{self.name}"]

    # ---- MEMORY (parametri appresi) ----
    async def get_params(self) -> dict:
        """
        Recupera i parametri appresi dal DB, mergiati sui default.

        Ordine di precedenza:
        1. default_params()  → base sempre completa
        2. doc DB            → sovrascrive quello che e' stato salvato/appreso

        Cosi' una chiave assente nel DB usa il default dichiarato,
        non un fallback inline sparso nel codice.
        """
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
        """
        Parametri COSI' COME SONO nel DB, senza merge.
        Utile per audit: mostra quali chiavi sono davvero persistite.
        """
        doc = await self._col_memory().find_one({"_id": "params"})
        if doc:
            doc.pop("_id", None)
            return doc
        return {}

    async def save_params(self, params: dict, reason: str = "manual"):
        """
        Salva i parametri appresi nel DB + appende una versione in history.

        v3.0: ogni save crea un documento in agent_param_history_{name}
        con timestamp, reason e snapshot completo. Tiene solo gli ultimi
        max_history documenti.
        """
        params = dict(params)
        params.pop("_id", None)

        # Snapshot corrente prima di sovrascrivere (per history)
        current = await self.get_params_raw()

        now = datetime.utcnow()
        params["updated_at"] = now
        params["agent_version"] = self.version

        # Incrementa param_version
        prev_version = current.get("param_version", 0)
        new_version = int(prev_version) + 1
        params["param_version"] = new_version

        # Scrive current
        await self._col_memory().update_one(
            {"_id": "params"}, {"$set": params}, upsert=True
        )

        # Appende history
        history_doc = {
            "param_version": new_version,
            "agent_version": self.version,
            "reason": reason,
            "params": {k: v for k, v in params.items()
                       if k not in ("updated_at", "agent_version", "param_version")},
            "created_at": now,
        }
        await self._col_param_history().insert_one(history_doc)

        # Trim history (tieni solo le ultime max_history)
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
        """
        v3.0 — Aggiornamento sicuro dei parametri con clamp sul delta massimo.

        Per ogni chiave numerica (int/float) limita la variazione relativa
        a self.max_param_delta_pct (default 25%). Le chiavi non numeriche
        (liste, dict, stringhe) vengono accettate cosi' come sono.

        Ritorna il dict effettivamente applicato + report delle variazioni.
        """
        current = await self.get_params()
        applied = deepcopy(current)
        changes = {}

        for key, new_val in new_params.items():
            if key in ("updated_at", "agent_version", "param_version", "_id"):
                continue

            old_val = current.get(key)

            # Non numerici → accettati full
            if not isinstance(new_val, (int, float)) or not isinstance(old_val, (int, float)):
                if new_val != old_val:
                    applied[key] = new_val
                    changes[key] = {"from": old_val, "to": new_val, "clamped": False}
                continue

            # Numerici → clamp relativo
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

            # Mantieni il tipo originale se era int
            if isinstance(old_val, int) and not isinstance(old_val, bool):
                new_val = int(round(new_val))

            if new_val != old_val:
                applied[key] = new_val
                changes[key] = {
                    "from": old_val,
                    "to": new_val,
                    "clamped": clamped,
                }

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
        """Storico dei set di parametri (piu' recenti prima)."""
        docs = await self._col_param_history().find().sort(
            "created_at", -1
        ).to_list(limit)
        for d in docs:
            d["_id"] = str(d["_id"])
        return docs

    def default_params(self) -> dict:
        """Override in ogni agente: parametri di default"""
        return {}

    # ---- DECISION LOGGING ----
    async def log_decision(self, decision_type: str, data: dict,
                           reasoning: str = "", confidence: float = 50.0):
        """
        Logga una decisione con contesto completo.

        Ogni decisione viene salvata con:
        - Tipo (es. "regime_change", "buy_signal", "risk_reject")
        - Dati completi del contesto
        - Reasoning (perche' ha preso questa decisione)
        - Confidence (0-100)
        - Timestamp per il time decay
        """
        doc = {
            "agent": self.name,
            "type": decision_type,
            "data": data,
            "reasoning": reasoning,
            "confidence": confidence,
            "outcome": None,  # Verra' aggiornato dopo dal learning
            "outcome_date": None,
            "created_at": datetime.utcnow(),
        }
        result = await self._col_decisions().insert_one(doc)
        return str(result.inserted_id)

    # ---- TIME DECAY ----
    def calc_weight(self, created_at: datetime) -> float:
        """
        Calcola il peso di una decisione in base a quanto e' vecchia.

        Exponential decay: peso = e^(-lambda * days)
        dove lambda = ln(2) / decay_days

        Dopo decay_days (60), il peso e' 0.5
        Dopo 120 giorni, il peso e' 0.25
        """
        if not created_at:
            return 0.1
        days_old = (datetime.utcnow() - created_at).days
        if days_old <= 0:
            return 1.0
        lam = math.log(2) / self.decay_days
        return math.exp(-lam * days_old)

    # ---- POSITION-CENTRIC HELPERS (v3.0) ----
    async def get_position_trades(self, lookback_days: int = 180) -> list:
        """
        Aggrega i trade per POSIZIONE (buy_order_id), non per singolo sell.

        Risolve il bias dei scale-out / tranche: una posizione che fa
        T1 + T2 + EXIT diventa UN solo sample con PnL totale e outcome netto.

        Ritorna lista di dict:
        {
            "buy_order_id": str,
            "ticker": str,
            "entry_date": datetime,
            "exit_date": datetime | None,
            "total_pnl_pct": float,
            "total_pnl_dollar": float,
            "is_closed": bool,
            "n_exits": int,
            "sample_trade": dict,
            "weight": float,
        }
        """
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

            # Preferisci un trade che ha feature di entry
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

    # ---- LEARNING ----
    async def evaluate_past_decisions(self, lookback_days: int = 90) -> list:
        """
        Recupera le decisioni passate con outcome gia' registrato.
        Applica il time decay per pesare le decisioni recenti di piu'.
        """
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
        """Ultime N decisioni (per debug/monitoring)"""
        decisions = await self._col_decisions().find(
            {"agent": self.name}
        ).sort("created_at", -1).to_list(limit)
        for d in decisions:
            d["_id"] = str(d["_id"])
        return decisions

    async def record_outcome(self, decision_id: str, outcome: dict):
        """Registra l'outcome di una decisione passata (per il learning loop)"""
        from bson import ObjectId
        await self._col_decisions().update_one(
            {"_id": ObjectId(decision_id)},
            {"$set": {
                "outcome": outcome,
                "outcome_date": datetime.utcnow(),
            }}
        )

    # ---- PERFORMANCE TRACKING ----
    async def save_performance(self, metrics: dict):
        """Salva snapshot delle performance dell'agente"""
        doc = {
            "agent": self.name,
            "metrics": metrics,
            "created_at": datetime.utcnow(),
        }
        await self._col_performance().insert_one(doc)

    async def get_performance_history(self, limit: int = 30) -> list:
        """Storico performance per grafici"""
        docs = await self._col_performance().find(
            {"agent": self.name}
        ).sort("created_at", -1).to_list(limit)
        for d in docs:
            d["_id"] = str(d["_id"])
        return docs

    # ---- ABSTRACT METHODS (override in ogni agente) ----
    async def analyze(self, context: dict) -> dict:
        """Metodo principale: analizza e produce output. Override obbligatorio."""
        raise NotImplementedError

    async def learn(self) -> dict:
        """Ciclo di apprendimento: analizza risultati e aggiusta parametri. Override obbligatorio."""
        raise NotImplementedError
