
from datetime import datetime, timedelta
from app.db.mongodb import get_db
import math


class BaseAgent:
    """
    Classe base per tutti gli agenti SwingLab.
    Ogni agente ha:
    - Un nome univoco (usato per le collection MongoDB)
    - Parametri appresi (memory) che si aggiornano nel tempo
    - Log di ogni decisione con contesto completo
    - Sistema di learning con time decay
    """

    def __init__(self, name: str, version: str = "1.0"):
        self.name = name
        self.version = version
        self.decay_days = 60  # Dopo 60 giorni un trade pesa 50%
        self.min_decisions_to_learn = 5  # Minimo decisioni per attivare learning

    def _col_memory(self):
        """Collection MongoDB per i parametri appresi"""
        return get_db()[f"agent_memory_{self.name}"]

    def _col_decisions(self):
        """Collection MongoDB per il log decisioni"""
        return get_db()[f"agent_decisions_{self.name}"]

    def _col_performance(self):
        """Collection MongoDB per le metriche di performance"""
        return get_db()[f"agent_performance_{self.name}"]

    # ---- MEMORY (parametri appresi) ----

    async def get_params(self) -> dict:
        """Recupera i parametri appresi dal DB"""
        doc = await self._col_memory().find_one({"_id": "params"})
        if doc:
            doc.pop("_id", None)
            return doc
        return self.default_params()

    async def save_params(self, params: dict):
        """Salva i parametri appresi nel DB"""
        params["updated_at"] = datetime.utcnow()
        params["agent_version"] = self.version
        await self._col_memory().update_one(
            {"_id": "params"}, {"$set": params}, upsert=True
        )

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
