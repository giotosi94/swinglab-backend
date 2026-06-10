"""
SwingLab Shared Brain
Central state that all agents read/write independently.
Stored in MongoDB collection 'shared_brain'.
"""

from datetime import datetime
from app.db.mongodb import get_db


class SharedBrain:
    """Shared state between all agents via MongoDB."""

    COLLECTION = "shared_brain"
    DOC_ID = "current"

    # ============================================
    # READ METHODS
    # ============================================

    async def get_market(self):
        """Read market state (written by MacroAnalyst)."""
        db = get_db()
        doc = await db[self.COLLECTION].find_one({"_id": self.DOC_ID})
        if doc and "market" in doc:
            return doc["market"]
        return {}

    async def get_candidates(self):
        """Read buy/sell candidates (written by AlphaStrategist)."""
        db = get_db()
        doc = await db[self.COLLECTION].find_one({"_id": self.DOC_ID})
        if doc and "candidates" in doc:
            return doc["candidates"]
        return {"buy": [], "sell": []}

    async def get_approved(self):
        """Read approved trades (written by RiskManager)."""
        db = get_db()
        doc = await db[self.COLLECTION].find_one({"_id": self.DOC_ID})
        if doc and "approved" in doc:
            return doc["approved"]
        return {"trades": [], "sells": []}

    async def get_executions(self):
        """Read execution log (written by Executor)."""
        db = get_db()
        doc = await db[self.COLLECTION].find_one({"_id": self.DOC_ID})
        if doc and "executions" in doc:
            return doc["executions"]
        return {"last_buys": [], "last_sells": []}

    async def get_full_state(self):
        """Read the entire shared brain state."""
        db = get_db()
        doc = await db[self.COLLECTION].find_one({"_id": self.DOC_ID})
        if doc:
            doc["_id"] = str(doc["_id"])
            return doc
        return {}

    # ============================================
    # WRITE METHODS
    # ============================================

    async def write_market(self, market_data):
        """MacroAnalyst writes market state."""
        db = get_db()
        await db[self.COLLECTION].update_one(
            {"_id": self.DOC_ID},
            {"$set": {
                "market": {
                    **market_data,
                    "updated_at": datetime.utcnow().isoformat(),
                },
            }},
            upsert=True,
        )

    async def write_candidates(self, buy_candidates, sell_signals):
        """AlphaStrategist writes candidates."""
        db = get_db()
        await db[self.COLLECTION].update_one(
            {"_id": self.DOC_ID},
            {"$set": {
                "candidates": {
                    "buy": buy_candidates,
                    "sell": sell_signals,
                    "updated_at": datetime.utcnow().isoformat(),
                },
            }},
            upsert=True,
        )

    async def write_approved(self, approved_trades, approved_sells, risk_report):
        """RiskManager writes approved trades."""
        db = get_db()
        await db[self.COLLECTION].update_one(
            {"_id": self.DOC_ID},
            {"$set": {
                "approved": {
                    "trades": approved_trades,
                    "sells": approved_sells,
                    "risk_report": risk_report,
                    "updated_at": datetime.utcnow().isoformat(),
                },
            }},
            upsert=True,
        )

    async def write_executions(self, executed_buys, executed_sells, details=None):
        """Executor writes execution results."""
        db = get_db()
        await db[self.COLLECTION].update_one(
            {"_id": self.DOC_ID},
            {"$set": {
                "executions": {
                    "last_buys": executed_buys,
                    "last_sells": executed_sells,
                    "details": details or {},
                    "updated_at": datetime.utcnow().isoformat(),
                },
            }},
            upsert=True,
        )

    async def clear_approved(self):
        """Executor clears approved trades after execution."""
        db = get_db()
        await db[self.COLLECTION].update_one(
            {"_id": self.DOC_ID},
            {"$set": {
                "approved.trades": [],
                "approved.sells": [],
                "approved.updated_at": datetime.utcnow().isoformat(),
            }},
        )

    # ============================================
    # UTILITY
    # ============================================

    async def is_stale(self, section, max_minutes=60):
        """Check if a section is stale (not updated recently)."""
        db = get_db()
        doc = await db[self.COLLECTION].find_one({"_id": self.DOC_ID})
        if not doc or section not in doc:
            return True
        updated = doc[section].get("updated_at", "")
        if not updated:
            return True
        try:
            updated_dt = datetime.fromisoformat(updated)
            age_minutes = (datetime.utcnow() - updated_dt).total_seconds() / 60
            return age_minutes > max_minutes
        except:
            return True


# ---- Singleton ----
brain = SharedBrain()
