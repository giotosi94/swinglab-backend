from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from app.db.mongodb import get_db

router = APIRouter()


class SettingsModel(BaseModel):
    max_positions: int = 5
    risk_pct_per_trade: float = 2.0
    max_position_pct: float = 20.0
    min_risk_reward: float = 1.5
    max_per_sector: int = 2
    daily_loss_limit_pct: float = -3.0
    weekly_loss_limit_pct: float = -5.0
    starting_capital: float = 100000.0


@router.get("/")
async def get_settings():
    db = get_db()
    doc = await db.app_settings.find_one({"_id": "risk_params"})
    if doc:
        doc["_id"] = str(doc["_id"])
        return doc
    return SettingsModel().dict()


@router.post("/")
async def save_settings(s: SettingsModel):
    db = get_db()
    data = s.dict()
    await db.app_settings.update_one(
        {"_id": "risk_params"},
        {"$set": data},
        upsert=True,
    )
    await db.agent_memory_risk_manager.update_one(
        {"_id": "params"},
        {"$set": {
            "max_positions": s.max_positions,
            "risk_pct_per_trade": s.risk_pct_per_trade,
            "max_position_pct": s.max_position_pct,
            "min_risk_reward": s.min_risk_reward,
            "max_per_sector": s.max_per_sector,
            "daily_loss_limit_pct": s.daily_loss_limit_pct,
            "weekly_loss_limit_pct": s.weekly_loss_limit_pct,
        }},
        upsert=True,
    )
    return {"message": "Settings saved", "settings": data}
