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

    # 1. Salva settings principali
    await db.app_settings.update_one(
        {"_id": "risk_params"},
        {"$set": data},
        upsert=True,
    )

    # 2. RiskManager — tutti i parametri di rischio
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

    # ===== FIX 19F: Propaga a TUTTI gli agenti =====

    # 3. AlphaStrategist — max_positions influenza quanti candidati genera
    await db.agent_memory_alpha_strategist.update_one(
        {"_id": "params"},
        {"$set": {
            "max_candidates": s.max_positions * 2,
            "max_positions": s.max_positions,
        }},
        upsert=True,
    )

    # 4. Executor — max_positions per limitare ordini + risk params
    await db.agent_memory_executor.update_one(
        {"_id": "params"},
        {"$set": {
            "max_positions": s.max_positions,
        }},
        upsert=True,
    )

    # 5. MacroAnalyst — starting_capital per calcoli percentuali
    await db.agent_memory_macro_analyst.update_one(
        {"_id": "params"},
        {"$set": {
            "starting_capital": s.starting_capital,
        }},
        upsert=True,
    )

    print(f"✅ Settings propagated to all 4 agents: max_pos={s.max_positions}, risk={s.risk_pct_per_trade}%, capital=${s.starting_capital}")
    return {"message": "Settings saved & propagated to all agents", "settings": data}
