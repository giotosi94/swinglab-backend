from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from app.db.mongodb import get_db

router = APIRouter()


class SettingsModel(BaseModel):
    """
    🔧 v2.2 — Extended with APM (Adaptive Position Manager) settings.
    Fonte capitale = Alpaca (endpoint /api/data/starting-capital).
    """
    # ===== RISK MANAGEMENT =====
    max_positions: int = 8
    risk_pct_per_trade: float = 2.0
    max_position_pct: float = 20.0
    min_risk_reward: float = 1.5
    max_per_sector: int = 2
    daily_loss_limit_pct: float = -3.0
    weekly_loss_limit_pct: float = -5.0
    
    # ===== FRACTIONAL / NOTIONAL TRADING =====
    position_sizing_mode: str = "notional"
    position_size_pct: float = 12.0
    fractionable_only: bool = True
    min_notional_per_trade: float = 100.0
    
    # ===== 🆕 v4.0 — APM (Adaptive Position Manager) =====
    apm_enabled: bool = True
    
    # EXIT thresholds
    apm_exit_confluence_threshold: int = 30
    apm_exit_ml_threshold: int = 40
    apm_exit_min_negative_factors: int = 2
    
    # SCALE OUT targets
    apm_scaling_enabled: bool = True
    apm_target_1_pct: float = 5.0
    apm_target_1_size: int = 50
    apm_target_2_pct: float = 10.0
    apm_target_2_size: int = 30
    apm_target_3_pct: float = 20.0
    apm_target_3_size: int = 20
    
    # TIGHTEN STOP
    apm_tighten_profit_threshold: float = 3.0
    apm_tighten_new_sl_distance: float = 2.0
    
    # FREQUENCY
    apm_check_interval_hours: int = 3
    apm_urgent_check_drop_pct: float = 5.0


@router.get("/")
async def get_settings():
    """
    Ritorna settings.
    Il capitale iniziale viene esposto dall'endpoint /api/data/starting-capital.
    """
    db = get_db()
    doc = await db.app_settings.find_one({"_id": "risk_params"})
    if doc:
        doc["_id"] = str(doc["_id"])
        # Rimuove starting_capital residuo (legacy)
        doc.pop("starting_capital", None)
        return doc
    return SettingsModel().dict()


@router.post("/")
async def save_settings(s: SettingsModel):
    """
    Save settings + propagazione ai 5 agenti (incluso APM).
    """
    db = get_db()
    data = s.dict()
    
    # 1. Salva settings principali
    await db.app_settings.update_one(
        {"_id": "risk_params"},
        {"$set": data},
        upsert=True,
    )
    
    # Cleanup starting_capital residuo (legacy)
    await db.app_settings.update_one(
        {"_id": "risk_params"},
        {"$unset": {"starting_capital": ""}},
    )
    
    # 2. RiskManager
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
            "position_sizing_mode": s.position_sizing_mode,
            "position_size_pct": s.position_size_pct,
            "fractionable_only": s.fractionable_only,
            "min_notional_per_trade": s.min_notional_per_trade,
        }},
        upsert=True,
    )
    
    # 3. AlphaStrategist
    await db.agent_memory_alpha_strategist.update_one(
        {"_id": "params"},
        {"$set": {
            "max_candidates": s.max_positions * 2,
            "max_positions": s.max_positions,
            "fractionable_only": s.fractionable_only,
        }},
        upsert=True,
    )
    
    # 4. Executor
    await db.agent_memory_executor.update_one(
        {"_id": "params"},
        {"$set": {
            "max_positions": s.max_positions,
            "position_sizing_mode": s.position_sizing_mode,
        }},
        upsert=True,
    )
    
    # 5. 🧹 MacroAnalyst — cleanup vecchio starting_capital
    await db.agent_memory_macro_analyst.update_one(
        {"_id": "params"},
        {"$unset": {"starting_capital": ""}},
    )
    
    # 6. 🆕 v4.0 — APM (Adaptive Position Manager)
    await db.agent_memory_adaptive_position_manager.update_one(
        {"_id": "params"},
        {"$set": {
            "apm_enabled": s.apm_enabled,
            # EXIT
            "apm_exit_confluence_threshold": s.apm_exit_confluence_threshold,
            "apm_exit_ml_threshold": s.apm_exit_ml_threshold,
            "apm_exit_min_negative_factors": s.apm_exit_min_negative_factors,
            # SCALE OUT
            "apm_scaling_enabled": s.apm_scaling_enabled,
            "apm_target_1_pct": s.apm_target_1_pct,
            "apm_target_1_size": s.apm_target_1_size,
            "apm_target_2_pct": s.apm_target_2_pct,
            "apm_target_2_size": s.apm_target_2_size,
            "apm_target_3_pct": s.apm_target_3_pct,
            "apm_target_3_size": s.apm_target_3_size,
            # TIGHTEN
            "apm_tighten_profit_threshold": s.apm_tighten_profit_threshold,
            "apm_tighten_new_sl_distance": s.apm_tighten_new_sl_distance,
            # FREQUENCY
            "apm_check_interval_hours": s.apm_check_interval_hours,
            "apm_urgent_check_drop_pct": s.apm_urgent_check_drop_pct,
        }},
        upsert=True,
    )
    
    print(
        f"✅ Settings propagated to 5 agents: "
        f"max_pos={s.max_positions}, "
        f"risk={s.risk_pct_per_trade}%, "
        f"sizing_mode={s.position_sizing_mode}, "
        f"pos_size={s.position_size_pct}%, "
        f"fractional={s.fractionable_only}, "
        f"APM={s.apm_enabled} (every {s.apm_check_interval_hours}h)"
    )
    
    return {
        "message": "Settings saved & propagated to all agents (capital from Alpaca)",
        "settings": data,
        "capital_source": "alpaca",
        "apm_enabled": s.apm_enabled,
    }



# ============================================
# v4.3 — RISK PROFILE PRESETS
# ============================================

RISK_PRESETS = {
    "conservative": {
        "name": "Conservativo",
        "emoji": "🛡️",
        "description": "Bassa esposizione, alta protezione. Ideale per iniziare.",
        "expected_return": "+8-12% annuo",
        "max_drawdown": "< 5%",
        "settings": {
            "max_positions": 5,
            "max_position_pct": 15.0,
            "position_size_pct": 8.0,
            "risk_pct_per_trade": 1.0,
            "min_risk_reward": 2.0,
            "max_per_sector": 1,
            "daily_loss_limit_pct": -2.0,
            "weekly_loss_limit_pct": -3.0,
            "min_cash_reserve_pct": 20.0,
            "dps_enabled": True,
            "dps_max_multiplier": 1.2,
            "dps_min_multiplier": 0.6,
            "dps_aggressiveness": 0.7,
            "kelly_enabled": False,
            "apm_exit_confluence_threshold": 40,
            "apm_exit_ml_threshold": 45,
            "apm_target_1_pct": 4.0,
            "apm_target_2_pct": 7.0,
            "apm_target_3_pct": 12.0,
            "apm_check_interval_hours": 2,
        }
    },
    "moderate": {
        "name": "Moderato",
        "emoji": "🎯",
        "description": "Bilanciato tra rischio e rendimento.",
        "expected_return": "+15-25% annuo",
        "max_drawdown": "5-10%",
        "settings": {
            "max_positions": 8,
            "max_position_pct": 20.0,
            "position_size_pct": 12.0,
            "risk_pct_per_trade": 2.0,
            "min_risk_reward": 1.5,
            "max_per_sector": 2,
            "daily_loss_limit_pct": -3.0,
            "weekly_loss_limit_pct": -5.0,
            "min_cash_reserve_pct": 10.0,
            "dps_enabled": True,
            "dps_max_multiplier": 1.4,
            "dps_min_multiplier": 0.6,
            "dps_aggressiveness": 1.0,
            "kelly_enabled": True,
            "kelly_fractional_factor": 0.20,
            "apm_exit_confluence_threshold": 30,
            "apm_exit_ml_threshold": 40,
            "apm_target_1_pct": 5.0,
            "apm_target_2_pct": 10.0,
            "apm_target_3_pct": 20.0,
            "apm_check_interval_hours": 1,
        }
    },
    "aggressive": {
        "name": "Aggressivo",
        "emoji": "⚡",
        "description": "Alta esposizione. Per investitori esperti.",
        "expected_return": "+25-40% annuo",
        "max_drawdown": "10-15%",
        "settings": {
            "max_positions": 12,
            "max_position_pct": 25.0,
            "position_size_pct": 18.0,
            "risk_pct_per_trade": 3.0,
            "min_risk_reward": 1.3,
            "max_per_sector": 3,
            "daily_loss_limit_pct": -5.0,
            "weekly_loss_limit_pct": -8.0,
            "min_cash_reserve_pct": 5.0,
            "dps_enabled": True,
            "dps_max_multiplier": 1.6,
            "dps_min_multiplier": 0.5,
            "dps_aggressiveness": 1.3,
            "kelly_enabled": True,
            "kelly_fractional_factor": 0.25,
            "apm_exit_confluence_threshold": 25,
            "apm_exit_ml_threshold": 35,
            "apm_target_1_pct": 6.0,
            "apm_target_2_pct": 12.0,
            "apm_target_3_pct": 25.0,
            "apm_check_interval_hours": 1,
        }
    },
    "super_aggressive": {
        "name": "Super Aggressivo",
        "emoji": "🚀",
        "description": "Massima esposizione. Alta volatilita. Solo pro.",
        "expected_return": "+35-60% annuo",
        "max_drawdown": "> 15%",
        "settings": {
            "max_positions": 15,
            "max_position_pct": 30.0,
            "position_size_pct": 22.0,
            "risk_pct_per_trade": 4.0,
            "min_risk_reward": 1.2,
            "max_per_sector": 4,
            "daily_loss_limit_pct": -7.0,
            "weekly_loss_limit_pct": -12.0,
            "min_cash_reserve_pct": 3.0,
            "dps_enabled": True,
            "dps_max_multiplier": 1.8,
            "dps_min_multiplier": 0.4,
            "dps_aggressiveness": 1.6,
            "kelly_enabled": True,
            "kelly_fractional_factor": 0.35,
            "apm_exit_confluence_threshold": 20,
            "apm_exit_ml_threshold": 30,
            "apm_target_1_pct": 7.0,
            "apm_target_2_pct": 15.0,
            "apm_target_3_pct": 30.0,
            "apm_check_interval_hours": 1,
        }
    },
}


@router.get("/presets")
async def get_risk_presets():
    """v4.3 Ritorna tutti i 4 preset di rischio disponibili."""
    return {
        "presets": RISK_PRESETS,
        "current": await get_current_preset_name(),
    }


async def get_current_preset_name():
    """Determina quale preset e attualmente attivo (best match)."""
    db = get_db()
    current = await db.app_settings.find_one({"_id": "risk_params"})
    if not current:
        return None
    
    max_pos = current.get("max_positions", 8)
    if max_pos <= 5:
        return "conservative"
    elif max_pos <= 8:
        return "moderate"
    elif max_pos <= 12:
        return "aggressive"
    else:
        return "super_aggressive"


@router.post("/preset/{preset_name}")
async def apply_risk_preset(preset_name: str):
    """v4.3 Applica un preset di rischio (aggiorna tutte le settings)."""
    if preset_name not in RISK_PRESETS:
        return {"error": f"Preset '{preset_name}' non trovato", "available": list(RISK_PRESETS.keys())}
    
    preset = RISK_PRESETS[preset_name]
    settings_data = preset["settings"]
    
    db = get_db()
    
    # 1. Aggiorna DB app_settings
    await db.app_settings.update_one(
        {"_id": "risk_params"},
        {"$set": settings_data},
        upsert=True,
    )
    
    # 2. Propaga a tutti gli agenti
    await db.agent_memory_risk_manager.update_one(
        {"_id": "params"},
        {"$set": {k: v for k, v in settings_data.items() if k.startswith(("max_", "risk_", "min_risk", "position_", "daily_", "weekly_", "kelly_", "dps_"))}},
        upsert=True,
    )
    
    await db.agent_memory_alpha_strategist.update_one(
        {"_id": "params"},
        {"$set": {
            "max_candidates": settings_data.get("max_positions", 8) * 2,
            "max_positions": settings_data.get("max_positions", 8),
        }},
        upsert=True,
    )
    
    await db.agent_memory_executor.update_one(
        {"_id": "params"},
        {"$set": {
            "max_positions": settings_data.get("max_positions", 8),
        }},
        upsert=True,
    )
    
    await db.agent_memory_adaptive_position_manager.update_one(
        {"_id": "params"},
        {"$set": {k: v for k, v in settings_data.items() if k.startswith("apm_")}},
        upsert=True,
    )
    
    print(f"Applied risk preset: {preset_name}")
    
    return {
        "message": f"Preset {preset['name']} applicato con successo",
        "preset": preset_name,
        "settings_applied": len(settings_data),
        "description": preset["description"],
    }
