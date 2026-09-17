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
    max_positions: int = 12
    risk_pct_per_trade: float = 3.0
    max_position_pct: float = 25.0
    min_risk_reward: float = 1.3
    max_per_sector: int = 3
    daily_loss_limit_pct: float = -5.0
    weekly_loss_limit_pct: float = -8.0
    
    # ===== FRACTIONAL / NOTIONAL TRADING =====
    position_sizing_mode: str = "notional"
    position_size_pct: float = 18.0
    fractionable_only: bool = True
    min_notional_per_trade: float = 100.0
    min_cash_reserve_pct: float = 5.0
    dps_enabled: bool = True
    dps_max_multiplier: float = 1.6
    dps_min_multiplier: float = 0.5
    dps_aggressiveness: float = 1.3
    kelly_enabled: bool = True
    kelly_fractional_factor: float = 0.25
    kelly_min_trades: int = 20
    active_preset: Optional[str] = None
    
    # ===== 🆕 v4.0 — APM (Adaptive Position Manager) =====
    apm_enabled: bool = True
    
    # EXIT thresholds
    apm_exit_confluence_threshold: int = 25
    apm_exit_ml_threshold: int = 35
    apm_exit_min_negative_factors: int = 3
    
    # SCALE OUT targets
    apm_scaling_enabled: bool = True
    apm_target_1_pct: float = 6.0
    apm_target_1_size: int = 30
    apm_target_2_pct: float = 12.0
    apm_target_2_size: int = 30
    apm_target_3_pct: float = 25.0
    apm_target_3_size: int = 25
    
    # TIGHTEN STOP
    apm_tighten_profit_threshold: float = 3.0
    apm_tighten_new_sl_distance: float = 2.0
    
    # FREQUENCY
    apm_check_interval_hours: int = 1
    apm_urgent_check_drop_pct: float = 5.0


@router.get("/")
async def get_settings():
    """
    Ritorna settings.
    Il capitale iniziale viene esposto dall'endpoint /api/data/starting-capital.
    """
    db = get_db()
    doc = await db.app_settings.find_one({"_id": "risk_params"})
    defaults = SettingsModel().model_dump()
    if doc:
        doc.pop("_id", None)
        doc.pop("starting_capital", None)
        defaults.update(doc)
    return defaults


@router.post("/")
async def save_settings(payload: dict):
    db = get_db()
    allowed = set(SettingsModel.model_fields.keys())
    data = {k: v for k, v in payload.items() if k in allowed and v is not None}

    if not data:
        return {"message": "Nessun parametro modificato", "settings": {}}

    validated = SettingsModel(**{**SettingsModel().model_dump(), **data})
    clean = validated.model_dump()
    data = {k: clean[k] for k in data}

    await db.app_settings.update_one(
        {"_id": "risk_params"},
        {"$set": data, "$unset": {"starting_capital": ""}},
        upsert=True,
    )

    risk_keys = {
        "max_positions", "risk_pct_per_trade", "max_position_pct", "min_risk_reward",
        "max_per_sector", "daily_loss_limit_pct", "weekly_loss_limit_pct",
        "position_sizing_mode", "position_size_pct", "fractionable_only",
        "min_notional_per_trade", "min_cash_reserve_pct", "dps_enabled",
        "dps_max_multiplier", "dps_min_multiplier", "dps_aggressiveness",
        "kelly_enabled", "kelly_fractional_factor", "kelly_min_trades",
    }
    risk_update = {k: v for k, v in data.items() if k in risk_keys}
    if risk_update:
        await db.agent_memory_risk_manager.update_one(
            {"_id": "params"}, {"$set": risk_update}, upsert=True
        )

    alpha_update = {}
    if "max_positions" in data:
        alpha_update["max_positions"] = data["max_positions"]
        alpha_update["max_candidates"] = data["max_positions"] * 2
    if "fractionable_only" in data:
        alpha_update["fractionable_only"] = data["fractionable_only"]
    if alpha_update:
        await db.agent_memory_alpha_strategist.update_one(
            {"_id": "params"}, {"$set": alpha_update}, upsert=True
        )

    executor_update = {k: data[k] for k in ("max_positions", "position_sizing_mode") if k in data}
    if executor_update:
        await db.agent_memory_executor.update_one(
            {"_id": "params"}, {"$set": executor_update}, upsert=True
        )

    apm_update = {k: v for k, v in data.items() if k.startswith("apm_")}
    if apm_update:
        await db.agent_memory_adaptive_position_manager.update_one(
            {"_id": "params"}, {"$set": apm_update}, upsert=True
        )

    return {
        "message": "Settings salvate e propagate senza sovrascrivere gli altri parametri",
        "settings": data,
        "propagated": {
            "risk_manager": list(risk_update.keys()),
            "alpha_strategist": list(alpha_update.keys()),
            "executor": list(executor_update.keys()),
            "adaptive_position_manager": list(apm_update.keys()),
        },
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
    
    explicit = current.get("active_preset")
    if explicit in RISK_PRESETS:
        return explicit
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
    settings_data = dict(preset["settings"])
    settings_data["active_preset"] = preset_name
    
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
