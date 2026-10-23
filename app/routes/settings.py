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
    if "max_per_sector" in data:
        alpha_update["max_per_sector"] = data["max_per_sector"]
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
            "max_per_sector": settings_data.get("max_per_sector", 2),
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


CONFIGURATION_GROUPS = {
    "alpha": "Alpha",
    "max": "Max Strategy",
    "trend": "Trend Leadership",
    "risk": "Rischio e capitale",
    "apm": "Gestione posizioni",
}


def _configuration_registry(agents):
    registry = {}
    risk = "risk_manager"
    alpha = "alpha_strategist"
    executor = "executor"
    apm = "adaptive_position_manager"
    defaults = {name: agent.default_params() for name, agent in agents.items()}

    def add(key, label, group, kind, minimum=None, maximum=None, owners=None, choices=None):
        targets = owners or [(risk, key)]
        required_targets = targets[:1] if key in ("max_positions", "fractionable_only") else targets
        supported = all(name in defaults and field in defaults[name] for name, field in required_targets)
        registry[key] = {
            "label": label,
            "group": group,
            "type": kind,
            "minimum": minimum,
            "maximum": maximum,
            "choices": choices,
            "targets": targets,
            "editable": supported,
            "reason": None if supported else "Parametro non dichiarato dagli agenti responsabili: modifica non abilitata.",
            "owner": targets[0][0],
            "mirrors": [{"agent": name, "parameter": field} for name, field in targets[1:]],
        }

    def risk_key(key, alias=None):
        if key in defaults.get(risk, {}):
            return key
        if alias and alias in defaults.get(risk, {}):
            return alias
        return key

    add("max_positions", "Posizioni massime complessive", "risk", "integer", 1, 50,
        [(risk, "max_positions"), (alpha, "max_positions")])
    add("max_per_sector", "Posizioni massime per settore", "risk", "integer", 1, 20,
        [(risk, "max_per_sector"), (alpha, "max_per_sector")])
    for key, label, lo, hi in (
        ("risk_pct_per_trade", "Rischio per operazione (%)", 0.1, 10),
        ("max_position_pct", "Peso massimo per posizione (%)", 1, 100),
        ("position_size_pct", "Dimensione base posizione (%)", 0.1, 100),
        ("min_risk_reward", "R/R minimo Alpha", 0, 10),
        ("min_notional_per_trade", "Ordine minimo ($)", 1, 100000),
        ("dps_min_multiplier", "DPS moltiplicatore minimo", 0.1, 3),
        ("dps_max_multiplier", "DPS moltiplicatore massimo", 0.1, 5),
        ("dps_aggressiveness", "DPS aggressivita", 0, 3),
        ("kelly_fractional_factor", "Frazione Kelly", 0.01, 1),
    ):
        add(key, label, "risk", "number", lo, hi)
    for key, alias, label, lo, hi in (
        ("daily_loss_limit_pct", "daily_loss_limit", "Limite perdita giornaliera (%)", -50, -0.1),
        ("weekly_loss_limit_pct", "weekly_loss_limit", "Limite perdita settimanale (%)", -80, -0.1),
        ("min_cash_reserve_pct", "min_cash_reserve", "Riserva minima cassa (%)", 0, 90),
    ):
        add(key, label, "risk", "number", lo, hi, [(risk, risk_key(key, alias))])
    for key, label in (("dps_enabled", "Sizing DPS"), ("kelly_enabled", "Kelly")):
        add(key, label, "risk", "boolean")
    add("fractionable_only", "Solo titoli frazionabili", "risk", "boolean", owners=[(risk, "fractionable_only"), (alpha, "fractionable_only")])
    add("position_sizing_mode", "Modalita sizing", "risk", "choice", owners=[(risk, "position_sizing_mode"), (executor, "position_sizing_mode")], choices=["notional", "shares"])
    add("kelly_min_trades", "Campione minimo Kelly", "risk", "integer", 1, 1000)
    for channel, title in (("max", "Max"), ("trend", "Trend")):
        add(f"{channel}_live_enabled", f"Canale {title} attivo", channel, "boolean", owners=[(alpha, f"{channel}_live_enabled"), (risk, f"{channel}_live_enabled")])
        add(f"{channel}_live_slots", f"Slot {title}", channel, "integer", 1, 20, [(alpha, f"{channel}_live_slots"), (risk, f"{channel}_live_slots")])
        add(f"{channel}_min_risk_reward", f"R/R minimo {title}", channel, "number", 0, 10)
    add("max_stop_cap_pct", "Stop massimo Max (%)", "max", "number", 1, 30, [(alpha, "max_stop_cap_pct")])
    add("max_target_atr_mult", "Target Max (ATR)", "max", "number", 0.5, 10, [(alpha, "max_target_atr_mult")])
    add("trend_max_from_high_pct", "Distanza massima dai massimi (%)", "trend", "number", 1, 40, [(alpha, "trend_max_from_high_pct")])
    add("trend_max_rsi", "RSI massimo Trend", "trend", "number", 40, 95, [(alpha, "trend_max_rsi")])
    add("trend_rs_lookback", "Lookback forza relativa (sedute)", "trend", "integer", 20, 126, [(alpha, "trend_rs_lookback")])
    add("min_confluence", "Confluence minima Alpha", "alpha", "number", 0, 100, [(alpha, "min_confluence")])
    add("max_rsi_entry", "RSI massimo Alpha", "alpha", "number", 30, 95, [(alpha, "max_rsi_entry")])
    for key, label, kind, lo, hi in (
        ("apm_enabled", "APM attivo", "boolean", None, None),
        ("apm_scaling_enabled", "Scale-out attivo", "boolean", None, None),
        ("apm_exit_confluence_threshold", "Soglia confluence uscita", "integer", 0, 100),
        ("apm_exit_ml_threshold", "Soglia ML uscita", "integer", 0, 100),
        ("apm_exit_min_negative_factors", "Fattori negativi minimi", "integer", 1, 10),
        ("apm_min_holding_hours", "Holding minimo (ore)", "number", 0, 720),
        ("apm_check_interval_hours", "Intervallo controlli (ore)", "integer", 1, 24),
        ("apm_urgent_check_drop_pct", "Calo per controllo urgente (%)", "number", 0.1, 30),
        ("apm_tighten_profit_threshold", "Profitto minimo per stringere stop (%)", "number", 0, 50),
        ("apm_tighten_new_sl_distance", "Distanza stop stretto (%)", "number", 0.1, 30),
    ):
        add(key, label, "apm", kind, lo, hi, [(apm, key)])
    for level in (1, 2, 3):
        add(f"apm_target_{level}_pct", f"Target T{level} di fallback (%)", "apm", "number", 0.1, 100, [(apm, f"apm_target_{level}_pct")])
        add(f"apm_target_{level}_size", f"Vendita T{level} sul residuo (%)", "apm", "integer", 0, 100, [(apm, f"apm_target_{level}_size")])
    return registry, defaults


async def _configuration_snapshot(db, session=None):
    from app.routes.agents import _get_orch
    agents = _get_orch().agents
    registry, defaults = _configuration_registry(agents)
    effective = {}
    persisted = {}
    for name, base in defaults.items():
        doc = await db[f"agent_memory_{name}"].find_one({"_id": "params"}, session=session) or {}
        persisted[name] = doc
        effective[name] = {**base, **{key: value for key, value in doc.items() if key != "_id"}}
    fields = {}
    for key, spec in registry.items():
        values = []
        for name, field in spec["targets"]:
            values.append({
                "agent": name,
                "parameter": field,
                "value": effective.get(name, {}).get(field),
                "source": "database" if field in persisted.get(name, {}) else "default",
            })
        fields[key] = {
            **spec,
            "value": values[0]["value"],
            "aligned": all(item["value"] == values[0]["value"] for item in values),
            "sources": values,
        }
    return {"groups": CONFIGURATION_GROUPS, "fields": fields, "warnings": _configuration_warnings(fields)}


def _configuration_warnings(fields):
    warnings = []
    for key, field in fields.items():
        if not field["aligned"]:
            warnings.append(f"Parametro disallineato tra agenti: {key}")
    try:
        slots = sum(
            int(fields[f"{channel}_live_slots"]["value"] or 0)
            for channel in ("max", "trend")
            if any(bool(source["value"]) for source in fields[f"{channel}_live_enabled"]["sources"])
        )
        maximum = int(fields["max_positions"]["value"] or 0)
        if slots > maximum:
            warnings.append(f"Slot riservati {slots} superiori al totale {maximum}: Alpha non ha posti disponibili.")
        elif slots == maximum:
            warnings.append("Tutti gli slot sono riservati: Alpha non ha posti disponibili.")
    except (TypeError, ValueError):
        warnings.append("Distribuzione slot non calcolabile: verificare i parametri salvati.")
    return warnings


def _validate_configuration(changes, snapshot):
    import math
    from fastapi import HTTPException
    fields = snapshot["fields"]
    errors = {}
    clean = {}
    for key, value in changes.items():
        spec = fields.get(key)
        if not spec or not spec["editable"]:
            errors[key] = "Parametro sconosciuto o non modificabile."
            continue
        kind = spec["type"]
        if kind == "boolean":
            valid = type(value) is bool
        elif kind == "choice":
            valid = isinstance(value, str) and value in spec["choices"]
        else:
            valid = type(value) in (int, float) and math.isfinite(value)
            if valid and kind == "integer":
                valid = float(value).is_integer()
            if valid:
                valid = spec["minimum"] <= value <= spec["maximum"]
        if not valid:
            errors[key] = "Tipo o valore non valido per questo parametro."
        else:
            clean[key] = int(value) if kind == "integer" else value
    merged = {key: spec["value"] for key, spec in fields.items()}
    merged.update(clean)
    slot_keys = {"max_positions", "max_live_enabled", "max_live_slots", "trend_live_enabled", "trend_live_slots"}
    if slot_keys.intersection(clean):
        try:
            reserved = sum(int(merged[f"{channel}_live_slots"] or 0) for channel in ("max", "trend") if merged[f"{channel}_live_enabled"])
            if reserved > int(merged["max_positions"]):
                errors["slots"] = "La somma degli slot dei canali attivi non puo superare max_positions."
        except (TypeError, ValueError):
            errors["slots"] = "Distribuzione slot non valida."
    if {"dps_min_multiplier", "dps_max_multiplier"}.intersection(clean):
        if merged["dps_min_multiplier"] > merged["dps_max_multiplier"]:
            errors["dps"] = "Il moltiplicatore minimo non puo superare il massimo."
    target_keys = {f"apm_target_{level}_pct" for level in (1, 2, 3)}
    if target_keys.intersection(clean):
        if not merged["apm_target_1_pct"] < merged["apm_target_2_pct"] < merged["apm_target_3_pct"]:
            errors["apm_targets"] = "I target devono essere crescenti: T1 < T2 < T3."
    if errors:
        raise HTTPException(status_code=422, detail={"errors": errors})
    return clean


@router.get("/configuration")
async def get_configuration():
    return await _configuration_snapshot(get_db())


@router.post("/configuration")
async def update_configuration(payload: dict):
    from datetime import datetime, timezone
    from fastapi import HTTPException
    changes = payload.get("changes")
    if not isinstance(changes, dict) or not changes:
        raise HTTPException(status_code=422, detail="Inviare changes come oggetto non vuoto.")
    if set(payload) - {"changes", "expected", "dry_run"}:
        raise HTTPException(status_code=422, detail="Campi del payload non riconosciuti.")
    if type(payload.get("dry_run", False)) is not bool:
        raise HTTPException(status_code=422, detail="dry_run deve essere booleano.")
    expected = payload.get("expected")
    if not isinstance(expected, dict) or set(expected) != set(changes):
        raise HTTPException(status_code=422, detail="expected deve contenere gli stessi parametri di changes, con le sources lette dal GET.")
    db = get_db()

    async def prepare(session=None):
        snapshot = await _configuration_snapshot(db, session=session)
        clean = _validate_configuration(changes, snapshot)
        conflicts = [key for key in clean if expected[key] != snapshot["fields"][key]["sources"]]
        if conflicts:
            raise HTTPException(status_code=409, detail={"message": "Configurazione cambiata: rileggere prima di salvare.", "fields": conflicts})
        return snapshot, clean

    if payload.get("dry_run", False):
        snapshot, clean = await prepare()
        return {"status": "validated", "changes": clean, "warnings": snapshot["warnings"], "saved": False}
    try:
        async with await db.client.start_session() as session:
            async with session.start_transaction():
                snapshot, clean = await prepare(session=session)
                updates = {}
                for key, value in clean.items():
                    for name, field in snapshot["fields"][key]["targets"]:
                        updates.setdefault(name, {})[field] = value
                if "max_positions" in clean:
                    updates.setdefault("alpha_strategist", {})["max_candidates"] = clean["max_positions"] * 2
                for name, values in updates.items():
                    await db[f"agent_memory_{name}"].update_one({"_id": "params"}, {"$set": values}, upsert=True, session=session)
                legacy = {key: value for key, value in clean.items() if key in SettingsModel.model_fields}
                if legacy:
                    await db.app_settings.update_one({"_id": "risk_params"}, {"$set": legacy}, upsert=True, session=session)
                await db.configuration_audit.insert_one({
                    "date": datetime.now(timezone.utc),
                    "changes": clean,
                    "previous_sources": {key: snapshot["fields"][key]["sources"] for key in clean},
                    "source": "settings_configuration",
                }, session=session)
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=503, detail="Salvataggio transazionale non completato. Rileggere la configurazione prima di riprovare.") from error
    return {"status": "saved", "changes": clean, "configuration": await _configuration_snapshot(db)}
