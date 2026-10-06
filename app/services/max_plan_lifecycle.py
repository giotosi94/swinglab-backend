from datetime import datetime

import numpy as np
import pandas as pd

LIFECYCLE_VERSION = 1
TRACKED_STATUSES = ("ARMED", "TRIGGERED", "CONFIRMED_4H", "CONFIRMED_DAILY", "WAIT_RETEST")
CONFIRMED_STATUSES = ("CONFIRMED_4H", "CONFIRMED_DAILY")
TERMINAL_STATES = ("INVALIDATED", "EXPIRED", "CLOSED", "SUPERSEDED")
STATE_RANK = {"ARMED": 1, "TRIGGERED": 2, "WAIT_RETEST": 3, "CONFIRMED_4H": 3, "CONFIRMED_DAILY": 3}
MATCH_TOLERANCE_ATR = 0.5
EXPIRATION_BARS = 15
OBSERVATION_BARS = 20
MAX_TRIGGER_FAILURES = 1
MAX_EVENTS = 40
LEGACY_COHORTS = ("ACTIONABLE", "RETEST_WATCH", "TRIGGER_WATCH", "INVALID", "LEGACY_INVALID")


def _num(value):
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _qualification(max_strategy, plan, signal_price):
    trigger = _num(plan.get("trigger_price"))
    maximum_entry = _num(plan.get("maximum_entry_price"))
    invalidation = _num(plan.get("invalidation_price"))
    levels_valid = bool(
        signal_price
        and trigger
        and maximum_entry
        and trigger > 0
        and maximum_entry > trigger
        and (invalidation is None or 0 < invalidation < trigger)
    )
    already_invalidated = bool(invalidation and signal_price and signal_price <= invalidation)
    qualified = bool(
        max_strategy.get("data_eligible")
        and max_strategy.get("strategy_eligible")
        and plan.get("weekly_plan_qualified")
        and not plan.get("blocking_phase")
        and levels_valid
        and not already_invalidated
    )
    return qualified, trigger, maximum_entry, invalidation


def _cohort(state, order_action):
    if state == "WAIT_RETEST":
        return "RETEST_WATCH"
    if state == "TRIGGERED" and order_action != "BUY_ALLOWED":
        return "TRIGGER_WATCH"
    return "ACTIONABLE"


def _same_structure(active, plan, trigger, atr):
    tolerance = max(_num(active.get("atr14")) or atr or 0.0, 0.0) * MATCH_TOLERANCE_ATR
    source = plan.get("trigger_source")
    anchor = plan.get("structure_anchor") or {}
    active_anchor = active.get("structure_anchor") or {}
    if source and active.get("trigger_source") and source != active.get("trigger_source"):
        return False
    if source and source.startswith("TRENDLINE"):
        if anchor.get("anchor_1_date") and active_anchor.get("anchor_1_date"):
            return (
                anchor.get("anchor_1_date") == active_anchor.get("anchor_1_date")
                and anchor.get("anchor_2_date") == active_anchor.get("anchor_2_date")
            )
    if source and source.startswith("NECK"):
        neck = _num(anchor.get("neck_high")) or _num(anchor.get("neck_center"))
        active_neck = _num(active_anchor.get("neck_high")) or _num(active_anchor.get("neck_center"))
        if neck and active_neck:
            return abs(neck - active_neck) <= tolerance
    reference = _num(active.get("trigger_price_initial"))
    return bool(reference and trigger and abs(trigger - reference) <= tolerance)


def _bars_after(df, date_text):
    if not date_text:
        return 0
    return int((df["datetime"] > pd.to_datetime(date_text)).sum())


def _plan_outcomes(df, plan_doc):
    triggered_at = plan_doc.get("triggered_at")
    entry = _num(plan_doc.get("entry_reference_price"))
    empty = {
        "bars_observed": 0,
        "entry_triggered": bool(triggered_at),
        "entry_triggered_at": triggered_at,
        "mfe_pct": None,
        "mae_pct": None,
        "invalidation_reached": False,
        "maximum_entry_exceeded": False,
        "return_5d_pct": None,
        "return_10d_pct": None,
        "return_20d_pct": None,
    }
    if not triggered_at or not entry or entry <= 0:
        return empty
    post = df[df["datetime"] >= pd.to_datetime(triggered_at)].reset_index(drop=True)
    if len(post) == 0:
        return empty
    highest = float(post["High"].max())
    lowest = float(post["Low"].min())
    invalidation = _num(plan_doc.get("invalidation_price"))
    maximum_entry = _num(plan_doc.get("maximum_entry_price"))

    def forward(index):
        if len(post) > index:
            return round((float(post["Close"].iloc[index]) - entry) / entry * 100, 2)
        return None

    return {
        "bars_observed": len(post),
        "entry_triggered": True,
        "entry_triggered_at": triggered_at,
        "mfe_pct": round(max(0.0, (highest - entry) / entry * 100), 2),
        "mae_pct": round(min(0.0, (lowest - entry) / entry * 100), 2),
        "invalidation_reached": bool(invalidation and lowest <= invalidation),
        "maximum_entry_exceeded": bool(maximum_entry and highest > maximum_entry),
        "return_5d_pct": forward(4),
        "return_10d_pct": forward(9),
        "return_20d_pct": forward(19),
    }


def _event(event_type, signal_date, from_state, to_state, reason=None, price=None):
    return {
        "type": event_type,
        "date": signal_date,
        "from": from_state,
        "to": to_state,
        "reason": reason,
        "price": price,
        "at": datetime.utcnow(),
    }


async def _write_signal(db, plan_doc, max_strategy, state, signal_date, signal_price):
    plan_key = plan_doc["plan_key"]
    setup_key = f"{plan_key}:{state}"
    now = datetime.utcnow()
    document = {
        "setup_key": setup_key,
        "plan_key": plan_key,
        "plan_id": plan_key,
        "lifecycle_version": LIFECYCLE_VERSION,
        "ticker": plan_doc["ticker"],
        "signal_date": signal_date,
        "strategy_version": max_strategy.get("version"),
        "status": state,
        "order_action": plan_doc.get("order_action"),
        "signal_price": signal_price,
        "trigger_source": plan_doc.get("trigger_source"),
        "trigger_price": plan_doc.get("trigger_price"),
        "maximum_entry_price": plan_doc.get("maximum_entry_price"),
        "invalidation_price": plan_doc.get("invalidation_price"),
        "entry_reference_price": plan_doc.get("entry_reference_price"),
        "market_phase": (max_strategy.get("market_phase") or {}).get("phase"),
        "strategy_type": max_strategy.get("strategy_type"),
        "max_score": max_strategy.get("max_score"),
        "created_at": now,
    }
    await db.max_strategy_signals.update_one(
        {"setup_key": setup_key},
        {"$setOnInsert": document, "$set": {
            "validation_cohort": plan_doc.get("validation_cohort"),
            "latest_status": state,
            "last_seen_at": now,
            "updated_at": now,
        }},
        upsert=True,
    )


async def _close_plan(db, plan_doc, state, signal_date, reason, df, price):
    outcomes = _plan_outcomes(df, plan_doc)
    now = datetime.utcnow()
    await db.max_strategy_plans.update_one(
        {"plan_key": plan_doc["plan_key"]},
        {
            "$set": {
                "lifecycle_state": state,
                "is_active": False,
                "terminal_reason": reason,
                "terminal_date": signal_date,
                "outcomes": outcomes,
                "last_processed_date": signal_date,
                "updated_at": now,
            },
            "$push": {"events": {"$each": [_event(state, signal_date, plan_doc.get("lifecycle_state"), state, reason, price)], "$slice": -MAX_EVENTS}},
        },
    )
    await db.max_strategy_signals.update_one(
        {"setup_key": f"{plan_doc['plan_key']}:{state}"},
        {"$setOnInsert": {
            "setup_key": f"{plan_doc['plan_key']}:{state}",
            "plan_key": plan_doc["plan_key"],
            "plan_id": plan_doc["plan_key"],
            "lifecycle_version": LIFECYCLE_VERSION,
            "ticker": plan_doc["ticker"],
            "signal_date": signal_date,
            "status": state,
            "terminal_reason": reason,
            "signal_price": price,
            "created_at": now,
        }, "$set": {
            "validation_cohort": plan_doc.get("validation_cohort"),
            "outcomes": outcomes,
            "updated_at": now,
        }},
        upsert=True,
    )
    await db.max_strategy_signals.update_many(
        {"plan_key": plan_doc["plan_key"], "lifecycle_version": LIFECYCLE_VERSION},
        {"$set": {"outcomes": outcomes, "plan_state": state, "updated_at": now}},
    )


async def _create_plan(db, ticker, max_strategy, plan, state, signal_date, signal_price, trigger, maximum_entry, invalidation):
    source = plan.get("trigger_source") or "UNKNOWN"
    plan_key = f"{ticker}:{source}:{signal_date}"
    now = datetime.utcnow()
    triggered = state in ("TRIGGERED",) + CONFIRMED_STATUSES + ("WAIT_RETEST",)
    plan_doc = {
        "plan_key": plan_key,
        "ticker": ticker,
        "lifecycle_version": LIFECYCLE_VERSION,
        "strategy_version": max_strategy.get("version"),
        "trigger_source": source,
        "structure_anchor": plan.get("structure_anchor"),
        "atr14": _num(plan.get("atr14")) or _num(max_strategy.get("atr14")),
        "created_signal_date": signal_date,
        "lifecycle_state": state,
        "is_active": True,
        "order_action": plan.get("order_action"),
        "validation_cohort": _cohort(state, plan.get("order_action")),
        "trigger_price_initial": trigger,
        "trigger_price": trigger,
        "maximum_entry_price": maximum_entry,
        "invalidation_price": invalidation,
        "entry_reference_price": trigger if triggered else None,
        "triggered_at": signal_date if triggered else None,
        "confirmed_at": signal_date if state in CONFIRMED_STATUSES or state == "WAIT_RETEST" else None,
        "trigger_failures": 0,
        "strategy_type": max_strategy.get("strategy_type"),
        "market_phase": (max_strategy.get("market_phase") or {}).get("phase"),
        "last_processed_date": signal_date,
        "last_seen_status": state,
        "events": [_event("CREATED", signal_date, None, state, None, signal_price)],
        "created_at": now,
        "updated_at": now,
    }
    await db.max_strategy_plans.update_one(
        {"plan_key": plan_key},
        {"$setOnInsert": plan_doc},
        upsert=True,
    )
    stored = await db.max_strategy_plans.find_one({"plan_key": plan_key})
    await _write_signal(db, stored, max_strategy, stored.get("lifecycle_state"), signal_date, signal_price)
    return plan_key


def _next_state(current, raw, failures):
    if raw not in TRACKED_STATUSES or raw == current:
        return current, None
    if current in CONFIRMED_STATUSES:
        return current, None
    if current == "WAIT_RETEST":
        return (raw, "CONFIRMED") if raw in CONFIRMED_STATUSES else (current, None)
    if current == "TRIGGERED":
        if raw == "ARMED":
            if failures < MAX_TRIGGER_FAILURES:
                return "ARMED", "TRIGGER_FAILED"
            return "INVALIDATED", "TRIGGER_FAILED_REPEATED"
        return raw, "CONFIRMED" if raw in CONFIRMED_STATUSES else "WAIT_RETEST"
    if current == "ARMED":
        if STATE_RANK.get(raw, 0) > STATE_RANK["ARMED"]:
            return raw, "TRIGGERED" if raw == "TRIGGERED" else "CONFIRMED" if raw in CONFIRMED_STATUSES else "WAIT_RETEST"
    return current, None


async def sync_max_plan(db, asset_doc, df):
    if df is None or len(df) == 0 or not asset_doc:
        return None
    ticker = asset_doc.get("ticker")
    max_strategy = asset_doc.get("max_strategy") or {}
    plan = max_strategy.get("entry_plan") or {}
    raw = plan.get("status")
    signal_date = df["datetime"].iloc[-1].strftime("%Y-%m-%d")
    close = float(df["Close"].iloc[-1])
    signal_price = _num(asset_doc.get("price")) or close
    qualified, trigger, maximum_entry, invalidation = _qualification(max_strategy, plan, signal_price)
    atr = _num(plan.get("atr14")) or _num(max_strategy.get("atr14")) or 0.0

    active = await db.max_strategy_plans.find_one({"ticker": ticker, "is_active": True})

    if active:
        frozen_invalidation = _num(active.get("invalidation_price"))
        if frozen_invalidation and close <= frozen_invalidation:
            await _close_plan(db, active, "INVALIDATED", signal_date, "CLOSE_BELOW_INVALIDATION", df, close)
            active = None
        elif max_strategy.get("data_quality", {}).get("corporate_action_suspected") or "CORPORATE_ACTION_SUSPECTED" in (max_strategy.get("data_rejection_reasons") or []):
            await _close_plan(db, active, "INVALIDATED", signal_date, "CORPORATE_ACTION_SUSPECTED", df, close)
            active = None

    if active:
        state = active.get("lifecycle_state")
        if state in ("ARMED", "TRIGGERED") and _bars_after(df, active.get("created_signal_date")) >= EXPIRATION_BARS:
            await _close_plan(db, active, "EXPIRED", signal_date, "EXPIRATION_BARS", df, close)
            active = None
        elif (state in CONFIRMED_STATUSES or state == "WAIT_RETEST") and _bars_after(df, active.get("triggered_at")) >= OBSERVATION_BARS:
            await _close_plan(db, active, "CLOSED", signal_date, "OBSERVATION_COMPLETE", df, close)
            active = None

    if active and raw in TRACKED_STATUSES and qualified and not _same_structure(active, plan, trigger, atr):
        await _close_plan(db, active, "SUPERSEDED", signal_date, "STRUCTURE_CHANGED", df, close)
        active = None

    if not active:
        if raw in TRACKED_STATUSES and qualified:
            previous = await db.max_strategy_plans.find_one(
                {"ticker": ticker, "is_active": False},
                sort=[("updated_at", -1)],
            )
            if previous and previous.get("terminal_reason") != "STRUCTURE_CHANGED" and _same_structure(previous, plan, trigger, atr):
                return None
            return await _create_plan(db, ticker, max_strategy, plan, raw, signal_date, signal_price, trigger, maximum_entry, invalidation)
        return None

    state = active.get("lifecycle_state")
    now = datetime.utcnow()
    base_set = {"last_seen_status": raw, "last_seen_date": signal_date, "updated_at": now}

    if active.get("last_processed_date") == signal_date and active.get("last_seen_status") == raw:
        outcomes = _plan_outcomes(df, active)
        await db.max_strategy_plans.update_one({"plan_key": active["plan_key"]}, {"$set": {**base_set, "outcomes": outcomes}})
        return active["plan_key"]

    new_state, event_type = _next_state(state, raw, int(active.get("trigger_failures", 0) or 0))

    if new_state == "INVALIDATED":
        await _close_plan(db, active, "INVALIDATED", signal_date, event_type, df, close)
        return active["plan_key"]

    update_set = dict(base_set)
    update_set["last_processed_date"] = signal_date
    update_inc = {}
    events = []

    if state in ("ARMED",) and raw in TRACKED_STATUSES and trigger and new_state == "ARMED":
        reference = _num(active.get("trigger_price"))
        if reference and abs(trigger - reference) > 1e-9:
            update_set["trigger_price"] = trigger
            update_set["maximum_entry_price"] = maximum_entry

    if event_type:
        events.append(_event(event_type, signal_date, state, new_state, None, signal_price))
        update_set["lifecycle_state"] = new_state
        update_set["order_action"] = plan.get("order_action")
        update_set["validation_cohort"] = _cohort(new_state, plan.get("order_action"))
        if event_type == "TRIGGER_FAILED":
            update_inc["trigger_failures"] = 1
            update_set["triggered_at"] = None
            update_set["entry_reference_price"] = None
        if new_state in ("TRIGGERED", "WAIT_RETEST") + CONFIRMED_STATUSES and not active.get("triggered_at"):
            update_set["triggered_at"] = signal_date
            update_set["entry_reference_price"] = trigger or active.get("trigger_price")
        if new_state in CONFIRMED_STATUSES and not active.get("confirmed_at"):
            update_set["confirmed_at"] = signal_date

    merged = {**active, **update_set}
    update_set["outcomes"] = _plan_outcomes(df, merged)
    update = {"$set": update_set}
    if update_inc:
        update["$inc"] = update_inc
    if events:
        update["$push"] = {"events": {"$each": events, "$slice": -MAX_EVENTS}}
    await db.max_strategy_plans.update_one({"plan_key": active["plan_key"]}, update)

    if event_type:
        await _write_signal(db, merged, max_strategy, new_state, signal_date, signal_price)
    if update_set.get("outcomes"):
        await db.max_strategy_signals.update_many(
            {"plan_key": active["plan_key"], "lifecycle_version": LIFECYCLE_VERSION},
            {"$set": {"outcomes": update_set["outcomes"], "plan_state": update_set.get("lifecycle_state", state)}},
        )
    return active["plan_key"]


async def expire_plans_for_stale(db, ticker, last_bar_date):
    active = await db.max_strategy_plans.find_one({"ticker": ticker, "is_active": True})
    if not active:
        return 0
    now = datetime.utcnow()
    await db.max_strategy_plans.update_one(
        {"plan_key": active["plan_key"]},
        {
            "$set": {
                "lifecycle_state": "EXPIRED",
                "is_active": False,
                "terminal_reason": "STALE_OR_DELISTED",
                "terminal_date": last_bar_date,
                "updated_at": now,
            },
            "$push": {"events": {"$each": [_event("EXPIRED", last_bar_date, active.get("lifecycle_state"), "EXPIRED", "STALE_OR_DELISTED")], "$slice": -MAX_EVENTS}},
        },
    )
    return 1


async def mark_legacy_max_signals(db):
    marked = 0
    for cohort in LEGACY_COHORTS:
        result = await db.max_strategy_signals.update_many(
            {"lifecycle_version": {"$exists": False}, "validation_cohort": cohort},
            {"$set": {"lifecycle_version": 0, "legacy_cohort": cohort, "validation_cohort": "LEGACY_PRE_LIFECYCLE"}},
        )
        marked += result.modified_count
    result = await db.max_strategy_signals.update_many(
        {"lifecycle_version": {"$exists": False}},
        {"$set": {"lifecycle_version": 0, "legacy_cohort": None, "validation_cohort": "LEGACY_PRE_LIFECYCLE"}},
    )
    return marked + result.modified_count
