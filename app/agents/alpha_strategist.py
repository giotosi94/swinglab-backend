"""
LLM Shadow for Max Strategy
===========================
Log-only layer: the LLM proposes what it *would* do on Max Strategy
candidates. Nothing is executed. Outcomes are filled later by an evaluator.

Collection: llm_shadow_decisions
Dashboard summary: agent_state._id = "max_strategy_llm_shadow"
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.db.mongodb import get_db
from app.services.llm_service import llm_ask, llm_available

PROMPT_VERSION = "max_shadow_v1"
VALID_ACTIONS = {"WOULD_BUY", "WOULD_ARM", "WOULD_WAIT", "WOULD_REJECT"}

SYSTEM_PROMPT = """You are a senior swing-trading structural analyst reviewing Max Strategy setups.
You do NOT place orders. You only say what you WOULD do and why.

Allowed actions (use exactly one):
- WOULD_BUY
- WOULD_ARM
- WOULD_WAIT
- WOULD_REJECT

Respond with ONLY a compact JSON object, no markdown, no extra text:
{
  "suggested_action": "WOULD_BUY|WOULD_ARM|WOULD_WAIT|WOULD_REJECT",
  "confidence": 0-100,
  "reasoning": "2-4 concise sentences in Italian",
  "key_factors": ["factor1", "factor2", "factor3"]
}

Be selective. Prefer WOULD_WAIT or WOULD_REJECT when structure is incomplete,
overhead is tight, phase is hostile, or regime is defensive. Use WOULD_BUY
only when structure + phase + plan are clearly aligned."""


def _context_hash(context: dict) -> str:
    payload = json.dumps(context, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def build_candidate_context(candidate: dict, market_ctx: dict) -> dict:
    """Dense, LLM-friendly context from a Max Strategy shadow candidate."""
    return {
        "ticker": candidate.get("ticker"),
        "max_score": candidate.get("max_score"),
        "strategy_type": candidate.get("strategy_type"),
        "market_phase": candidate.get("market_phase"),
        "plan_status": candidate.get("plan_status"),
        "order_action": candidate.get("order_action"),
        "execution_mode": candidate.get("execution_mode"),
        "trade_ready": bool(candidate.get("trade_ready")),
        "strategy_eligible": bool(candidate.get("strategy_eligible")),
        "weekly_qualified": bool(candidate.get("weekly_qualified")),
        "blocking_phase": bool(candidate.get("blocking_phase")),
        "current_price": candidate.get("signal_price"),
        "trigger_price": candidate.get("trigger_price"),
        "maximum_entry_price": candidate.get("maximum_entry_price"),
        "invalidation_price": candidate.get("invalidation_price"),
        "rejection_reasons": list(candidate.get("rejection_reasons") or []),
        "regime": market_ctx.get("market_regime") or market_ctx.get("regime") or "NEUTRAL",
        "regime_detail": market_ctx.get("regime_detail"),
        "exposure_multiplier": market_ctx.get("exposure_multiplier"),
    }


def parse_llm_response(raw: Optional[str]) -> Optional[dict]:
    if not raw or not isinstance(raw, str):
        return None
    text = raw.strip()
    # Strip accidental markdown fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except Exception:
        # Best-effort extraction of first JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except Exception:
            return None

    action = str(data.get("suggested_action") or "").upper().strip()
    if action not in VALID_ACTIONS:
        return None

    try:
        confidence = int(float(data.get("confidence", 0)))
    except Exception:
        confidence = 0
    confidence = max(0, min(100, confidence))

    reasoning = str(data.get("reasoning") or "").strip()[:600]
    key_factors = data.get("key_factors") or []
    if not isinstance(key_factors, list):
        key_factors = []
    key_factors = [str(x)[:40] for x in key_factors[:6]]

    return {
        "suggested_action": action,
        "confidence": confidence,
        "reasoning": reasoning,
        "key_factors": key_factors,
        "raw_response": raw[:2000],
    }


async def log_llm_shadow(doc: dict) -> str:
    db = get_db()
    doc = dict(doc)
    doc.setdefault("created_at", datetime.utcnow())
    doc.setdefault("outcome", {
        "evaluated_at": None,
        "horizon_days": 7,
        "price_at_horizon": None,
        "pnl_pct": None,
        "max_adverse_excursion_pct": None,
        "system_was_correct": None,
        "llm_was_correct": None,
        "llm_better_than_system": None,
        "notes": None,
    })
    result = await db.llm_shadow_decisions.insert_one(doc)
    return str(result.inserted_id)


def select_candidates_for_llm(candidates: List[dict], limit: int = 8) -> List[dict]:
    """Prefer actionable / high-score setups; skip boring REJECT noise."""
    priority = {"WOULD_BUY": 0, "WOULD_ARM": 1, "WOULD_WAIT": 2, "WOULD_REJECT": 3}
    ranked = sorted(
        candidates or [],
        key=lambda c: (
            priority.get(c.get("shadow_action"), 9),
            -float(c.get("max_score") or 0),
        ),
    )
    selected = []
    for c in ranked:
        action = c.get("shadow_action")
        if action == "WOULD_REJECT" and float(c.get("max_score") or 0) < 55:
            continue
        if action == "WOULD_WAIT" and float(c.get("max_score") or 0) < 45:
            continue
        selected.append(c)
        if len(selected) >= limit:
            break
    return selected


async def run_max_strategy_llm_shadow(
    shadow_result: dict,
    market_ctx: dict,
    limit: int = 8,
) -> dict:
    """
    Run LLM shadow on top Max Strategy candidates.
    Purely observational — never affects orders.
    Returns a summary suitable for agent_state / Dashboard.
    """
    summary = {
        "mode": "LLM_SHADOW",
        "prompt_version": PROMPT_VERSION,
        "enabled": True,
        "llm_available": llm_available(),
        "n_candidates_considered": 0,
        "n_llm_calls": 0,
        "n_parsed": 0,
        "agreement": 0,
        "divergence": 0,
        "action_counts": {a: 0 for a in VALID_ACTIONS},
        "items": [],
        "updated_at": datetime.utcnow(),
    }

    if not llm_available():
        summary["enabled"] = False
        summary["message"] = "LLM not available"
        return summary

    candidates = select_candidates_for_llm(
        (shadow_result or {}).get("candidates") or [],
        limit=limit,
    )
    summary["n_candidates_considered"] = len(candidates)
    if not candidates:
        summary["message"] = "No interesting candidates"
        return summary

    items = []
    for candidate in candidates:
        context = build_candidate_context(candidate, market_ctx or {})
        ticker = context.get("ticker") or "?"
        system_action = candidate.get("shadow_action") or "WOULD_WAIT"

        user_prompt = (
            f"Ticker: {ticker}\n"
            f"System (Max Strategy rules) action: {system_action}\n"
            f"Context JSON:\n{json.dumps(context, ensure_ascii=False, default=str)}"
        )

        t0 = time.time()
        raw = llm_ask(
            SYSTEM_PROMPT,
            user_prompt,
            max_tokens=280,
            temperature=0.2,
            agent_name="max_strategy_llm_shadow",
        )
        latency_ms = int((time.time() - t0) * 1000)
        summary["n_llm_calls"] += 1

        parsed = parse_llm_response(raw)
        if not parsed:
            continue

        summary["n_parsed"] += 1
        llm_action = parsed["suggested_action"]
        summary["action_counts"][llm_action] = summary["action_counts"].get(llm_action, 0) + 1

        agrees = llm_action == system_action
        if agrees:
            summary["agreement"] += 1
        else:
            summary["divergence"] += 1

        doc = {
            "source": "max_strategy",
            "agent": "alpha_strategist",
            "ticker": ticker,
            "created_at": datetime.utcnow(),
            "context": context,
            "system_shadow": {
                "action": system_action,
                "reasons": list(candidate.get("rejection_reasons") or []),
                "max_score": candidate.get("max_score"),
                "plan_status": candidate.get("plan_status"),
            },
            "llm": {
                "provider": None,  # llm_ask does not return provider today
                "model": None,
                "prompt_version": PROMPT_VERSION,
                "suggested_action": llm_action,
                "confidence": parsed["confidence"],
                "reasoning": parsed["reasoning"],
                "key_factors": parsed["key_factors"],
                "raw_response": parsed.get("raw_response"),
            },
            "meta": {
                "latency_ms": latency_ms,
                "context_hash": _context_hash(context),
                "agrees_with_system": agrees,
            },
        }

        try:
            doc_id = await log_llm_shadow(doc)
            doc["_id"] = doc_id
        except Exception as e:
            print(f"  ⚠️ llm_shadow log error ({ticker}): {e}")

        items.append({
            "ticker": ticker,
            "system_action": system_action,
            "llm_action": llm_action,
            "confidence": parsed["confidence"],
            "agrees": agrees,
            "max_score": candidate.get("max_score"),
            "reasoning": parsed["reasoning"],
            "key_factors": parsed["key_factors"],
        })

    # Prefer showing divergences first in the dashboard strip
    items.sort(key=lambda x: (x.get("agrees", True), -float(x.get("confidence") or 0)))
    summary["items"] = items[:12]
    return summary
