"""
Professional Trader Report Generator
====================================
Generates a single, continuous market analysis written as a senior
portfolio manager would write the daily desk brief.

Called at the end of every pipeline run (Orchestrator) or on-demand
via POST /api/agents/professional-report.

Token policy:
- Target ~160-220 parole (chiude sempre la frase)
- max_tokens 700 (margine, non obiettivo)
- Prompt utente compatto: niente copie lunghe di macro/risk LLM
- Rigenera solo se contesto cambiato o report piu' vecchio di N ore
"""

from datetime import datetime, timedelta
from app.services.llm_service import llm_ask, llm_available


# Quante ore puo' restare valido un brief se il contesto non cambia
REPORT_MAX_AGE_HOURS = 6

SYSTEM_PROMPT = """Sei un portfolio manager senior (swing trading US, 3-20 giorni).

Scrivi in italiano, tono professionale e diretto.

Regole strette:
- Un unico testo continuo, niente elenchi, niente markdown.
- Lunghezza: 160-220 parole. NON superare 220 parole.
- Inizia da regime e bias operativo.
- Collega leadership/rotazione al posizionamento.
- Chiudi SEMPRE con una frase completa su rischi e stance (aggressiva / neutrale / difensiva).
- Niente disclaimers, niente "come AI", niente frasi vuote.
- Se stai per finire i token, chiudi la frase corrente e stop."""


def _safe(v, default="N/D"):
    return v if v is not None and v != "" else default


def context_fingerprint(context):
    """Firma compatta del contesto: se uguale, il brief puo' restare."""
    focus = ",".join((context.get("focus_sectors") or [])[:5])
    avoid = ",".join((context.get("avoid_sectors") or [])[:5])
    picks = ",".join((context.get("top_picks") or [])[:4])
    return "|".join([
        str(_safe(context.get("regime"))),
        str(_safe(context.get("regime_detail"))),
        str(round(float(context.get("exposure_multiplier") or 0), 2)),
        str(round(float(context.get("breadth_pct") or 0), 0)),
        str(_safe(context.get("volatility"))),
        str(_safe(context.get("leadership"))),
        str(_safe(context.get("rotation"))),
        focus,
        avoid,
        picks,
    ])


def build_user_prompt(context):
    """Prompt utente compatto: solo fatti operativi, zero ripetizioni lunghe."""
    regime = _safe(context.get("regime"))
    detail = _safe(context.get("regime_detail"))
    conf = context.get("confidence", 0)
    exposure = context.get("exposure_multiplier", 0)
    breadth = context.get("breadth_pct", 0)
    vol = _safe(context.get("volatility"))
    leadership = _safe(context.get("leadership"))
    rotation = _safe(context.get("rotation"))
    focus = context.get("focus_sectors") or []
    avoid = context.get("avoid_sectors") or []
    top_picks = context.get("top_picks") or []
    sells = context.get("sells") or []
    approved = context.get("approved_count", 0)
    rejected = context.get("rejected_count", 0)
    executed_buys = context.get("executed_buys", 0)
    executed_sells = context.get("executed_sells", 0)

    lines = [
        "Regime: %s / %s (conf %s)" % (regime, detail, conf),
        "Esposizione: %s%% | Breadth: %s%% | Vol: %s" % (
            round(float(exposure or 0) * 100), breadth, vol
        ),
        "Leadership: %s | Rotazione: %s" % (leadership, rotation),
    ]
    if focus:
        lines.append("FOCUS: %s" % ", ".join(focus[:5]))
    if avoid:
        lines.append("EVITA: %s" % ", ".join(avoid[:5]))
    if top_picks:
        lines.append("Top picks: %s" % ", ".join(top_picks[:5]))
    if sells:
        lines.append("Sell: %s" % ", ".join(sells[:4]))
    lines.append(
        "Risk oggi: appro %s / rej %s | exec BUY %s SELL %s"
        % (approved, rejected, executed_buys, executed_sells)
    )
    # Nota macro solo se corta (risparmio token input)
    macro_llm = (context.get("macro_llm") or "").strip()
    if macro_llm and len(macro_llm) <= 180:
        lines.append("Nota macro: %s" % macro_llm)

    lines.append(
        "Scrivi il desk brief (160-220 parole). Chiudi con stance e rischio principale."
    )
    return "\n".join(lines)


def _looks_truncated(text):
    """True se sembra tagliato a meta' frase."""
    t = (text or "").strip()
    if not t or len(t) < 80:
        return True
    if t[-1] in ".!?:;\"'"":
        return False
    # finisce con virgola / parola spezzata
    return True


def generate_professional_report(context):
    """Genera il report. Ritorna None se LLM non disponibile o output inutile."""
    if not llm_available():
        print("  professional_report: llm_available() = False")
        return None

    user_prompt = build_user_prompt(context)
    text = llm_ask(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_tokens=700,
        temperature=0.3,
        agent_name=None,
    )
    if not text or len(text.strip()) < 80:
        print("  professional_report: llm_ask returned empty/short")
        return None

    text = text.strip()
    if _looks_truncated(text):
        # Non salvare pezzi a meta' frase: spreca UI e confonde
        print("  professional_report: output truncated, discarded")
        return None

    return text


async def should_regenerate_report(context):
    """
    True solo se conviene spendere token:
    - nessun report in DB
    - report piu' vecchio di REPORT_MAX_AGE_HOURS
    - fingerprint contesto cambiato (regime/focus/exp/picks)
    """
    from app.db.mongodb import get_db

    db = get_db()
    doc = await db.market_context.find_one({"_id": "latest"}) or {}
    existing = (doc.get("professional_report") or "").strip()
    if not existing or _looks_truncated(existing):
        return True, "missing_or_truncated"

    ts_raw = doc.get("professional_report_at")
    if ts_raw:
        try:
            ts = datetime.fromisoformat(str(ts_raw).replace("Z", ""))
            age_h = (datetime.utcnow() - ts).total_seconds() / 3600
            if age_h >= REPORT_MAX_AGE_HOURS:
                return True, "stale_%.1fh" % age_h
        except Exception:
            return True, "bad_timestamp"
    else:
        return True, "no_timestamp"

    meta = doc.get("professional_report_meta") or {}
    old_fp = meta.get("fingerprint")
    new_fp = context_fingerprint(context)
    if old_fp and old_fp != new_fp:
        return True, "context_changed"

    return False, "reuse_existing"


async def save_professional_report(report_text, meta=None):
    """Salva report su SharedBrain (market) e su market_context."""
    from app.db.mongodb import get_db
    from app.agents.shared_brain import brain

    db = get_db()
    payload = {
        "professional_report": report_text,
        "professional_report_at": datetime.utcnow().isoformat(),
        "professional_report_meta": meta or {},
    }

    try:
        market = await brain.get_market()
        market.update(payload)
        await brain.write_market(market)
    except Exception as e:
        print("  professional_report brain write error: %s" % e)

    try:
        await db.market_context.update_one(
            {"_id": "latest"},
            {"$set": payload},
            upsert=True,
        )
    except Exception as e:
        print("  professional_report market_context write error: %s" % e)


async def generate_and_save_if_needed(context, force=False):
    """
    Entry-point pipeline: genera solo se serve.
    Ritorna dict {status, text?, reason}
    """
    fp = context_fingerprint(context)

    if not force:
        need, reason = await should_regenerate_report(context)
        if not need:
            print("  Professional report reused (%s)" % reason)
            return {"status": "reused", "reason": reason}
    else:
        reason = "force"

    text = generate_professional_report(context)
    if not text:
        return {"status": "skipped", "reason": reason}

    await save_professional_report(text, meta={
        "regime": context.get("regime"),
        "exposure": context.get("exposure_multiplier"),
        "fingerprint": fp,
        "source": "pipeline" if not force else "force",
    })
    print("  Professional report generated (%s)" % reason)
    return {"status": "generated", "reason": reason, "text": text}


async def build_context_from_brain():
    """Context on-demand da SharedBrain + market_context."""
    from app.db.mongodb import get_db
    from app.agents.shared_brain import brain

    db = get_db()
    market = await brain.get_market() or {}
    candidates = await brain.get_candidates() or {}
    approved = await brain.get_approved() or {}
    executions = await brain.get_executions() or {}
    ctx_doc = await db.market_context.find_one({"_id": "latest"}) or {}

    buy = candidates.get("buy") or []
    top_picks = [
        c["ticker"] for c in buy[:6]
        if isinstance(c, dict) and c.get("ticker")
    ]
    sells = [
        s["ticker"] for s in (candidates.get("sell") or [])[:5]
        if isinstance(s, dict) and s.get("ticker")
    ]
    risk_report = approved.get("risk_report") or {}

    leadership = market.get("leadership")
 if isinstance(ctx_doc.get("leadership"), dict):
            leadership = leadership or ctx_doc.get("leadership", {}).get("state")
        else:
            leadership = leadership or ctx_doc.get("leadership_state")

    return {
        "regime": market.get("regime") or ctx_doc.get("market_regime"),
        "regime_detail": market.get("regime_detail") or ctx_doc.get("regime_detail"),
        "confidence": market.get("confidence") or ctx_doc.get("regime_confidence"),
        "exposure_multiplier": market.get("exposure_multiplier") or ctx_doc.get("exposure_multiplier"),
        "breadth_pct": market.get("breadth_pct") or ctx_doc.get("breadth_pct"),
        "volatility": market.get("volatility") or ctx_doc.get("volatility_regime"),
        "leadership": leadership,
        "rotation": market.get("rotation") or ctx_doc.get("rotation_signal") or ctx_doc.get("rotation_state"),
        "focus_sectors": ctx_doc.get("focus_sectors") or [],
        "avoid_sectors": ctx_doc.get("avoid_sectors") or [],
        "macro_llm": market.get("llm_reasoning") or ctx_doc.get("llm_reasoning"),
        "risk_llm": risk_report.get("llm_reasoning") if isinstance(risk_report, dict) else None,
        "apm_summary": {},
        "top_picks": top_picks,
        "sells": sells,
        "approved_count": len(approved.get("trades") or []),
        "rejected_count": 0,
        "executed_buys": len(executions.get("last_buys") or []),
        "executed_sells": len(executions.get("last_sells") or []),
    }
