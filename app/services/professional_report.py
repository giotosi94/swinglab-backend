"""
Professional Trader Report Generator
====================================
Generates a single, continuous market analysis written as a senior
portfolio manager would write the daily desk brief.

Called at the end of every pipeline run (Orchestrator) or on-demand
via POST /api/agents/professional-report.
"""

from datetime import datetime
from app.services.llm_service import llm_ask, llm_available


SYSTEM_PROMPT = """Sei un portfolio manager senior con 20 anni di esperienza su mercati azionari US (swing trading, orizzonte 3-20 giorni).

Scrivi un'analisi di mercato in italiano, tono professionale, diretto, senza riempitivi.

Regole:
- Un unico testo continuo (niente elenchi puntati lunghi, niente titoli markdown).
- Lunghezza ideale: 180-280 parole.
- Inizia dal regime e dal bias operativo attuale.
- Collega leadership/rotazione settoriale alle decisioni di posizionamento.
- Commenta candidati Alpha e cosa sta facendo l'APM sulle posizioni aperte solo se rilevante.
- Chiudi con rischi principali e stance operativa chiara (aggressiva / neutrale / difensiva).
- Niente disclaimers, niente "come AI", niente frasi generiche tipo "i mercati sono volatili".
- Scrivi come se parlassi a un collega trader esperto."""


def _safe(v, default="N/D"):
    return v if v is not None and v != "" else default


def build_user_prompt(context):
    """Costruisce il prompt utente dai dati gia' prodotti dalla pipeline."""
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
    macro_llm = context.get("macro_llm") or ""
    risk_llm = context.get("risk_llm") or ""
    apm_summary = context.get("apm_summary") or {}
    top_picks = context.get("top_picks") or []
    sells = context.get("sells") or []
    approved = context.get("approved_count", 0)
    rejected = context.get("rejected_count", 0)
    executed_buys = context.get("executed_buys", 0)
    executed_sells = context.get("executed_sells", 0)

    lines = [
        "Regime: %s / dettaglio %s (confidenza %s)" % (regime, detail, conf),
        "Esposizione consentita: %s%%" % round(float(exposure or 0) * 100),
        "Breadth (sopra EMA50): %s%%  |  Volatilita': %s" % (breadth, vol),
        "Leadership: %s  |  Rotazione: %s" % (leadership, rotation),
    ]
    if focus:
        lines.append("Settori FOCUS: %s" % ", ".join(focus[:5]))
    if avoid:
        lines.append("Settori da EVITARE: %s" % ", ".join(avoid[:5]))
    if top_picks:
        lines.append("Top candidati Alpha: %s" % ", ".join(top_picks[:6]))
    if sells:
        lines.append("Sell signals: %s" % ", ".join(sells[:5]))
    lines.append(
        "Risk: approvati %s, rifiutati %s | Eseguiti oggi: BUY %s, SELL %s"
        % (approved, rejected, executed_buys, executed_sells)
    )
    if apm_summary:
        lines.append("APM summary: %s" % apm_summary)
    if macro_llm:
        lines.append("Nota Macro gia' prodotta: %s" % macro_llm[:350])
    if risk_llm:
        lines.append("Nota Risk gia' prodotta: %s" % risk_llm[:250])

    lines.append(
        "\nScrivi ora l'analisi professionale completa del desk, "
        "un unico testo continuo, come descritto nel system prompt."
    )
    return "\n".join(lines)


def generate_professional_report(context):
    """Genera il report. Ritorna None se LLM non disponibile."""
    if not llm_available():
        print("  professional_report: llm_available() = False")
        return None

    user_prompt = build_user_prompt(context)
    # agent_name=None: niente cooldown/cache agent (dopo pipeline e' spesso in cooldown)
    text = llm_ask(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_tokens=450,
        temperature=0.35,
        agent_name=None,
    )
    if text and len(text.strip()) > 80:
        return text.strip()
    print("  professional_report: llm_ask returned empty/short")
    return None


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


async def build_context_from_brain():
    """Costruisce il context dal SharedBrain + market_context per generazione on-demand."""
    from app.db.mongodb import get_db
    from app.agents.shared_brain import brain

    db = get_db()
    market = await brain.get_market() or {}
    candidates = await brain.get_candidates() or {}
    approved = await brain.get_approved() or {}
    executions = await brain.get_executions() or {}
    ctx_doc = await db.market_context.find_one({"_id": "latest"}) or {}

    buy = candidates.get("buy") or []
    top_picks = []
    for c in buy[:6]:
        if isinstance(c, dict) and c.get("ticker"):
            top_picks.append(c["ticker"])

    sells = []
    for s in (candidates.get("sell") or [])[:5]:
        if isinstance(s, dict) and s.get("ticker"):
            sells.append(s["ticker"])

    risk_report = approved.get("risk_report") or {}

    return {
        "regime": market.get("regime") or ctx_doc.get("market_regime"),
        "regime_detail": market.get("regime_detail") or ctx_doc.get("regime_detail"),
        "confidence": market.get("confidence") or ctx_doc.get("regime_confidence"),
        "exposure_multiplier": market.get("exposure_multiplier") or ctx_doc.get("exposure_multiplier"),
        "breadth_pct": market.get("breadth_pct") or ctx_doc.get("breadth_pct"),
        "volatility": market.get("volatility") or ctx_doc.get("volatility_regime"),
        "leadership": market.get("leadership") or ctx_doc.get("leadership", {}).get("state") if isinstance(ctx_doc.get("leadership"), dict) else ctx_doc.get("leadership_state"),
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
