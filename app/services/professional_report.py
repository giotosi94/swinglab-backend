"""
Professional Trader Report Generator
====================================
Generates a single, continuous market analysis written as a senior
portfolio manager would write the daily desk brief.

Called at the end of every pipeline run (Orchestrator).
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


def build_user_prompt(context: dict) -> str:
    """Costruisce il prompt utente dai dati già prodotti dalla pipeline."""
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
        f"Regime: {regime} / dettaglio {detail} (confidenza {conf})",
        f"Esposizione consentita: {round(float(exposure or 0) * 100)}%",
        f"Breadth (sopra EMA50): {breadth}%  |  Volatilità: {vol}",
        f"Leadership: {leadership}  |  Rotazione: {rotation}",
    ]
    if focus:
        lines.append(f"Settori FOCUS: {', '.join(focus[:5])}")
    if avoid:
        lines.append(f"Settori da EVITARE: {', '.join(avoid[:5])}")
    if top_picks:
        lines.append(f"Top candidati Alpha: {', '.join(top_picks[:6])}")
    if sells:
        lines.append(f"Sell signals: {', '.join(sells[:5])}")
    lines.append(
        f"Risk: approvati {approved}, rifiutati {rejected} | "
        f"Eseguiti oggi: BUY {executed_buys}, SELL {executed_sells}"
    )
    if apm_summary:
        lines.append(f"APM summary: {apm_summary}")
    if macro_llm:
        lines.append(f"Nota Macro già prodotta: {macro_llm[:350]}")
    if risk_llm:
        lines.append(f"Nota Risk già prodotta: {risk_llm[:250]}")

    lines.append(
        "\nScrivi ora l'analisi professionale completa del desk, "
        "un unico testo continuo, come descritto nel system prompt."
    )
    return "\n".join(lines)


def generate_professional_report(context: dict) -> str | None:
    """
    Genera il report. Ritorna None se LLM non disponibile.
    """
    if not llm_available():
        return None

    user_prompt = build_user_prompt(context)
    text = llm_ask(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_tokens=450,
        temperature=0.35,
        agent_name="professional_report",
    )
    if text and len(text.strip()) > 80:
        return text.strip()
    return None


async def save_professional_report(report_text: str, meta: dict | None = None):
    """Salva report su SharedBrain (market) e su market_context."""
    from app.db.mongodb import get_db
    from app.agents.shared_brain import brain

    db = get_db()
    payload = {
        "professional_report": report_text,
        "professional_report_at": datetime.utcnow().isoformat(),
        "professional_report_meta": meta or {},
    }

    # SharedBrain market section
    try:
        market = await brain.get_market()
        market.update(payload)
        await brain.write_market(market)
    except Exception as e:
        print(f"  professional_report brain write error: {e}")

    # market_context collection (usata dalla Dashboard)
    try:
        await db.market_context.update_one(
            {"_id": "latest"},
            {"$set": payload},
            upsert=True,
        )
    except Exception as e:
        print(f"  professional_report market_context write error: {e}")
