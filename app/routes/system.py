"""
SwingLab — System Health Dashboard
Checklist automatica dello stato del sistema. NO token LLM nel check base.
"""

from fastapi import APIRouter
from datetime import datetime, timedelta
from app.db.mongodb import get_db

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/health")
async def system_health():
    """
    Health check completo del sistema. NON consuma token LLM.
    Solo query DB + ping Alpaca. Sicuro da chiamare in polling.
    """
    db = get_db()
    checks = {}

    # ============================================
    # CHECK 1 — Freschezza dati stock (il bug SPY!)
    # ============================================
    try:
        assets = await db.assets.find({}, {"ticker": 1, "updated_at": 1}).to_list(300)
        n_assets = len(assets)
        stale = 0
        now = datetime.utcnow()
        for a in assets:
            upd = a.get("updated_at")
            if not upd:
                stale += 1
            elif (now - upd).total_seconds() / 3600 > 48:
                stale += 1
        status = "ok" if stale == 0 else ("warning" if stale < n_assets * 0.2 else "critical")
        checks["data_freshness"] = {
            "status": status,
            "n_assets": n_assets,
            "stale_assets": stale,
            "message": f"{n_assets - stale}/{n_assets} asset freschi",
        }
    except Exception as e:
        checks["data_freshness"] = {"status": "error", "error": str(e)}

    # ============================================
    # CHECK 2 — SPY / Macro fresco (benchmark)
    # ============================================
    try:
        spy = await db.stock_bars.find_one({"ticker": "SPY"})
        if spy and spy.get("bars"):
            last_date = spy["bars"][-1].get("date", "")
            days_old = (datetime.utcnow() - datetime.strptime(last_date, "%Y-%m-%d")).days
            status = "ok" if days_old <= 4 else ("warning" if days_old <= 10 else "critical")
            checks["spy_benchmark"] = {
                "status": status,
                "last_bar": last_date,
                "days_old": days_old,
                "message": f"SPY ultima barra {last_date} ({days_old}g fa)",
            }
        else:
            checks["spy_benchmark"] = {"status": "critical", "message": "SPY senza barre"}
    except Exception as e:
        checks["spy_benchmark"] = {"status": "error", "error": str(e)}

    # ============================================
    # CHECK 3 — Pipeline attiva (updated_at recente)
    # ============================================
    try:
        latest = await db.assets.find_one({}, sort=[("updated_at", -1)])
        if latest and latest.get("updated_at"):
            hours = (datetime.utcnow() - latest["updated_at"]).total_seconds() / 3600
            # In orario mercato ci si aspetta < 1h; weekend tollerato fino a ~72h
            status = "ok" if hours < 24 else ("warning" if hours < 80 else "critical")
            checks["pipeline"] = {
                "status": status,
                "hours_since_last_update": round(hours, 1),
                "message": f"Ultimo update pipeline {round(hours, 1)}h fa",
            }
        else:
            checks["pipeline"] = {"status": "critical", "message": "Nessun update registrato"}
    except Exception as e:
        checks["pipeline"] = {"status": "error", "error": str(e)}

    # ============================================
    # CHECK 4 — Alpaca connesso
    # ============================================
    try:
        from app.services.alpaca_trader import get_account
        account = await get_account()
        if account and float(account.get("equity", 0)) > 0:
            checks["alpaca"] = {
                "status": "ok",
                "equity": round(float(account.get("equity", 0)), 2),
                "message": f"Alpaca connesso · equity ${float(account.get('equity', 0)):,.0f}",
            }
        else:
            checks["alpaca"] = {"status": "critical", "message": "Alpaca non risponde"}
    except Exception as e:
        checks["alpaca"] = {"status": "error", "error": str(e)}

    # ============================================
    # CHECK 5 — ML sano (varianza predizioni)
    # ============================================
    try:
        from app.ml.model import ml_model
        status_ml = await ml_model.get_status()
        if status_ml.get("is_trained"):
            acc = status_ml.get("accuracy", 0)
            status = "ok" if acc > 55 else ("warning" if acc > 45 else "critical")
            checks["ml_model"] = {
                "status": status,
                "accuracy": acc,
                "n_real_positions": status_ml.get("n_real_positions", 0),
                "message": f"ML accuracy {acc}% · {status_ml.get('n_real_positions', 0)} posizioni reali",
            }
        else:
            checks["ml_model"] = {"status": "warning", "message": "ML non ancora addestrato"}
    except Exception as e:
        checks["ml_model"] = {"status": "error", "error": str(e)}

    # ============================================
    # CHECK 6 — Coerenza posizioni Alpaca <-> DB
    # ============================================
    try:
        # 🔧 Usa get_positions() (funzione dedicata) invece di account.get("positions"),
        # che spesso non contiene le posizioni → dava sempre "0" (falso allarme).
        from app.services.alpaca_trader import get_positions
        alpaca_positions = await get_positions() or []
        n_alpaca = len(alpaca_positions)
        db_open = await db.trade_history.count_documents({"side": "buy", "sell_linked": {"$ne": True}})
        diff = abs(n_alpaca - db_open)
        # Tollera glitch temporanei: se Alpaca torna 0 ma il DB ha posizioni,
        # è quasi certo un timeout API, non una desync reale → non allarmare.
        if n_alpaca == 0 and db_open > 0:
            status = "ok"
        else:
            status = "ok" if diff <= 1 else ("warning" if diff <= 3 else "critical")
        checks["positions_sync"] = {
            "status": status,
            "alpaca_positions": n_alpaca,
            "db_open_buys": db_open,
            "message": f"Alpaca {n_alpaca} vs DB {db_open} posizioni",
        }
    except Exception as e:
        checks["positions_sync"] = {"status": "error", "error": str(e)}

    # ============================================
    # CHECK 7 — APM attivo (decisioni recenti)
    # ============================================
    try:
        last_apm = await db.agent_decisions_adaptive_position_manager.find_one(
            {}, sort=[("created_at", -1)]
        )
        if last_apm:
            checks["apm"] = {
                "status": "ok",
                "last_decision": str(last_apm.get("created_at", ""))[:19],
                "message": "APM ha decisioni registrate",
            }
        else:
            checks["apm"] = {"status": "warning", "message": "Nessuna decisione APM registrata"}
    except Exception as e:
        checks["apm"] = {"status": "error", "error": str(e)}

    # ============================================
    # CHECK 8 — Regime valido
    # ============================================
    try:
        spy_regime = await db.market_regime.find_one({"symbol": "SPY"})
        if spy_regime and spy_regime.get("price", 0) > 0:
            rsi = spy_regime.get("rsi", 50)
            valid = 0 <= rsi <= 100
            checks["market_regime"] = {
                "status": "ok" if valid else "warning",
                "spy_price": spy_regime.get("price"),
                "spy_rsi": rsi,
                "message": f"SPY ${spy_regime.get('price')} · RSI {rsi}",
            }
        else:
            checks["market_regime"] = {"status": "critical", "message": "Regime macro non disponibile"}
    except Exception as e:
        checks["market_regime"] = {"status": "error", "error": str(e)}

    # ============================================
    # OVERALL STATUS
    # ============================================
    critical = sum(1 for c in checks.values() if c.get("status") in ("critical", "error"))
    warning = sum(1 for c in checks.values() if c.get("status") == "warning")

    if critical > 0:
        overall = "critical"
    elif warning > 0:
        overall = "warning"
    else:
        overall = "healthy"

    return {
        "overall": overall,
        "summary": {
            "total_checks": len(checks),
            "ok": sum(1 for c in checks.values() if c.get("status") == "ok"),
            "warning": warning,
            "critical": critical,
        },
        "checks": checks,
        "checked_at": datetime.utcnow().isoformat(),
        "llm_used": False,
    }


@router.get("/health/report")
async def system_health_report():
    """
    Report AI in italiano dello stato del sistema.
    ⚠️ CONSUMA 1 chiamata LLM (Gemini/Groq/Cerebras). Solo on-demand.
    """
    health = await system_health()

    from app.services.llm_service import llm_ask, llm_available
    if not llm_available():
        return {**health, "llm_report": "LLM non disponibile al momento."}

    # Costruisci un sommario compatto dei check per il prompt
    lines = []
    for name, c in health["checks"].items():
        lines.append(f"- {name}: {c.get('status', '?').upper()} — {c.get('message', c.get('error', ''))}")
    checks_text = "\n".join(lines)

    try:
        report = llm_ask(
            system_prompt=(
                "Sei un supervisore di sistema per una piattaforma di trading algoritmico. "
                "Analizza lo stato dei check e scrivi un report BREVE in italiano (max 5 frasi). "
                "Evidenzia i problemi critici, spiega se qualcosa è normale (es. pipeline ferma nel weekend), "
                "e dai 1-2 azioni concrete se serve. Sii diretto, tecnico, no disclaimer."
            ),
            user_prompt=f"Stato generale: {health['overall'].upper()}\n\nCheck:\n{checks_text}",
            max_tokens=250,
            temperature=0.3,
            agent_name="system_health",
        )
    except Exception as e:
        report = f"Errore generazione report: {e}"

    return {**health, "llm_report": report, "llm_used": True}
