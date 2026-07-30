"""
🎯 Progetto Alpha — Crash Deploy
Quando il Crash Radar segnala DEPLOY, compra SPY con la fetta long-term.
/simulate = dry-run (calcola, NON compra) · /execute = compra davvero.
"""

from fastapi import APIRouter
from datetime import datetime
from app.db.mongodb import get_db

router = APIRouter(prefix="/api/crash-deploy", tags=["crash-deploy"])

# Config
LONGTERM_SLICE_PCT = 20.0   # % capitale riservata ai crash (fetta long-term)
DEPLOY_TICKER = "SPY"


async def _build_deploy_plan():
    """Legge il Crash Radar e calcola il piano di deploy (senza eseguire)."""
    db = get_db()
    ctx = await db.market_context.find_one({"_id": "latest"})
    if not ctx:
        return {"error": "No market context. Run macro first."}

    cr = ctx.get("crash_radar", {})
    level = cr.get("crash_level", "NORMAL")
    score = cr.get("crash_risk_score", 0)

    # Frazione della fetta da deployare secondo il livello
    deploy_fraction = {"DEPLOY": 0.5, "DEPLOY_MAX": 1.0}.get(level, 0.0)

    # Account
    from app.services.alpaca_trader import get_account, get_positions
    account = await get_account()
    equity = float(account.get("equity", 0)) if account else 0
    cash = float(account.get("cash", 0)) if account else 0

    slice_usd = equity * (LONGTERM_SLICE_PCT / 100)
    deploy_usd = round(slice_usd * deploy_fraction, 2)

    # Quanto SPY ho già (per non sovra-deployare)
    positions = await get_positions() or []
    spy_held = sum(float(p.get("market_value", 0)) for p in positions if p.get("symbol") == DEPLOY_TICKER)

    # Non superare il cash disponibile
    deploy_usd = min(deploy_usd, cash * 0.95)
    if deploy_usd < 0:
        deploy_usd = 0

    return {
        "crash_level": level,
        "crash_score": score,
        "spy_drawdown_pct": cr.get("spy_drawdown_pct", 0),
        "vixy": cr.get("vixy_price", 0),
        "deploy_signal": level in ("DEPLOY", "DEPLOY_MAX"),
        "deploy_fraction": deploy_fraction,
        "longterm_slice_pct": LONGTERM_SLICE_PCT,
        "equity": round(equity, 2),
        "cash": round(cash, 2),
        "slice_usd": round(slice_usd, 2),
        "spy_already_held_usd": round(spy_held, 2),
        "deploy_usd": deploy_usd,
        "ticker": DEPLOY_TICKER,
        "would_buy": deploy_usd >= 100,
    }


@router.get("/simulate")
async def crash_deploy_simulate():
    """🔬 DRY-RUN — calcola cosa comprerebbe, NON esegue nulla."""
    plan = await _build_deploy_plan()
    return {"mode": "SIMULATE (no order placed)", "plan": plan, "at": datetime.utcnow().isoformat()}


@router.post("/execute")
async def crash_deploy_execute():
    """🚨 REALE — compra SPY con la fetta long-term. Solo se DEPLOY attivo."""
    plan = await _build_deploy_plan()
    if plan.get("error"):
        return plan
    if not plan["deploy_signal"]:
        return {"status": "skipped", "reason": f"Crash level {plan['crash_level']} — no deploy", "plan": plan}
    if not plan["would_buy"]:
        return {"status": "skipped", "reason": "Deploy amount too small or no cash", "plan": plan}

    # 🆕 Cooldown 3 giorni: non ricomprare in continuo durante lo stesso crash
    from datetime import timedelta
    db = get_db()
    recent = await db.crash_deploy_log.find_one(
        {"date": {"$gte": datetime.utcnow() - timedelta(days=3)}}
    )
    if recent:
        return {"status": "cooldown", "reason": "Deploy già fatto negli ultimi 3 giorni",
                "last_deploy": str(recent.get("date"))[:19], "plan": plan}

    # ESEGUE il buy notional su SPY
    from app.services.alpaca_trader import place_notional_buy
    try:
        result = await place_notional_buy(DEPLOY_TICKER, plan["deploy_usd"])
        if not result:
            return {"status": "error", "reason": "Buy order failed", "plan": plan}

        # Log
        await db.crash_deploy_log.insert_one({
            "ticker": DEPLOY_TICKER,
            "deploy_usd": plan["deploy_usd"],
            "crash_level": plan["crash_level"],
            "crash_score": plan["crash_score"],
            "spy_drawdown_pct": plan["spy_drawdown_pct"],
            "order_id": result.get("id", ""),
            "date": datetime.utcnow(),
        })

        # Telegram
        try:
            from app.services.telegram_bot import send_telegram
            await send_telegram(
                f"🚨 <b>CRASH DEPLOY ESEGUITO</b>\n\n"
                f"Comprato SPY per ${plan['deploy_usd']:,.0f}\n"
                f"Crash level: {plan['crash_level']} (score {plan['crash_score']})\n"
                f"SPY drawdown: {plan['spy_drawdown_pct']}%\n"
                f"'Compra l'inferno' 🔥"
            )
        except Exception:
            pass

        return {"status": "executed", "order_id": result.get("id", ""), "plan": plan}
    except Exception as e:
        return {"status": "error", "reason": str(e), "plan": plan}


@router.post("/load-spy-history")
async def load_spy_history(days: int = 730):
    """
    🔬 Carica storico lungo SPY (default 2 anni) in collection dedicata.
    Serve per validare il deploy scalare ai POC su crash reali (es. dazi aprile 2025).
    NON tocca stock_bars (resta a 300 barre).
    """
    import httpx
    from datetime import timedelta
    from app.config import settings

    db = get_db()
    end = datetime.utcnow() - timedelta(minutes=20)
    start = end - timedelta(days=days)

    url = "https://data.alpaca.markets/v2/stocks/SPY/bars"
    headers = {
        "APCA-API-KEY-ID": settings.ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": settings.ALPACA_SECRET_KEY,
    }
    all_bars = []
    page_token = None
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            for _ in range(10):  # max 10 pagine (paginazione Alpaca)
                params = {
                    "timeframe": "1Day",
                    "start": start.strftime("%Y-%m-%dT00:00:00Z"),
                    "end": end.strftime("%Y-%m-%dT23:59:59Z"),
                    "limit": 1000,
                    "feed": "iex",
                    "adjustment": "split",
                }
                if page_token:
                    params["page_token"] = page_token
                r = await client.get(url, headers=headers, params=params)
                if r.status_code != 200:
                    return {"error": f"Alpaca {r.status_code}: {r.text[:200]}"}
                data = r.json()
                for b in data.get("bars", []):
                    all_bars.append({
                        "date": b["t"][:10],
                        "o": b["o"], "h": b["h"], "l": b["l"], "c": b["c"], "v": b["v"],
                    })
                page_token = data.get("next_page_token")
                if not page_token:
                    break
    except Exception as e:
        return {"error": str(e)}

    if not all_bars:
        return {"error": "No bars returned"}

    all_bars.sort(key=lambda x: x["date"])
    await db.spy_longterm.update_one(
        {"_id": "SPY"},
        {"$set": {"bars": all_bars, "updated_at": datetime.utcnow(),
                  "count": len(all_bars),
                  "first": all_bars[0]["date"], "last": all_bars[-1]["date"]}},
        upsert=True,
    )
    return {
        "status": "ok",
        "bars_loaded": len(all_bars),
        "first_date": all_bars[0]["date"],
        "last_date": all_bars[-1]["date"],
    }

