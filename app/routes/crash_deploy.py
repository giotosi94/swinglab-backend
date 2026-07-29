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

    # ESEGUE il buy notional su SPY
    from app.services.alpaca_trader import place_notional_buy
    db = get_db()
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
