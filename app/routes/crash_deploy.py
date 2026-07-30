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

# 🆕 Deploy SCALARE a fette (validato su crash reale dazi 2025: +32.84% vs 0% del secco)
# Ogni livello di drawdown SPY toccato → deploya la sua fetta della riserva long-term.
SCALAR_LEVELS = [
    {"level": "L1", "dd": -8,  "fraction": 0.25},
    {"level": "L2", "dd": -13, "fraction": 0.25},
    {"level": "L3", "dd": -18, "fraction": 0.25},
    {"level": "L4", "dd": -25, "fraction": 0.25},
]
LEVEL_COOLDOWN_DAYS = 30   # stesso livello non si ri-deploya entro l'episodio


async def _build_deploy_plan():
    """Deploy SCALARE: legge il drawdown SPY e determina quali livelli deployare."""
    db = get_db()
    ctx = await db.market_context.find_one({"_id": "latest"})
    if not ctx:
        return {"error": "No market context. Run macro first."}

    cr = ctx.get("crash_radar", {})
    dd = cr.get("spy_drawdown_pct", 0)  # <= 0

    from app.services.alpaca_trader import get_account, get_positions
    account = await get_account()
    equity = float(account.get("equity", 0)) if account else 0
    cash = float(account.get("cash", 0)) if account else 0
    slice_usd = equity * (LONGTERM_SLICE_PCT / 100)

    # Livelli ATTIVI (drawdown li ha toccati) e NON già deployati di recente
    from datetime import timedelta
    cutoff = datetime.utcnow() - timedelta(days=LEVEL_COOLDOWN_DAYS)
    recent_logs = await db.crash_deploy_log.find(
        {"date": {"$gte": cutoff}}
    ).to_list(50)
    already_done = {log.get("level") for log in recent_logs}

    pending = []
    for lvl in SCALAR_LEVELS:
        if dd <= lvl["dd"] and lvl["level"] not in already_done:
            amt = round(slice_usd * lvl["fraction"], 2)
            pending.append({"level": lvl["level"], "dd_threshold": lvl["dd"], "usd": amt})

    total_pending = sum(p["usd"] for p in pending)
    total_pending = min(total_pending, cash * 0.95)
    if total_pending < 0:
        total_pending = 0

    positions = await get_positions() or []
    spy_held = sum(float(p.get("market_value", 0)) for p in positions if p.get("symbol") == DEPLOY_TICKER)

    return {
        "spy_drawdown_pct": dd,
        "vixy": cr.get("vixy_price", 0),
        "equity": round(equity, 2),
        "cash": round(cash, 2),
        "slice_usd": round(slice_usd, 2),
        "spy_already_held_usd": round(spy_held, 2),
        "pending_levels": pending,
        "deploy_usd": round(total_pending, 2),
        "levels_already_done": list(already_done),
        "ticker": DEPLOY_TICKER,
        "would_buy": total_pending >= 100 and len(pending) > 0,
    }


@router.get("/simulate")
async def crash_deploy_simulate():
    """🔬 DRY-RUN — calcola cosa comprerebbe, NON esegue nulla."""
    plan = await _build_deploy_plan()
    return {"mode": "SIMULATE (no order placed)", "plan": plan, "at": datetime.utcnow().isoformat()}


@router.post("/execute")
async def crash_deploy_execute():
    """🚨 REALE — deploy SCALARE: compra SPY per ogni livello di drawdown toccato."""
    plan = await _build_deploy_plan()
    if plan.get("error"):
        return plan
    if not plan["would_buy"]:
        return {"status": "skipped",
                "reason": f"Nessun livello nuovo da deployare (dd {plan['spy_drawdown_pct']}%)",
                "plan": plan}

    from app.services.alpaca_trader import place_notional_buy
    from app.services.telegram_bot import send_telegram
    db = get_db()
    executed = []

    for lvl in plan["pending_levels"]:
        usd = lvl["usd"]
        if usd < 100:
            continue
        try:
            result = await place_notional_buy(DEPLOY_TICKER, usd)
            if not result:
                continue
            await db.crash_deploy_log.insert_one({
                "ticker": DEPLOY_TICKER,
                "level": lvl["level"],
                "dd_threshold": lvl["dd_threshold"],
                "deploy_usd": usd,
                "spy_drawdown_pct": plan["spy_drawdown_pct"],
                "order_id": result.get("id", ""),
                "date": datetime.utcnow(),
            })
            executed.append({"level": lvl["level"], "usd": usd, "order_id": result.get("id", "")})
        except Exception as e:
            print(f"  Crash deploy {lvl['level']} error: {e}")

    if not executed:
        return {"status": "error", "reason": "Nessun ordine eseguito", "plan": plan}

    try:
        lines = "\n".join(f"  {e['level']}: ${e['usd']:,.0f}" for e in executed)
        await send_telegram(
            f"🚨 <b>CRASH DEPLOY SCALARE</b>\n\n"
            f"SPY drawdown {plan['spy_drawdown_pct']}%\n"
            f"Fette deployate:\n{lines}\n"
            f"'Compra l'inferno a fette' 🔥"
        )
    except Exception:
        pass

    return {"status": "executed", "deployed": executed, "plan": plan}


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

@router.get("/backtest-scalar")
async def backtest_scalar_deploy(fwd_days: int = 180):
    """
    🔬 Valida il deploy SCALARE ai livelli vs SECCO vs BUY-ALL, su storico SPY reale.
    Per ogni crash trovato: simula le 3 strategie e confronta il rendimento forward.
    """
    db = get_db()
    doc = await db.spy_longterm.find_one({"_id": "SPY"})
    if not doc or not doc.get("bars"):
        return {"error": "No SPY longterm data. Run /load-spy-history first."}

    bars = doc["bars"]
    closes = [b["c"] for b in bars]
    dates = [b["date"] for b in bars]
    n = len(bars)

    # Livelli di deploy scalare (drawdown dal picco → % della fetta da deployare)
    SCALAR_LEVELS = [
        {"dd": -8,  "fraction": 0.25},
        {"dd": -13, "fraction": 0.25},
        {"dd": -18, "fraction": 0.25},
        {"dd": -25, "fraction": 0.25},
    ]
    SECCO_DD = -20  # deploy secco: tutto a -20%

    # Trova gli EPISODI di crash: drawdown dal max mobile a 252g che scende sotto -8%
    episodes = []
    i = 60
    while i < n:
        window = closes[max(0, i - 252):i + 1]
        peak = max(window)
        dd = (closes[i] - peak) / peak * 100
        if dd <= -8:
            # inizio episodio: trova il bottom e traccia i deploy scalari
            start_i = i
            peak_ref = peak
            filled_levels = []
            slice_capital = 10000.0  # fetta simulata $10k
            scalar_shares = 0.0
            scalar_spent = 0.0
            secco_shares = 0.0
            secco_spent = 0.0
            j = i
            while j < n:
                w = closes[max(0, j - 252):j + 1]
                pk = max(w)
                d = (closes[j] - pk) / pk * 100
                # deploy scalare: riempi i livelli toccati
                for lvl in SCALAR_LEVELS:
                    if d <= lvl["dd"] and lvl["dd"] not in filled_levels:
                        amt = slice_capital * lvl["fraction"]
                        scalar_shares += amt / closes[j]
                        scalar_spent += amt
                        filled_levels.append(lvl["dd"])
                # deploy secco: tutto a -20%
                if d <= SECCO_DD and secco_spent == 0:
                    secco_shares = slice_capital / closes[j]
                    secco_spent = slice_capital
                # fine episodio: risalito sopra -5% dal picco
                if d > -5 and j > start_i + 5:
                    break
                j += 1

            bottom_idx = min(range(start_i, min(j + 1, n)), key=lambda k: closes[k])
            # rendimento forward dal bottom
            fwd_idx = min(bottom_idx + fwd_days, n - 1)
            price_at_fwd = closes[fwd_idx]

            # BUY-ALL: compra tutto all'inizio dell'episodio
            buyall_shares = slice_capital / closes[start_i]

            def value(shares, spent):
                if spent == 0:
                    return None
                return round((shares * price_at_fwd - spent) / spent * 100, 2)

            episodes.append({
                "start_date": dates[start_i],
                "bottom_date": dates[bottom_idx],
                "max_drawdown_pct": round((closes[bottom_idx] - peak_ref) / peak_ref * 100, 2),
                "levels_filled": filled_levels,
                "fwd_days": fwd_days,
                "return_scalar_pct": value(scalar_shares, scalar_spent),
                "return_secco_pct": value(secco_shares, secco_spent),
                "return_buyall_pct": value(buyall_shares, slice_capital),
                "scalar_capital_deployed": round(scalar_spent, 0),
                "secco_capital_deployed": round(secco_spent, 0),
            })
            i = j + 20  # salta oltre l'episodio
        else:
            i += 1

    return {
        "spy_period": {"first": dates[0], "last": dates[-1], "bars": n},
        "episodes_found": len(episodes),
        "episodes": episodes,
        "scalar_levels": SCALAR_LEVELS,
    }
