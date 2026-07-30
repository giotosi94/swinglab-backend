"""
SPY History Loader — storico lungo di SPY per validare il crash deploy.
ISOLATO: collection separata 'spy_history', NON tocca stock_bars (il live).
"""

import httpx
from datetime import datetime, timedelta
from app.config import settings
from app.db.mongodb import get_db

ALPACA_DATA_URL = "https://data.alpaca.markets"
ALPACA_HEADERS = {
    "APCA-API-KEY-ID": settings.ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": settings.ALPACA_SECRET_KEY,
}


async def load_spy_history(years: int = 7):
    """
    Scarica ~years anni di barre daily SPY da Alpaca (sort=desc, con paginazione)
    e le salva in 'spy_history'. One-shot: lancialo una volta.
    """
    db = get_db()
    end = datetime.utcnow() - timedelta(minutes=20)
    start = end - timedelta(days=years * 365)

    all_bars = []
    page_token = None
    async with httpx.AsyncClient(timeout=30) as client:
        for _ in range(50):  # max 50 pagine di sicurezza
            params = {
                "timeframe": "1Day",
                "start": start.strftime("%Y-%m-%dT00:00:00Z"),
                "end": end.strftime("%Y-%m-%dT23:59:59Z"),
                "limit": 10000,
                "feed": "iex",
                "adjustment": "split",
            }
            if page_token:
                params["page_token"] = page_token
            try:
                r = await client.get(
                    f"{ALPACA_DATA_URL}/v2/stocks/SPY/bars",
                    headers=ALPACA_HEADERS, params=params,
                )
                if r.status_code != 200:
                    return {"error": f"Alpaca {r.status_code}: {r.text[:200]}"}
                data = r.json()
                bars = data.get("bars", [])
                for b in bars:
                    all_bars.append({
                        "date": b["t"][:10],
                        "o": b["o"], "h": b["h"], "l": b["l"], "c": b["c"], "v": b["v"],
                    })
                page_token = data.get("next_page_token")
                if not page_token:
                    break
            except Exception as e:
                return {"error": f"fetch error: {e}"}

    if not all_bars:
        return {"error": "No SPY bars returned"}

    # dedup + sort cronologico
    seen = {}
    for b in all_bars:
        seen[b["date"]] = b
    bars_sorted = sorted(seen.values(), key=lambda x: x["date"])

    await db.spy_history.delete_many({})
    await db.spy_history.insert_one({
        "ticker": "SPY",
        "bars": bars_sorted,
        "count": len(bars_sorted),
        "first": bars_sorted[0]["date"],
        "last": bars_sorted[-1]["date"],
        "loaded_at": datetime.utcnow(),
    })

    return {
        "status": "ok",
        "count": len(bars_sorted),
        "first": bars_sorted[0]["date"],
        "last": bars_sorted[-1]["date"],
    }

def _drawdown_from_peak(closes_upto):
    if len(closes_upto) < 20:
        return 0.0
    peak = max(closes_upto)
    return (closes_upto[-1] - peak) / peak * 100 if peak > 0 else 0.0


async def backtest_crash_deploy_spy(start_date: str = None, end_date: str = None,
                                    starting_capital: float = 100000):
    """
    Mini-backtest ISOLATO del crash deploy SOLO su SPY storico.
    Confronta: DEPLOY (compra le fette nei crash) vs BUY&HOLD SPY.
    """
    db = get_db()
    doc = await db.spy_history.find_one({"ticker": "SPY"})
    if not doc or not doc.get("bars"):
        return {"error": "No spy_history. Run load-spy-history first."}

    bars = doc["bars"]
    if start_date:
        bars = [b for b in bars if b["date"] >= start_date]
    if end_date:
        bars = [b for b in bars if b["date"] <= end_date]
    if len(bars) < 60:
        return {"error": "Not enough bars in range"}

    closes = [b["c"] for b in bars]
    dates = [b["date"] for b in bars]

    # ===== Strategia CRASH DEPLOY a fette =====
    cash = starting_capital
    spy_pos = {"shares": 0.0, "invested": 0.0, "fette": set(), "peak_at_entry": 0, "trimmed": False}
    events = []
    fette = [{"dd": -8, "id": 1, "frac": 0.30},
             {"dd": -15, "id": 2, "frac": 0.40},
             {"dd": -25, "id": 3, "frac": 0.30}]

    for i in range(len(bars)):
        dd = _drawdown_from_peak(closes[:i + 1])
        peak = max(closes[:i + 1])
        px = closes[i]

        for f in fette:
            if dd <= f["dd"] and f["id"] not in spy_pos["fette"]:
                deploy = cash * f["frac"]
                if deploy > 100:
                    spy_pos["shares"] += deploy / px
                    spy_pos["invested"] += deploy
                    spy_pos["fette"].add(f["id"])
                    spy_pos["peak_at_entry"] = max(spy_pos["peak_at_entry"], peak)
                    cash -= deploy
                    events.append({"date": dates[i], "action": f"FETTA_{f['id']}",
                                   "dd": round(dd, 1), "price": round(px, 2)})

        # trim 80% al ritorno sul massimo
        if (spy_pos["shares"] > 0 and not spy_pos["trimmed"]
                and px >= spy_pos["peak_at_entry"] and spy_pos["peak_at_entry"] > 0):
            sell = spy_pos["shares"] * 0.80
            cash += sell * px
            cost = spy_pos["invested"] * 0.80
            pnl = (sell * px - cost) / cost * 100 if cost > 0 else 0
            spy_pos["shares"] -= sell
            spy_pos["invested"] -= cost
            spy_pos["trimmed"] = True
            events.append({"date": dates[i], "action": "TRIM_80", "pnl_pct": round(pnl, 2)})

    # chiudi residuo a fine periodo
    final_val = cash + spy_pos["shares"] * closes[-1]
    deploy_return = (final_val - starting_capital) / starting_capital * 100

    # ===== Benchmark: BUY & HOLD SPY =====
    bh_return = (closes[-1] - closes[0]) / closes[0] * 100

    # max drawdown delle 2 strategie (semplificato: buy&hold)
    peak_bh = closes[0]
    max_dd_bh = 0
    for c in closes:
        peak_bh = max(peak_bh, c)
        max_dd_bh = max(max_dd_bh, (peak_bh - c) / peak_bh * 100)

    return {
        "period": {"start": dates[0], "end": dates[-1], "days": len(bars)},
        "crash_deploy": {
            "return_pct": round(deploy_return, 2),
            "final_value": round(final_val, 2),
            "events": events,
            "n_deploys": len([e for e in events if "FETTA" in e["action"]]),
        },
        "buy_and_hold": {
            "return_pct": round(bh_return, 2),
            "max_drawdown_pct": round(max_dd_bh, 2),
        },
        "verdict": "DEPLOY meglio" if deploy_return > bh_return else "BUY&HOLD meglio",
        "difference_pct": round(deploy_return - bh_return, 2),
    }
