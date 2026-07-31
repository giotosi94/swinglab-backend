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

async def load_sectors_history(years: int = 7):
    """
    Scarica storico lungo degli 11 ETF settoriali da Alpaca.
    Collection separata 'sectors_history'. One-shot.
    """
    db = get_db()
    sectors = ["XLE","XLC","XLI","XLU","XLF","XLK","XLRE","XLY","XLB","XLP","XLV"]
    end = datetime.utcnow() - timedelta(minutes=20)
    start = end - timedelta(days=years * 365)
    saved = {}

    async with httpx.AsyncClient(timeout=30) as client:
        for etf in sectors:
            all_bars = []
            page_token = None
            for _ in range(50):
                params = {
                    "timeframe": "1Day",
                    "start": start.strftime("%Y-%m-%dT00:00:00Z"),
                    "end": end.strftime("%Y-%m-%dT23:59:59Z"),
                    "limit": 10000, "feed": "iex", "adjustment": "split",
                }
                if page_token:
                    params["page_token"] = page_token
                try:
                    r = await client.get(
                        f"{ALPACA_DATA_URL}/v2/stocks/{etf}/bars",
                        headers=ALPACA_HEADERS, params=params,
                    )
                    if r.status_code != 200:
                        break
                    data = r.json()
                    for b in data.get("bars", []):
                        all_bars.append({
                            "date": b["t"][:10],
                            "o": b["o"], "h": b["h"], "l": b["l"], "c": b["c"], "v": b["v"],
                        })
                    page_token = data.get("next_page_token")
                    if not page_token:
                        break
                except Exception:
                    break

            if all_bars:
                seen = {b["date"]: b for b in all_bars}
                bars_sorted = sorted(seen.values(), key=lambda x: x["date"])
                await db.sectors_history.update_one(
                    {"ticker": etf},
                    {"$set": {"ticker": etf, "bars": bars_sorted,
                              "count": len(bars_sorted),
                              "first": bars_sorted[0]["date"],
                              "last": bars_sorted[-1]["date"],
                              "loaded_at": datetime.utcnow()}},
                    upsert=True,
                )
                saved[etf] = len(bars_sorted)

    return {"status": "ok", "sectors": saved}


def _rotation_signal_from_closes(closes):
    """Rotazione (metodo Rea) da array di chiusure fino a una data."""
    if len(closes) < 126:
        return "NEUTRAL", 0
    c_now = closes[-1]
    r3 = c_now / closes[-63] - 1
    r6 = c_now / closes[-126] - 1
    ann3 = ((1 + r3) ** (252/63) - 1) * 100 if r3 > -1 else 0
    ann6 = ((1 + r6) ** (252/126) - 1) * 100 if r6 > -1 else 0
    accel = ann3 - ann6
    last20 = closes[-20:]
    mean20 = sum(last20) / len(last20)
    compr = (max(last20) - min(last20)) / mean20 if mean20 > 0 else 0.5
    if accel > 5 and compr < 0.07:
        return "EXPLOSIVE", accel
    elif accel > 5:
        return "ROTATING_IN", accel
    elif accel < -8:
        return "ROTATING_OUT", accel
    return "NEUTRAL", accel


async def backtest_sector_rotation(start_date: str, end_date: str,
                                   starting_capital: float = 100000):
    """
    Mini-backtest ISOLATO: rotazione settoriale sui soli ETF.
    Strategia: ogni mese, tieni gli ETF in EXPLOSIVE/ROTATING_IN (equipesati),
    evita ROTATING_OUT. Confronta vs equal-weight di tutti gli 11 ETF (buy&hold).
    """
    db = get_db()
    docs = await db.sectors_history.find({}).to_list(20)
    if not docs:
        return {"error": "No sectors_history. Run load-sectors-history first."}

    # closes per ETF, filtrati per periodo, indicizzati per data
    etf_data = {}
    all_dates = set()
    for d in docs:
        bars = [b for b in d["bars"] if start_date <= b["date"] <= end_date]
        if len(bars) < 130:
            continue
        etf_data[d["ticker"]] = {b["date"]: b["c"] for b in bars}
        etf_data[d["ticker"]]["_ordered"] = [b["c"] for b in sorted(bars, key=lambda x: x["date"])]
        etf_data[d["ticker"]]["_dates"] = [b["date"] for b in sorted(bars, key=lambda x: x["date"])]
        all_dates.update(b["date"] for b in bars)

    if not etf_data:
        return {"error": "Not enough data in range"}

    dates = sorted(all_dates)
    # ribilancio mensile (ogni ~21 giorni)
    rebal_idx = list(range(126, len(dates), 21))

    # Strategia ROTAZIONE
    cash_rot = starting_capital
    holdings_rot = {}  # etf -> shares
    equity_rot = []

    for i, date in enumerate(dates):
        # valore corrente
        val = cash_rot + sum(
            sh * etf_data[etf].get(date, etf_data[etf]["_ordered"][-1])
            for etf, sh in holdings_rot.items()
        )
        equity_rot.append(val)

        if i in rebal_idx:
            # calcola segnali a questa data
            chosen = []
            for etf, data in etf_data.items():
                if date not in data:
                    continue
                d_idx = data["_dates"].index(date) if date in data["_dates"] else -1
                if d_idx < 126:
                    continue
                closes_upto = data["_ordered"][:d_idx + 1]
                sig, _ = _rotation_signal_from_closes(closes_upto)
                if sig in ("EXPLOSIVE", "ROTATING_IN"):
                    chosen.append(etf)
            # vendi tutto
            for etf, sh in holdings_rot.items():
                cash_rot += sh * etf_data[etf].get(date, etf_data[etf]["_ordered"][-1])
            holdings_rot = {}
            # compra equipesato i chosen
            if chosen:
                per = cash_rot / len(chosen)
                for etf in chosen:
                    px = etf_data[etf].get(date)
                    if px:
                        holdings_rot[etf] = per / px
                        cash_rot -= per

    final_rot = cash_rot + sum(
        sh * etf_data[etf]["_ordered"][-1] for etf, sh in holdings_rot.items()
    )
    rot_return = (final_rot - starting_capital) / starting_capital * 100

    # Benchmark: equal-weight buy&hold di tutti gli 11 ETF
    bh_returns = []
    for etf, data in etf_data.items():
        o = data["_ordered"]
        bh_returns.append((o[-1] - o[0]) / o[0] * 100)
    bh_return = sum(bh_returns) / len(bh_returns) if bh_returns else 0

    return {
        "period": {"start": dates[0], "end": dates[-1], "days": len(dates)},
        "rotation": {"return_pct": round(rot_return, 2), "final": round(final_rot, 2)},
        "equal_weight_hold": {"return_pct": round(bh_return, 2)},
        "difference_pct": round(rot_return - bh_return, 2),
        "verdict": "ROTATION meglio" if rot_return > bh_return else "HOLD meglio",
    }
