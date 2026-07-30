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
