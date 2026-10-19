import asyncio
from datetime import datetime, timedelta

import numpy as np

from app.config import settings

ALPACA_DATA_URL = "https://data.alpaca.markets"
ALPACA_HEADERS = {
    "APCA-API-KEY-ID": settings.ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": settings.ALPACA_SECRET_KEY,
}
HISTORY_YEARS = 10
WINDOW_3Y_WEEKS = 156
REFRESH_AFTER_DAYS = 6
INCREMENTAL_WEEKS = 10
DEFAULT_FETCH_BUDGET = 60
FETCH_CONCURRENCY = 4
CYCLE_VERSION = 1


class CycleBudget:
    def __init__(self, max_fetch=DEFAULT_FETCH_BUDGET):
        self.remaining = int(max_fetch)
        self.fetched = 0
        self.full = 0
        self.incremental = 0
        self.errors = 0

    def take(self):
        if self.remaining <= 0:
            return False
        self.remaining -= 1
        return True


def _round(value, digits=4):
    return round(float(value), digits) if value is not None and np.isfinite(value) else None


async def _fetch_weekly(client, symbol, start):
    end = datetime.utcnow() - timedelta(minutes=20)
    url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars"
    for feed in ("sip", "iex"):
        bars = []
        page_token = None
        failed = False
        while True:
            params = {
                "timeframe": "1Week",
                "start": start.strftime("%Y-%m-%dT00:00:00Z"),
                "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "limit": 10000,
                "feed": feed,
                "adjustment": "split",
                "sort": "asc",
            }
            if page_token:
                params["page_token"] = page_token
            response = await client.get(url, headers=ALPACA_HEADERS, params=params)
            if response.status_code != 200:
                failed = True
                break
            payload = response.json()
            bars.extend(payload.get("bars") or [])
            page_token = payload.get("next_page_token")
            if not page_token:
                break
        if not failed and bars:
            return bars, feed
    return [], None


def compute_cycle_static(dates, highs, lows, closes, end_index=None):
    n = len(closes) if end_index is None else end_index + 1
    if n < 26:
        return None
    h = np.asarray(highs[:n], dtype=float)
    l = np.asarray(lows[:n], dtype=float)
    hi_idx = int(np.nanargmax(h))
    lo_idx = int(np.nanargmin(l))
    after_high = l[hi_idx:]
    low_since_high_idx = hi_idx + int(np.nanargmin(after_high))
    start_3y = max(0, n - WINDOW_3Y_WEEKS)
    h3 = h[start_3y:]
    l3 = l[start_3y:]
    hi3_idx = start_3y + int(np.nanargmax(h3))
    lo3_idx = start_3y + int(np.nanargmin(l3))
    before_high = l[:hi_idx + 1]
    low_before_high = float(np.nanmin(before_high)) if len(before_high) else float(l[hi_idx])
    return {
        "version": CYCLE_VERSION,
        "weeks": int(n),
        "first_date": dates[0],
        "last_week_date": dates[n - 1],
        "years": round(n / 52.0, 1),
        "cycle_high": _round(h[hi_idx]),
        "cycle_high_date": dates[hi_idx],
        "cycle_low": _round(l[lo_idx]),
        "cycle_low_date": dates[lo_idx],
        "low_before_high": _round(low_before_high),
        "low_since_high": _round(l[low_since_high_idx]),
        "low_since_high_date": dates[low_since_high_idx],
        "weeks_since_high": int(n - 1 - hi_idx),
        "weeks_since_low_since_high": int(n - 1 - low_since_high_idx),
        "high_3y": _round(h[hi3_idx]),
        "high_3y_date": dates[hi3_idx],
        "low_3y": _round(l[lo3_idx]),
        "low_3y_date": dates[lo3_idx],
        "last_close": _round(closes[n - 1]),
    }


def cycle_position(static, price):
    if not static or not price or price <= 0:
        return None
    cycle_high = max(static["cycle_high"] or 0, price)
    cycle_low = min(static["cycle_low"] or price, price)
    high_3y = max(static["high_3y"] or 0, price)
    low_3y = min(static["low_3y"] or price, price)
    span = cycle_high - cycle_low
    span_3y = high_3y - low_3y
    low_since_high = min(static["low_since_high"] or price, price)
    prior_advance = (
        (static["cycle_high"] / static["low_before_high"] - 1) * 100
        if static.get("low_before_high") and static["low_before_high"] > 0 else None
    )
    position_pct = (price - cycle_low) / span * 100 if span > 0 else 50.0
    position_3y_pct = (price - low_3y) / span_3y * 100 if span_3y > 0 else 50.0
    drawdown_pct = (price / cycle_high - 1) * 100
    drawdown_3y_pct = (price / high_3y - 1) * 100
    decline_pct = (low_since_high / static["cycle_high"] - 1) * 100 if static["cycle_high"] else None
    recovery_pct = (price / low_since_high - 1) * 100 if low_since_high > 0 else None
    if position_pct <= 33:
        zone = "LOWER"
    elif position_pct <= 66:
        zone = "MIDDLE"
    else:
        zone = "UPPER"
    return {
        **static,
        "price": round(float(price), 4),
        "position_pct": round(position_pct, 1),
        "position_3y_pct": round(position_3y_pct, 1),
        "drawdown_from_high_pct": round(drawdown_pct, 1),
        "drawdown_from_high_3y_pct": round(drawdown_3y_pct, 1),
        "decline_high_to_low_pct": round(decline_pct, 1) if decline_pct is not None else None,
        "recovery_from_low_pct": round(recovery_pct, 1) if recovery_pct is not None else None,
        "prior_advance_pct": round(prior_advance, 1) if prior_advance is not None else None,
        "zone": zone,
    }


def _normalize(raw_bars):
    rows = {}
    for bar in raw_bars:
        date = str(bar.get("t", ""))[:10]
        if not date:
            continue
        try:
            rows[date] = (
                float(bar["o"]), float(bar["h"]), float(bar["l"]),
                float(bar["c"]), float(bar.get("v") or 0),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return rows


def _to_columns(rows):
    dates = sorted(rows)
    return {
        "d": dates,
        "o": [rows[d][0] for d in dates],
        "h": [rows[d][1] for d in dates],
        "l": [rows[d][2] for d in dates],
        "c": [rows[d][3] for d in dates],
        "v": [rows[d][4] for d in dates],
    }


async def _refresh_one(client, db, ticker, existing, budget):
    now = datetime.utcnow()
    full = existing is None or not existing.get("d")
    if full:
        start = now - timedelta(days=int(HISTORY_YEARS * 365.25) + 30)
    else:
        start = now - timedelta(weeks=INCREMENTAL_WEEKS)
    raw, feed = await _fetch_weekly(client, ticker, start)
    if not raw:
        budget.errors += 1
        await db.stock_bars_weekly.update_one(
            {"ticker": ticker},
            {"$set": {"ticker": ticker, "last_error_at": now, "updated_at": now}},
            upsert=True,
        )
        return None
    rows = {} if full else {
        d: (o, h, l, c, v)
        for d, o, h, l, c, v in zip(existing["d"], existing["o"], existing["h"], existing["l"], existing["c"], existing["v"])
    }
    rows.update(_normalize(raw))
    columns = _to_columns(rows)
    static = compute_cycle_static(columns["d"], columns["h"], columns["l"], columns["c"])
    await db.stock_bars_weekly.update_one(
        {"ticker": ticker},
        {"$set": {
            "ticker": ticker,
            **columns,
            "feed": feed or (existing or {}).get("feed"),
            "last_date": columns["d"][-1] if columns["d"] else None,
            "cycle": static,
            "updated_at": now,
            "last_error_at": None,
        }},
        upsert=True,
    )
    budget.fetched += 1
    if full:
        budget.full += 1
    else:
        budget.incremental += 1
    return static


async def get_cycle_contexts(client, db, tickers, prices, budget):
    docs = {}
    cursor = db.stock_bars_weekly.find(
        {"ticker": {"$in": list(tickers)}},
        {"ticker": 1, "cycle": 1, "updated_at": 1, "last_error_at": 1},
    )
    async for doc in cursor:
        docs[doc["ticker"]] = doc
    now = datetime.utcnow()
    to_refresh = []
    for ticker in tickers:
        doc = docs.get(ticker)
        if doc is None or not doc.get("cycle"):
            last_error = (doc or {}).get("last_error_at")
            if last_error and (now - last_error).total_seconds() < 86400:
                continue
            to_refresh.append(ticker)
        elif doc.get("updated_at") and (now - doc["updated_at"]).days >= REFRESH_AFTER_DAYS:
            to_refresh.append(ticker)
    to_refresh = [ticker for ticker in to_refresh if budget.take()]
    if to_refresh:
        semaphore = asyncio.Semaphore(FETCH_CONCURRENCY)

        async def refresh(ticker):
            async with semaphore:
                existing = None
                if docs.get(ticker, {}).get("cycle"):
                    existing = await db.stock_bars_weekly.find_one(
                        {"ticker": ticker}, {"d": 1, "o": 1, "h": 1, "l": 1, "c": 1, "v": 1, "feed": 1}
                    )
                try:
                    static = await _refresh_one(client, db, ticker, existing, budget)
                except Exception as error:
                    budget.errors += 1
                    print(f"  Cycle history error {ticker}: {error}")
                    static = None
                if static:
                    docs[ticker] = {"ticker": ticker, "cycle": static}

        await asyncio.gather(*[refresh(ticker) for ticker in to_refresh])
    contexts = {}
    for ticker in tickers:
        static = (docs.get(ticker) or {}).get("cycle")
        contexts[ticker] = cycle_position(static, prices.get(ticker)) if static else None
    return contexts


async def get_cycle_coverage(db):
    total = 0
    complete = 0
    years = []
    async for doc in db.stock_bars_weekly.find({}, {"cycle.years": 1}):
        total += 1
        value = (doc.get("cycle") or {}).get("years")
        if value:
            complete += 1
            years.append(value)
    return {
        "tickers_with_weekly": total,
        "tickers_with_cycle": complete,
        "years_min": min(years) if years else 0,
        "years_median": float(np.median(years)) if years else 0,
        "years_max": max(years) if years else 0,
    }
