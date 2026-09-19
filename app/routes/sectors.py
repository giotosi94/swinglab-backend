from fastapi import APIRouter, Query
from app.db.mongodb import get_db

router = APIRouter()

SECTOR_CODES = ["XLK", "XLF", "XLV", "XLI", "XLY", "XLP", "XLE", "XLU", "XLB", "XLRE", "XLC"]


@router.get("/")
async def get_sectors(sort_by: str = "composite_score"):
    db = get_db()
    sectors = await db.sectors.find().to_list(100)
    for sector in sectors:
        sector["_id"] = str(sector["_id"])
    sectors.sort(key=lambda item: item.get(sort_by, 0), reverse=True)
    return sectors


@router.get("/relative-strength")
async def get_sector_relative_strength(days: int = Query(60, ge=5, le=750)):
    db = get_db()
    requested = ["SPY"] + SECTOR_CODES
    docs = await db.stock_bars.find({"ticker": {"$in": requested}}, {"ticker": 1, "bars": 1}).to_list(20)
    bars_by_ticker = {doc.get("ticker"): doc.get("bars", []) for doc in docs}
    spy_bars = bars_by_ticker.get("SPY", [])

    if not spy_bars:
        return {"error": "SPY history unavailable"}

    spy_by_date = {bar.get("date"): float(bar.get("c", 0) or 0) for bar in spy_bars if bar.get("date") and bar.get("c")}
    common_dates = sorted(spy_by_date.keys())[-days:]
    series = []

    sector_docs = await db.sectors.find({"code": {"$in": SECTOR_CODES}}).to_list(20)
    sector_names = {doc.get("code"): doc.get("name", doc.get("code")) for doc in sector_docs}

    for code in SECTOR_CODES:
        sector_by_date = {
            bar.get("date"): float(bar.get("c", 0) or 0)
            for bar in bars_by_ticker.get(code, [])
            if bar.get("date") and bar.get("c")
        }
        dates = [date for date in common_dates if date in sector_by_date and spy_by_date.get(date, 0) > 0]
        if len(dates) < 2:
            continue

        first_ratio = sector_by_date[dates[0]] / spy_by_date[dates[0]]
        if first_ratio <= 0:
            continue

        points = []
        for date in dates:
            ratio = sector_by_date[date] / spy_by_date[date]
            points.append({"date": date, "value": round(ratio / first_ratio * 100, 3)})

        final_value = points[-1]["value"]
        lookback_index = max(0, len(points) - 21)
        acceleration = final_value - points[lookback_index]["value"]
        series.append({
            "code": code,
            "name": sector_names.get(code, code),
            "points": points,
            "relative_return_pct": round(final_value - 100, 2),
            "acceleration_20d": round(acceleration, 2),
            "last_value": final_value,
        })

    series.sort(key=lambda item: item["relative_return_pct"], reverse=True)

    assets = await db.assets.find(
        {"sector_code": {"$in": SECTOR_CODES}},
        {"ticker": 1, "sector_code": 1, "setup_score": 1, "setup_type": 1, "confluence": 1, "rsi": 1, "price": 1, "mtf": 1, "poc_shift": 1},
    ).to_list(400)

    best_stocks = {}
    for code in SECTOR_CODES:
        candidates = [asset for asset in assets if asset.get("sector_code") == code]
        candidates.sort(
            key=lambda asset: float(asset.get("confluence", asset.get("setup_score", 0)) or 0),
            reverse=True,
        )
        best_stocks[code] = []
        for asset in candidates[:5]:
            best_stocks[code].append({
                "ticker": asset.get("ticker"),
                "score": round(float(asset.get("confluence", asset.get("setup_score", 0)) or 0), 1),
                "setup_type": asset.get("setup_type", "neutral"),
                "rsi": round(float(asset.get("rsi", 0) or 0), 1),
                "price": round(float(asset.get("price", 0) or 0), 2),
                "weekly_trend": (asset.get("mtf") or {}).get("weekly_trend", "UNKNOWN"),
                "poc_shift": bool((asset.get("poc_shift") or {}).get("shifted_bull", False)),
            })

    return {
        "benchmark": "SPY",
        "requested_days": days,
        "executed_days": len(common_dates),
        "start_date": common_dates[0] if common_dates else None,
        "end_date": common_dates[-1] if common_dates else None,
        "baseline": 100,
        "series": series,
        "ranking": [
            {
                "rank": index + 1,
                "code": item["code"],
                "name": item["name"],
                "relative_return_pct": item["relative_return_pct"],
                "acceleration_20d": item["acceleration_20d"],
            }
            for index, item in enumerate(series)
        ],
        "best_stocks": best_stocks,
    }


@router.get("/{code}")
async def get_sector(code: str):
    db = get_db()
    sector = await db.sectors.find_one({"code": code.upper()})
    if sector:
        sector["_id"] = str(sector["_id"])
    return sector or {"error": "Sector not found"}
