from fastapi import APIRouter, Query
from typing import Optional
from app.db.mongodb import get_db

router = APIRouter()

# Ticker esclusi dall'universo trade (indici/ETF macro + residui ricerche)
EXCLUDED_TICKERS = [
    "SPY", "QQQ", "IWM", "DIA", "VXX", "VIXY", "TLT", "HYG",
    "LQD", "GLD", "USO", "RSP", "IWO", "FXE", "UUP", "EEM", "IYT",
]


@router.get("/")
async def get_assets(
    sector: Optional[str] = None,
    min_score: float = 0,
    sort_by: str = Query("setup_score"),
    limit: int = 50
):
    db = get_db()
    query = {
        "sector_code": {"$ne": "SEARCH"},
        "ticker": {"$nin": EXCLUDED_TICKERS},
    }
    if sector:
        query["sector_code"] = sector.upper()
    if min_score > 0:
        query["setup_score"] = {"$gte": min_score}

    projection = {
        "price_history": 0,
        "vp_distribution": 0,
        "multi_tf_vp": 0,
        "fvg": 0,
        "history": 0,
    }

    assets = await db.assets.find(query, projection).to_list(limit)
    for a in assets:
        a["_id"] = str(a["_id"])
    assets.sort(key=lambda x: x.get(sort_by, 0), reverse=True)
    return assets


@router.get("/{ticker}")
async def get_asset(ticker: str):
    db = get_db()
    asset = await db.assets.find_one({"ticker": ticker.upper()})
    if asset:
        asset["_id"] = str(asset["_id"])
    return asset or {"error": "Asset not found"}
