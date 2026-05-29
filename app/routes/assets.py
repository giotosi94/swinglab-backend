from fastapi import APIRouter, Query
from typing import Optional
from app.db.mongodb import get_db

router = APIRouter()

@router.get("/")
async def get_assets(
    sector: Optional[str] = None,
    min_score: float = 0,
    sort_by: str = Query("setup_score"),
    limit: int = 50
):
    db = get_db()
    query = {}
    if sector:
        query["sector_code"] = sector.upper()
    if min_score > 0:
        query["setup_score"] = {"$gte": min_score}

    assets = await db.assets.find(query).to_list(limit)
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
