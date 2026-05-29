from fastapi import APIRouter
from app.db.mongodb import get_db

router = APIRouter()

@router.get("/")
async def get_sectors(sort_by: str = "composite_score"):
    db = get_db()
    sectors = await db.sectors.find().to_list(100)
    for s in sectors:
        s["_id"] = str(s["_id"])
    sectors.sort(key=lambda x: x.get(sort_by, 0), reverse=True)
    return sectors

@router.get("/{code}")
async def get_sector(code: str):
    db = get_db()
    sector = await db.sectors.find_one({"code": code.upper()})
    if sector:
        sector["_id"] = str(sector["_id"])
    return sector or {"error": "Sector not found"}
