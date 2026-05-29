from fastapi import APIRouter
from pydantic import BaseModel
from typing import Literal
from app.db.mongodb import get_db

router = APIRouter()

class ScanRequest(BaseModel):
    universe: Literal["stocks", "sectors", "both"] = "both"
    min_score: float = 50
    top_n: int = 10

@router.post("/run")
async def run_scanner(req: ScanRequest):
    db = get_db()
    results = []

    if req.universe in ("stocks", "both"):
        assets = await db.assets.find(
            {"setup_score": {"$gte": req.min_score}}
        ).sort("setup_score", -1).to_list(req.top_n)
        for a in assets:
            results.append({
                "type": "stock",
                "symbol": a["ticker"],
                "name": a.get("name", ""),
                "sector": a.get("sector_code", ""),
                "score": a.get("setup_score", 0),
                "poc_price": a.get("poc_price"),
                "price": a.get("price", 0),
                "setup_type": a.get("setup_type"),
            })

    if req.universe in ("sectors", "both"):
        sectors = await db.sectors.find(
            {"composite_score": {"$gte": req.min_score}}
        ).sort("composite_score", -1).to_list(req.top_n)
        for s in sectors:
            results.append({
                "type": "sector",
                "symbol": s["code"],
                "name": s["name"],
                "score": s.get("composite_score", 0),
            })

    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return {"count": len(results[:req.top_n]), "results": results[:req.top_n]}
