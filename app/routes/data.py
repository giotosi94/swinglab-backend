from fastapi import APIRouter
from app.services.data_fetcher import fetch_and_analyze_sectors, fetch_and_analyze_stocks

router = APIRouter()

@router.post("/refresh/sectors")
async def refresh_sectors():
    results = await fetch_and_analyze_sectors()
    return {"message": "Sectors updated", "count": len(results)}

@router.post("/refresh/stocks")
async def refresh_stocks():
    results = await fetch_and_analyze_stocks()
    return {"message": "Stocks updated", "count": len(results)}

@router.post("/refresh/all")
async def refresh_all():
    sectors = await fetch_and_analyze_sectors()
    stocks = await fetch_and_analyze_stocks()
    return {
        "message": "Full refresh completed",
        "sectors": len(sectors),
        "stocks": len(stocks),
    }
