from fastapi import APIRouter
from app.services.data_fetcher import fetch_and_analyze_sectors, fetch_and_analyze_stocks
from app.services.stock_search import search_and_analyze_stock

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

@router.get("/search/{ticker}")
async def search_stock(ticker: str):
    result = await search_and_analyze_stock(ticker.upper())
    if result:
        return result
    return {"error": f"Could not find data for {ticker.upper()}"}

@router.get("/regime")
async def get_regime():
    from app.db.mongodb import get_db
    db = get_db()
    spy = await db.market_regime.find_one({"symbol": "SPY"})
    if spy:
        spy["_id"] = str(spy["_id"])
    return spy or {"error": "No regime data"}

@router.post("/notify")
async def send_notifications():
    from app.services.notifications import check_and_notify
    result = await check_and_notify()
    return result
