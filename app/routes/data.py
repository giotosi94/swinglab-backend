from fastapi import APIRouter
from app.services.data_fetcher import fetch_and_analyze_sectors, fetch_and_analyze_stocks
from app.services.stock_search import search_and_analyze_stock
from app.services.notifications import send_daily_briefing, send_telegram

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

@router.post("/notify")
async def send_notifications():
    msg = await send_daily_briefing()
    if msg:
        return {"message": "Notification sent", "preview": msg[:200]}
    return {"error": "Failed to send notification"}

@router.post("/notify/test")
async def test_notification():
    ok = await send_telegram("SwingLab test notification - everything works!")
    return {"message": "Test sent" if ok else "Failed"}
