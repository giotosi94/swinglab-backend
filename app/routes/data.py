from fastapi import APIRouter, Query
from app.services.data_fetcher import fetch_and_analyze_sectors, fetch_and_analyze_stocks
from app.services.stock_search import search_and_analyze_stock
from app.services.auto_trader import run_auto_trader, reset_auto_trader, get_auto_trader_state

router = APIRouter()

@router.post("/refresh/sectors")
async def refresh_sectors():
    results = await fetch_and_analyze_sectors()
    return {"message": "Sectors updated", "count": len(results)}

@router.post("/refresh/stocks")
async def refresh_stocks():
    results = await fetch_and_analyze_stocks()
    # Run auto-trader after stock refresh
    trader_result = await run_auto_trader()
    return {"message": "Stocks updated", "count": len(results), "auto_trader": trader_result}

@router.post("/refresh/all")
async def refresh_all():
    sectors = await fetch_and_analyze_sectors()
    stocks = await fetch_and_analyze_stocks()
    trader_result = await run_auto_trader()
    return {"message": "Full refresh completed", "sectors": len(sectors), "stocks": len(stocks), "auto_trader": trader_result}

@router.get("/search/{ticker}")
async def search_stock(ticker: str):
    result = await search_and_analyze_stock(ticker.upper())
    if result:
        return result
    return {"error": f"Could not find data for {ticker.upper()}"}

@router.get("/autotrader")
async def get_trader():
    state = await get_auto_trader_state()
    if state:
        return state
    return {"error": "Auto-trader not initialized. Run a stock refresh first."}

@router.post("/autotrader/run")
async def run_trader():
    result = await run_auto_trader()
    return result

@router.post("/autotrader/reset")
async def reset_trader(capital: float = Query(default=10000)):
    state = await reset_auto_trader(capital)
    state["_id"] = str(state["_id"])
    return {"message": f"Auto-trader reset with ${capital}", "state": state}
