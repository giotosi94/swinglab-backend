from fastapi import APIRouter, Query
from app.services.data_fetcher import fetch_and_analyze_sectors, fetch_and_analyze_stocks
from app.services.stock_search import search_and_analyze_stock
from app.services.auto_trader import run_auto_trader, reset_auto_trader, get_auto_trader_state
from app.services.alpaca_trader import (
    get_alpaca_summary, place_order, place_bracket_order,
    cancel_order, close_position, close_all_positions, get_account
)
from app.db.mongodb import get_db

router = APIRouter()

@router.post("/refresh/sectors")
async def refresh_sectors():
    results = await fetch_and_analyze_sectors()
    return {"message": "Sectors updated", "count": len(results)}

@router.post("/refresh/stocks")
async def refresh_stocks():
    results = await fetch_and_analyze_stocks()
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
    return {"error": "Could not find data for {}".format(ticker.upper())}

@router.get("/autotrader")
async def get_trader():
    state = await get_auto_trader_state()
    if state:
        return state
    return {"error": "Auto-trader not initialized"}

@router.post("/autotrader/run")
async def run_trader():
    return await run_auto_trader()

@router.post("/autotrader/reset")
async def reset_trader(capital: float = Query(default=10000)):
    state = await reset_auto_trader(capital)
    state["_id"] = str(state["_id"])
    return {"message": "Reset with ${}".format(capital), "state": state}

@router.get("/market")
async def get_market_data():
    db = get_db()
    spy = await db.market_regime.find_one({"symbol": "SPY"})
    vix = await db.market_regime.find_one({"symbol": "VIX"})
    if spy: spy["_id"] = str(spy["_id"])
    if vix: vix["_id"] = str(vix["_id"])
    return {"spy": spy, "vix": vix}
@router.get("/live")
async def live_prices():
    from app.services.alpaca_trader import get_live_prices
    db = get_db()
    assets = await db.assets.find({}, {"ticker": 1}).to_list(200)
    symbols = [a["ticker"] for a in assets if a.get("ticker")]
    # Alpaca max 50 per call, so batch
    all_prices = {}
    for i in range(0, len(symbols), 50):
        batch = symbols[i:i+50]
        prices = await get_live_prices(batch)
        all_prices.update(prices)
    return all_prices
@router.get("/alpaca")
async def alpaca_summary():
    return await get_alpaca_summary()

@router.post("/alpaca/buy")
async def alpaca_buy(symbol: str, qty: int = 1):
    result = await place_order(symbol.upper(), qty, "buy")
    return result or {"error": "Order failed"}

@router.post("/alpaca/sell")
async def alpaca_sell(symbol: str, qty: int = 1):
    result = await place_order(symbol.upper(), qty, "sell")
    return result or {"error": "Order failed"}

@router.post("/alpaca/bracket")
async def alpaca_bracket(symbol: str, qty: int = 1, entry: float = 0, target: float = 0, stop: float = 0):
    result = await place_bracket_order(symbol.upper(), qty, entry, target, stop)
    return result or {"error": "Bracket order failed"}

@router.post("/alpaca/close/{symbol}")
async def alpaca_close(symbol: str):
    result = await close_position(symbol.upper())
    return result or {"error": "Close failed"}

@router.post("/alpaca/close-all")
async def alpaca_close_all():
    result = await close_all_positions()
    return result or {"error": "Close all failed"}

@router.delete("/alpaca/order/{order_id}")
async def alpaca_cancel(order_id: str):
    result = await cancel_order(order_id)
    return result or {"error": "Cancel failed"}
