from fastapi import APIRouter, Query
from app.services.data_fetcher import fetch_and_analyze_sectors, fetch_and_analyze_stocks
from app.services.stock_search import search_and_analyze_stock
from app.services.auto_trader import run_auto_trader, reset_auto_trader, get_auto_trader_state
from app.services.alpaca_trader import (
    get_alpaca_summary, place_order, place_bracket_order,
    cancel_order, close_position, close_all_positions, get_account,
    get_live_prices, get_portfolio_periods, cancel_all_orders
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
    return {"message": "Reset with ${}".format(capital)}


@router.get("/market")
async def get_market_data():
    db = get_db()
    symbols = [
        "SPY", "QQQ", "IWM", "DIA",
        "VIXY", "VXX",
        "TLT", "HYG", "LQD",
        "GLD", "USO",
        "RSP", "IWO",
        "FXE", "UUP",
        "EEM",
        "IYT",
        "BTC/USD", "ETH/USD",
    ]
    result = {}
    for sym in symbols:
        doc = await db.market_regime.find_one({"symbol": sym})
        if doc:
            doc["_id"] = str(doc["_id"])
            result[sym] = doc
    return result


@router.get("/live")
async def live_prices():
    db = get_db()
    assets = await db.assets.find({}, {"ticker": 1}).to_list(300)
    symbols = [a["ticker"] for a in assets if a.get("ticker")]
    all_prices = {}
    for i in range(0, len(symbols), 50):
        batch = symbols[i:i+50]
        prices = await get_live_prices(batch)
        all_prices.update(prices)
    return all_prices


@router.get("/agent/brain")
async def get_brain():
    db = get_db()
    params = await db.agent_memory_alpha_strategist.find_one({"_id": "params"})
    if params:
        params["_id"] = str(params["_id"])
        return params
    old_params = await db.agent_brain.find_one({"_id": "learned_params"})
    if old_params:
        old_params["_id"] = str(old_params["_id"])
        return old_params
    return {"min_confluence": 35, "max_rsi_entry": 68, "best_setups": ["pullback_to_poc", "ema_bounce", "breakout"], "total_trades": 0}


@router.get("/agent/decisions")
async def get_decisions():
    db = get_db()
    all_decisions = []
    for agent_name in ["macro_analyst", "alpha_strategist", "risk_manager", "executor"]:
        col_name = f"agent_decisions_{agent_name}"
        decisions = await db[col_name].find().sort("created_at", -1).to_list(15)
        for d in decisions:
            d["_id"] = str(d["_id"])
            d["agent"] = agent_name
        all_decisions.extend(decisions)
    all_decisions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return all_decisions[:50]


@router.get("/alpaca")
async def alpaca_summary():
    return await get_alpaca_summary()


@router.get("/alpaca/history")
async def alpaca_history():
    return await get_portfolio_periods()


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


@router.delete("/alpaca/orders-all")
async def alpaca_cancel_all_orders():
    result = await cancel_all_orders()
    return result or {"message": "All orders cancelled"}


@router.delete("/alpaca/order/{order_id}")
async def alpaca_cancel(order_id: str):
    result = await cancel_order(order_id)
    return result or {"error": "Cancel failed"}


@router.delete("/reset-bars")
async def reset_all_bars():
    db = get_db()
    result = await db.stock_bars.delete_many({})
    return {"deleted": result.deleted_count, "message": "All bars deleted. Next refresh will re-download."}


@router.get("/test-bars/{symbol}")
async def test_bars(symbol: str):
    from app.services.data_fetcher import fetch_bars_from_api
    import httpx
    async with httpx.AsyncClient(timeout=15) as client:
        bars = await fetch_bars_from_api(client, symbol, limit=5)
        if bars:
            return {"count": len(bars), "last": bars[-1]["t"][:10], "bars": bars}
        return {"count": 0, "error": "No bars returned"}


@router.delete("/trades/{trade_id}")
async def delete_trade(trade_id: str):
    from bson import ObjectId
    db = get_db()
    result = await db.trade_history.delete_one({"_id": ObjectId(trade_id)})
    return {"deleted": result.deleted_count}


@router.get("/news/{symbol}")
async def get_stock_news(symbol: str):
    from app.services.news_service import get_stock_news_with_sentiment
    return await get_stock_news_with_sentiment(symbol.upper())


@router.get("/benchmark/spy")
async def get_spy_benchmark(period: str = "1M"):
    """SPY performance matching the selected period."""
    db = get_db()
    spy_bars = await db.stock_bars.find_one({"ticker": "SPY"})
    if not spy_bars or not spy_bars.get("bars"):
        return {"error": "No SPY data"}

    bars = spy_bars["bars"]

    # Filter bars by period
    period_days = {"1D": 1, "1W": 7, "1M": 30, "3M": 90, "6M": 180, "1Y": 365, "YTD": 365}
    days = period_days.get(period, 30)

    if len(bars) > days:
        bars = bars[-days:]

    points = []
    if bars:
        start_price = bars[0]["c"]
        for b in bars:
            pct = round(((b["c"] - start_price) / start_price) * 100, 2)
            points.append({
                "date": b["date"],
                "price": round(b["c"], 2),
                "pct_change": pct,
            })

    return {
        "ticker": "SPY",
        "period": period,
        "points": points,
        "total_return": points[-1]["pct_change"] if points else 0,
        "current_price": points[-1]["price"] if points else 0,
    }
