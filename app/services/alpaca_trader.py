import httpx
from app.config import settings
from app.db.mongodb import get_db
from datetime import datetime


ALPACA_BASE = "https://paper-api.alpaca.markets" if settings.ALPACA_PAPER else "https://api.alpaca.markets"
ALPACA_DATA = "https://data.alpaca.markets"

HEADERS = {
    "APCA-API-KEY-ID": settings.ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": settings.ALPACA_SECRET_KEY,
}


async def alpaca_request(method, url, json=None):
    async with httpx.AsyncClient(timeout=30) as client:
        if method == "GET":
            r = await client.get(url, headers=HEADERS)
        elif method == "POST":
            r = await client.post(url, headers=HEADERS, json=json)
        elif method == "DELETE":
            r = await client.delete(url, headers=HEADERS)
        else:
            return None
        if r.status_code in (200, 201, 204):
            return r.json() if r.text else {}
        else:
            print("Alpaca error {}: {}".format(r.status_code, r.text[:200]))
            return None


async def get_account():
    return await alpaca_request("GET", "{}/v2/account".format(ALPACA_BASE))


async def get_positions():
    return await alpaca_request("GET", "{}/v2/positions".format(ALPACA_BASE))


async def get_orders(status="all", limit=50):
    return await alpaca_request("GET", "{}/v2/orders?status={}&limit={}&direction=desc".format(ALPACA_BASE, status, limit))


async def place_order(symbol, qty, side, order_type="market", time_in_force="day", limit_price=None, stop_price=None):
    order = {
        "symbol": symbol,
        "qty": str(qty),
        "side": side,
        "type": order_type,
        "time_in_force": time_in_force,
    }
    if limit_price:
        order["limit_price"] = str(limit_price)
    if stop_price:
        order["stop_price"] = str(stop_price)
    result = await alpaca_request("POST", "{}/v2/orders".format(ALPACA_BASE), json=order)
    if result:
        db = get_db()
        await db.alpaca_orders.insert_one({
            "order_id": result.get("id"),
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "type": order_type,
            "status": result.get("status"),
            "created_at": datetime.utcnow(),
            "raw": result,
        })
    return result


async def place_bracket_order(symbol, qty, limit_price, take_profit, stop_loss):
    order = {
        "symbol": symbol,
        "qty": str(qty),
        "side": "buy",
        "type": "limit",
        "time_in_force": "gtc",
        "limit_price": str(round(limit_price, 2)),
        "order_class": "bracket",
        "take_profit": {"limit_price": str(round(take_profit, 2))},
        "stop_loss": {"stop_price": str(round(stop_loss, 2))},
    }
    return await alpaca_request("POST", "{}/v2/orders".format(ALPACA_BASE), json=order)


async def cancel_order(order_id):
    return await alpaca_request("DELETE", "{}/v2/orders/{}".format(ALPACA_BASE, order_id))


async def cancel_all_orders():
    return await alpaca_request("DELETE", "{}/v2/orders".format(ALPACA_BASE))


async def close_position(symbol):
    return await alpaca_request("DELETE", "{}/v2/positions/{}".format(ALPACA_BASE, symbol))


async def close_all_positions():
    return await alpaca_request("DELETE", "{}/v2/positions".format(ALPACA_BASE))


async def get_latest_price(symbol):
    url = "{}/v2/stocks/{}/quotes/latest".format(ALPACA_DATA, symbol)
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, headers=HEADERS)
        if r.status_code == 200:
            data = r.json()
            quote = data.get("quote", {})
            return (quote.get("ap", 0) + quote.get("bp", 0)) / 2
    return None


async def get_portfolio_history(period="1M", timeframe="1D"):
    return await alpaca_request("GET", "{}/v2/account/portfolio/history?period={}&timeframe={}".format(ALPACA_BASE, period, timeframe))


async def get_alpaca_summary():
    account = await get_account()
    positions = await get_positions()
    orders = await get_orders(status="all", limit=20)
    history = await get_portfolio_history()

    if not account:
        return {"error": "Cannot connect to Alpaca"}

    equity = float(account.get("equity", 0))
    cash = float(account.get("cash", 0))
    buying_power = float(account.get("buying_power", 0))
    portfolio_value = float(account.get("portfolio_value", 0))
    initial = float(account.get("last_equity", equity))
    pnl = equity - initial
    pnl_pct = (pnl / initial * 100) if initial > 0 else 0

    pos_list = []
    if positions:
        for p in positions:
            pos_list.append({
                "symbol": p.get("symbol"),
                "qty": int(float(p.get("qty", 0))),
                "side": p.get("side"),
                "entry_price": round(float(p.get("avg_entry_price", 0)), 2),
                "current_price": round(float(p.get("current_price", 0)), 2),
                "market_value": round(float(p.get("market_value", 0)), 2),
                "pnl": round(float(p.get("unrealized_pl", 0)), 2),
                "pnl_pct": round(float(p.get("unrealized_plpc", 0)) * 100, 2),
                "change_today": round(float(p.get("change_today", 0)) * 100, 2),
            })

    order_list = []
    if orders:
        for o in orders[:20]:
            order_list.append({
                "id": o.get("id"),
                "symbol": o.get("symbol"),
                "qty": o.get("qty"),
                "side": o.get("side"),
                "type": o.get("type"),
                "status": o.get("status"),
                "filled_avg_price": o.get("filled_avg_price"),
                "created_at": o.get("created_at"),
            })

    equity_history = []
    if history and history.get("equity"):
        timestamps = history.get("timestamp", [])
        equities = history.get("equity", [])
        pnls = history.get("profit_loss_pct", [])
        for i in range(len(timestamps)):
            if equities[i]:
                equity_history.append({
                    "date": datetime.fromtimestamp(timestamps[i]).strftime("%Y-%m-%d"),
                    "equity": round(equities[i], 2),
                    "pnl_pct": round((pnls[i] or 0) * 100, 2),
                })

    return {
        "connected": True,
        "paper": settings.ALPACA_PAPER,
        "equity": round(equity, 2),
        "cash": round(cash, 2),
        "buying_power": round(buying_power, 2),
        "portfolio_value": round(portfolio_value, 2),
        "daily_pnl": round(pnl, 2),
        "daily_pnl_pct": round(pnl_pct, 2),
        "positions": pos_list,
        "orders": order_list,
        "equity_history": equity_history,
        "account_status": account.get("status"),
    }
