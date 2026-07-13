import httpx
import asyncio
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


# ============================================
# ACCOUNT & POSITIONS
# ============================================

async def get_account():
    return await alpaca_request("GET", "{}/v2/account".format(ALPACA_BASE))


async def get_positions():
    return await alpaca_request("GET", "{}/v2/positions".format(ALPACA_BASE))


async def get_orders(status="all", limit=50, nested=True):
    url = "{}/v2/orders?status={}&limit={}&direction=desc&nested={}".format(
        ALPACA_BASE, status, limit, str(nested).lower()
    )
    return await alpaca_request("GET", url)


# ============================================
# 🆕 ASSET INFO (per check fractionable)
# ============================================

async def get_asset_info(symbol: str):
    """
    🆕 Ritorna info su un asset Alpaca, incluso il flag fractionable.
    Usato dal RiskManager per filtrare titoli non frazionabili.
    """
    return await alpaca_request("GET", "{}/v2/assets/{}".format(ALPACA_BASE, symbol))


async def is_fractionable(symbol: str) -> bool:
    """🆕 Check rapido se un titolo supporta fractional shares."""
    info = await get_asset_info(symbol)
    if info:
        return bool(info.get("fractionable", False))
    return False


# ============================================
# ORDERS — Place (modificato per supportare qty float)
# ============================================

async def place_order(symbol, qty, side, order_type="market", time_in_force="day", limit_price=None, stop_price=None):
    """
    🔧 Modificato: qty può essere float (per fractional).
    """
    order = {
        "symbol": symbol,
        "qty": str(qty),  # Alpaca accetta stringhe sia per int che per float
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
            "qty": float(qty),  # 🔧 salvato come float
            "side": side,
            "type": order_type,
            "status": result.get("status"),
            "created_at": datetime.utcnow(),
            "raw": result,
        })
    return result


# ============================================
# 🆕 NOTIONAL BUY (acquisto in dollari, supporta fractional)
# ============================================

async def place_notional_buy(symbol: str, notional_usd: float, time_in_force: str = "day"):
    """
    🆕 Piazza un BUY market notional (in dollari) — supporta fractional shares.
    
    Esempio: place_notional_buy("LIN", 2000) compra $2.000 di LIN
    (potrebbe risultare in 3.809 shares se LIN vale $525).
    
    NOTA: Alpaca permette notional SOLO con:
    - type=market
    - time_in_force=day
    - NO order_class (no bracket diretto)
    """
    order = {
        "symbol": symbol,
        "notional": str(round(notional_usd, 2)),
        "side": "buy",
        "type": "market",
        "time_in_force": time_in_force,
    }
    
    result = await alpaca_request("POST", "{}/v2/orders".format(ALPACA_BASE), json=order)
    
    if result:
        db = get_db()
        await db.alpaca_orders.insert_one({
            "order_id": result.get("id"),
            "symbol": symbol,
            "notional": float(notional_usd),
            "side": "buy",
            "type": "market_notional",
            "status": result.get("status"),
            "created_at": datetime.utcnow(),
            "raw": result,
        })
        print(f"  💵 NOTIONAL BUY {symbol}: ${notional_usd:.2f} (id={result.get('id', '')[:8]})")
    else:
        print(f"  ❌ NOTIONAL BUY {symbol} FAILED: ${notional_usd:.2f}")
    
    return result


# ============================================
# 🆕 WAIT FOR FILL (polling)
# ============================================

async def wait_for_fill(order_id: str, timeout_sec: int = 15, poll_interval: float = 1.0):
    """
    🆕 Polla un ordine fino a fill (o timeout).
    
    Ritorna dict con:
    - filled: bool
    - filled_qty: float (quantità effettivamente acquistata, frazionaria)
    - filled_avg_price: float (prezzo medio di esecuzione)
    - status: str
    - order: dict completo
    
    Necessario dopo place_notional_buy per ottenere la qty reale
    e poi piazzare SL/TP con quella quantità.
    """
    url = "{}/v2/orders/{}".format(ALPACA_BASE, order_id)
    elapsed = 0
    
    while elapsed < timeout_sec:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(url, headers=HEADERS)
                if r.status_code == 200:
                    order = r.json()
                    status = order.get("status", "")
                    filled_qty = float(order.get("filled_qty", 0) or 0)
                    
                    if status == "filled" and filled_qty > 0:
                        return {
                            "filled": True,
                            "filled_qty": filled_qty,
                            "filled_avg_price": float(order.get("filled_avg_price", 0) or 0),
                            "status": status,
                            "order": order,
                        }
                    elif status in ("canceled", "rejected", "expired"):
                        return {
                            "filled": False,
                            "filled_qty": 0,
                            "filled_avg_price": 0,
                            "status": status,
                            "order": order,
                        }
        except Exception as e:
            print(f"  ⚠️  wait_for_fill poll error: {e}")
        
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval
    
    # Timeout — leggi ultimo stato
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, headers=HEADERS)
            if r.status_code == 200:
                order = r.json()
                filled_qty = float(order.get("filled_qty", 0) or 0)
                return {
                    "filled": filled_qty > 0,
                    "filled_qty": filled_qty,
                    "filled_avg_price": float(order.get("filled_avg_price", 0) or 0),
                    "status": order.get("status", "timeout"),
                    "order": order,
                }
    except Exception:
        pass
    
    return {"filled": False, "filled_qty": 0, "filled_avg_price": 0, "status": "timeout", "order": None}


# ============================================
# 🆕 PLACE BRACKETS AFTER FILL (SL + TP separati, qty frazionaria)
# ============================================

async def place_brackets_after_fill(symbol: str, qty: float, take_profit: float, stop_loss: float,
                                     time_in_force: str = "gtc"):
    """
    🆕 Dopo un buy notional con fill, piazza SL e TP separati con qty frazionaria.
    
    Alpaca NON permette bracket+notional, quindi facciamo:
    1. STOP LOSS sell order (type=stop)
    2. TAKE PROFIT sell order (type=limit)
    
    NOTA: non sono linkati come OCO (Alpaca non supporta OCO con fractional).
    L'Executor / RiskManager si occupa di cancellare l'altro quando uno triggera.
    
    Ritorna dict con i 2 ordini piazzati.
    """
    result = {"stop_loss_order": None, "take_profit_order": None, "errors": []}
    
    qty_str = str(round(qty, 4))  # max 4 decimali per Alpaca
    
    # 1. STOP LOSS
    try:
        sl_payload = {
            "symbol": symbol,
            "qty": qty_str,
            "side": "sell",
            "type": "stop",
            "time_in_force": time_in_force,
            "stop_price": str(round(stop_loss, 2)),
        }
        sl_result = await alpaca_request("POST", "{}/v2/orders".format(ALPACA_BASE), json=sl_payload)
        if sl_result:
            result["stop_loss_order"] = sl_result
            print(f"    🛡️  SL placed {symbol}: qty={qty_str} @ ${stop_loss:.2f}")
        else:
            result["errors"].append("SL order failed")
    except Exception as e:
        result["errors"].append(f"SL: {str(e)}")
        print(f"    ❌ SL error {symbol}: {e}")
    
    # 2. TAKE PROFIT
    try:
        tp_payload = {
            "symbol": symbol,
            "qty": qty_str,
            "side": "sell",
            "type": "limit",
            "time_in_force": time_in_force,
            "limit_price": str(round(take_profit, 2)),
        }
        tp_result = await alpaca_request("POST", "{}/v2/orders".format(ALPACA_BASE), json=tp_payload)
        if tp_result:
            result["take_profit_order"] = tp_result
            print(f"    🎯 TP placed {symbol}: qty={qty_str} @ ${take_profit:.2f}")
        else:
            result["errors"].append("TP order failed")
    except Exception as e:
        result["errors"].append(f"TP: {str(e)}")
        print(f"    ❌ TP error {symbol}: {e}")
    
    return result


# ============================================
# OCO Order (legacy — mantenuto per retrocompat, ma NON usato col fractional)
# ============================================

async def place_oco_order(
    symbol: str,
    qty: int,
    take_profit_price: float,
    stop_loss_price: float,
    time_in_force: str = "gtc",
):
    """
    🎯 Piazza un ordine OCO (One-Cancels-Other) di vendita.
    LEGACY: usato solo per shares intere. Non compatibile con fractional.
    """
    payload = {
        "symbol": symbol,
        "qty": str(qty),
        "side": "sell",
        "type": "limit",
        "time_in_force": time_in_force,
        "limit_price": str(round(take_profit_price, 2)),
        "order_class": "oco",
        "stop_loss": {
            "stop_price": str(round(stop_loss_price, 2))
        },
        "take_profit": {
            "limit_price": str(round(take_profit_price, 2))
        }
    }
    result = await alpaca_request("POST", "{}/v2/orders".format(ALPACA_BASE), json=payload)
    if result:
        print(f"✅ OCO {symbol}: TP={take_profit_price:.2f} SL={stop_loss_price:.2f} id={result.get('id', '')[:8]}")
        db = get_db()
        await db.alpaca_orders.insert_one({
            "order_id": result.get("id"),
            "symbol": symbol,
            "qty": qty,
            "side": "sell",
            "type": "oco",
            "status": result.get("status"),
            "take_profit_price": round(take_profit_price, 2),
            "stop_loss_price": round(stop_loss_price, 2),
            "created_at": datetime.utcnow(),
            "raw": result,
        })
    else:
        print(f"❌ OCO {symbol} FAILED: TP={take_profit_price:.2f} SL={stop_loss_price:.2f}")
    return result


# ============================================
# Bracket Order (legacy — solo shares intere)
# ============================================

async def place_bracket_order(symbol, qty, limit_price, take_profit, stop_loss):
    """
    LEGACY: bracket order con shares intere.
    Mantenuto per retrocompat, ma il nuovo flusso usa place_notional_buy + place_brackets_after_fill.
    """
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


# ============================================
# CANCEL / CLOSE
# ============================================

async def cancel_order(order_id):
    return await alpaca_request("DELETE", "{}/v2/orders/{}".format(ALPACA_BASE, order_id))


async def cancel_all_orders():
    return await alpaca_request("DELETE", "{}/v2/orders".format(ALPACA_BASE))


async def close_position(symbol):
    return await alpaca_request("DELETE", "{}/v2/positions/{}".format(ALPACA_BASE, symbol))



async def close_position_partial(symbol: str, qty: float):
    """
    🆕 v4.0 FASE 4 — Chiude parzialmente una posizione (X shares invece che tutta).
    
    Alpaca supporta partial close via query param `qty` sull'endpoint DELETE /positions/{symbol}.
    Supporta anche fractional shares.
    
    Esempio: close_position_partial("NOW", 60.5) chiude 60.5 shares di NOW.
    
    Ritorna:
    - dict con l'ordine sell creato (se successo)
    - None se errore
    """
    if qty <= 0:
        print(f"  ⚠️ close_position_partial: qty {qty} invalid for {symbol}")
        return None
    
    qty_str = str(round(qty, 4))  # max 4 decimali per fractional
    url = "{}/v2/positions/{}?qty={}".format(ALPACA_BASE, symbol, qty_str)
    
    result = await alpaca_request("DELETE", url)
    
    if result:
        db = get_db()
        await db.alpaca_orders.insert_one({
            "order_id": result.get("id"),
            "symbol": symbol,
            "qty": float(qty),
            "side": "sell",
            "type": "partial_close",
            "status": result.get("status"),
            "created_at": datetime.utcnow(),
            "raw": result,
            "source": "apm_scale_out",
        })
        print(f"  🟡 PARTIAL CLOSE {symbol}: {qty_str} shares (id={result.get('id', '')[:8]})")
    else:
        print(f"  ❌ PARTIAL CLOSE FAILED {symbol}: qty={qty_str}")
    
    return result


async def close_all_positions():
    return await alpaca_request("DELETE", "{}/v2/positions".format(ALPACA_BASE))


# ============================================
# UPDATE STOP LOSS (modificato per qty float)
# ============================================

async def update_stop_loss(symbol, new_stop_price):
    """
    🔧 Update stop loss by cancelling old SL and placing new one.
    Ora supporta qty float (fractional).
    """
    try:
        orders = await get_orders(status="open", limit=100)
        if not orders:
            return None
        for order in orders:
            if (order.get("symbol") == symbol and
                order.get("side") == "sell" and
                order.get("type") == "stop" and
                order.get("status") in ("new", "accepted", "pending_new")):
                old_stop = float(order.get("stop_price", 0))
                if new_stop_price > old_stop:
                    await cancel_order(order["id"])
                    # 🔧 qty preservato come stringa (può essere "3.809")
                    qty_raw = order.get("qty", "1")
                    new_order = await place_order(
                        symbol=symbol,
                        qty=qty_raw,  # passa la stringa originale (gestisce float)
                        side="sell",
                        order_type="stop",
                        time_in_force="gtc",
                        stop_price=new_stop_price
                    )
                    if new_order:
                        print(f"  Updated SL {symbol}: ${old_stop} -> ${new_stop_price}")
                        return new_order
        return None
    except Exception as e:
        print(f"  Update SL error {symbol}: {e}")
        return None


async def get_latest_price(symbol):
    url = "{}/v2/stocks/{}/quotes/latest".format(ALPACA_DATA, symbol)
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, headers=HEADERS)
        if r.status_code == 200:
            data = r.json()
            quote = data.get("quote", {})
            return (quote.get("ap", 0) + quote.get("bp", 0)) / 2
    return None
    # ============================================
# PORTFOLIO HISTORY & SUMMARY
# ============================================

async def get_portfolio_history(period="1M", timeframe="1D"):
    url = "{}/v2/account/portfolio/history?period={}&timeframe={}".format(ALPACA_BASE, period, timeframe)
    return await alpaca_request("GET", url)


async def get_alpaca_summary():
    account = await get_account()
    positions = await get_positions()
    orders = await get_orders(status="all", limit=50)
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
                "qty": float(p.get("qty", 0)),  # 🔧 ora float per fractional
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
            order_data = {
                "id": o.get("id"),
                "symbol": o.get("symbol"),
                "qty": o.get("qty"),
                "notional": o.get("notional"),  # 🆕 mostra anche notional se presente
                "side": o.get("side"),
                "type": o.get("type"),
                "status": o.get("status"),
                "filled_avg_price": o.get("filled_avg_price"),
                "filled_qty": o.get("filled_qty"),  # 🆕 utile per debug fractional
                "created_at": o.get("created_at"),
                "limit_price": o.get("limit_price"),
                "stop_price": o.get("stop_price"),
                "order_class": o.get("order_class"),
            }
            # Include bracket legs
            if o.get("legs"):
                order_data["legs"] = []
                for leg in o["legs"]:
                    order_data["legs"].append({
                        "id": leg.get("id"),
                        "symbol": leg.get("symbol"),
                        "qty": leg.get("qty"),
                        "side": leg.get("side"),
                        "type": leg.get("type"),
                        "status": leg.get("status"),
                        "limit_price": leg.get("limit_price"),
                        "stop_price": leg.get("stop_price"),
                        "filled_avg_price": leg.get("filled_avg_price"),
                    })
            order_list.append(order_data)
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


# ============================================
# LIVE PRICES
# ============================================

async def get_live_prices(symbols):
    if not symbols:
        return {}
    symbols_str = ",".join(symbols[:50])
    url = "{}/v2/stocks/snapshots?symbols={}&feed=iex".format(ALPACA_DATA, symbols_str)
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            r = await client.get(url, headers=HEADERS)
            if r.status_code != 200:
                return {}
            data = r.json()
            prices = {}
            for sym, snap in data.items():
                latest = snap.get("latestTrade", {})
                minute = snap.get("minuteBar", {})
                daily = snap.get("dailyBar", {})
                prev = snap.get("prevDailyBar", {})
                price = latest.get("p", 0) or minute.get("c", 0) or daily.get("c", 0)
                prev_close = prev.get("c", price)
                change = price - prev_close
                change_pct = (change / prev_close * 100) if prev_close > 0 else 0
                prices[sym] = {
                    "price": round(price, 2),
                    "prev_close": round(prev_close, 2),
                    "change": round(change, 2),
                    "change_pct": round(change_pct, 2),
                    "volume": daily.get("v", 0),
                    "high": round(daily.get("h", 0), 2),
                    "low": round(daily.get("l", 0), 2),
                }
            return prices
        except Exception as e:
            print("Live prices error: {}".format(e))
            return {}


# ============================================
# PORTFOLIO PERIODS (per benchmark chart)
# ============================================

async def get_portfolio_periods():
    periods = {}
    for label, period, tf in [
        ("1D", "1D", "15Min"),
        ("1W", "1W", "1D"),
        ("1M", "1M", "1D"),
        ("3M", "3M", "1D"),
        ("6M", "6M", "1D"),
        ("1Y", "1A", "1D"),
        ("YTD", "1A", "1D"),
    ]:
        url = "{}/v2/account/portfolio/history?period={}&timeframe={}".format(ALPACA_BASE, period, tf)
        async with httpx.AsyncClient(timeout=15) as client:
            try:
                r = await client.get(url, headers=HEADERS)
                if r.status_code == 200:
                    data = r.json()
                    ts = data.get("timestamp", [])
                    eq = data.get("equity", [])
                    pl = data.get("profit_loss", [])
                    plp = data.get("profit_loss_pct", [])
                    points = []
                    for i in range(len(ts)):
                        if eq[i]:
                            points.append({
                                "date": str(ts[i]) if tf == "15Min" else datetime.fromtimestamp(ts[i]).strftime("%Y-%m-%d"),
                                "equity": round(eq[i], 2),
                                "pnl": round((pl[i] or 0), 2),
                                "pnl_pct": round((plp[i] or 0) * 100, 2),
                            })
                    if points:
                        periods[label] = points
            except:
                pass
    return periods
