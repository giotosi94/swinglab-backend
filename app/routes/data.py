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
import asyncio
from datetime import datetime as _dt

router = APIRouter()

_STOCKS_PIPELINE_LOCK = asyncio.Lock()
_STOCKS_PIPELINE_STATE = {"started_at": None, "source": None}


def _pipeline_busy_response(source):
    return {
        "status": "skipped",
        "reason": "STOCKS_PIPELINE_ALREADY_RUNNING",
        "requested_by": source,
        "running_since": _STOCKS_PIPELINE_STATE.get("started_at"),
        "running_source": _STOCKS_PIPELINE_STATE.get("source"),
    }


async def _run_stocks_pipeline_locked(source, with_sectors=False):
    async with _STOCKS_PIPELINE_LOCK:
        _STOCKS_PIPELINE_STATE["started_at"] = _dt.utcnow().isoformat()
        _STOCKS_PIPELINE_STATE["source"] = source
        try:
            sectors = await fetch_and_analyze_sectors() if with_sectors else None
            stocks = await fetch_and_analyze_stocks()
            trader_result = await run_auto_trader()
            return sectors, stocks, trader_result
        finally:
            _STOCKS_PIPELINE_STATE["started_at"] = None
            _STOCKS_PIPELINE_STATE["source"] = None


@router.post("/refresh/sectors")
async def refresh_sectors():
    results = await fetch_and_analyze_sectors()
    return {"message": "Sectors updated", "count": len(results)}


@router.post("/refresh/stocks")
async def refresh_stocks():
    if _STOCKS_PIPELINE_LOCK.locked():
        print("[LOCK] refresh/stocks skipped: pipeline already running")
        return _pipeline_busy_response("refresh/stocks")
    _, results, trader_result = await _run_stocks_pipeline_locked("refresh/stocks")
    return {"message": "Stocks updated", "count": len(results), "auto_trader": trader_result}


# ============================================
# 🆕 v3.5 — ASYNC REFRESH (per cron-job.org free tier)
# ============================================

@router.post("/refresh/stocks-async")
async def refresh_stocks_async():
    """
    🆕 v3.5 — Fire-and-forget refresh stocks.
    
    Avvia la pipeline in background e ritorna 200 OK immediatamente.
    Utile per cron con timeout stretti (es. cron-job.org free = 30s).
    
    Il refresh continua a girare in background senza bloccare il cron.
    Log dell'esecuzione visibili su Render.
    """
    import asyncio
    from datetime import datetime
    
    if _STOCKS_PIPELINE_LOCK.locked():
        print("[LOCK] refresh/stocks-async skipped: pipeline already running")
        return _pipeline_busy_response("refresh/stocks-async")

    async def _run_in_background():
        try:
            print(f"[ASYNC] Pipeline started at {datetime.utcnow().isoformat()}")
            _, results, trader_result = await _run_stocks_pipeline_locked("refresh/stocks-async")
            buys = len(trader_result.get('steps', {}).get('executor', {}).get('details', {}).get('executed_buys', []))
            sells = len(trader_result.get('steps', {}).get('executor', {}).get('details', {}).get('executed_sells', []))
            print(f"[ASYNC] Pipeline completed: {len(results)} stocks, buys={buys}, sells={sells}")
        except Exception as e:
            print(f"[ASYNC] Pipeline error: {e}")
    
    # Fire-and-forget task
    asyncio.create_task(_run_in_background())
    
    return {
        "status": "started",
        "message": "Pipeline started in background",
        "started_at": datetime.utcnow().isoformat(),
    }


@router.post("/refresh/all")
async def refresh_all():
    if _STOCKS_PIPELINE_LOCK.locked():
        print("[LOCK] refresh/all skipped: pipeline already running")
        return _pipeline_busy_response("refresh/all")
    sectors, stocks, trader_result = await _run_stocks_pipeline_locked("refresh/all", with_sectors=True)
    return {"message": "Full refresh completed", "sectors": len(sectors), "stocks": len(stocks), "auto_trader": trader_result}

@router.post("/wipe/sector-bars")
async def wipe_sector_bars():
    """v4.3 — Wipe cache bars di sectors ETF per force refresh fresh."""
    db = get_db()
    sector_etfs = ["XLK", "XLF", "XLV", "XLI", "XLY", "XLP", 
                   "XLE", "XLU", "XLB", "XLRE", "XLC"]
    
    result = await db.stock_bars.delete_many({"ticker": {"$in": sector_etfs}})
    
    return {
        "message": "Sector ETF bars cache wiped",
        "deleted": result.deleted_count,
        "sectors": sector_etfs,
    }

@router.post("/refresh/market")
async def refresh_market_data():
    """
    🆕 v4.2 — Aggiorna macro data (SPY, QQQ, VXX, TLT, GLD, ecc.) da Alpaca IEX.
    Bypass Twelve Data che ha ETF stale.
    Popola market_regime collection con prezzi live + RSI/EMA calcolati.
    """
    from app.services.alpaca_trader import fetch_macro_data_alpaca
    from datetime import datetime
    
    db = get_db()
    
    # Lista macro ETF + indici
    macro_symbols = [
        "SPY", "QQQ", "IWM", "DIA",
        "VXX", "VIXY",
        "TLT", "HYG", "LQD",
        "GLD", "USO",
        "RSP", "IWO",
        "FXE", "UUP",
        "EEM", "IYT",
    ]
    
    print(f"🔄 Refreshing {len(macro_symbols)} macro symbols from Alpaca IEX...")
    
    results = await fetch_macro_data_alpaca(macro_symbols)
    
    updated_count = 0
    failed = []
    for symbol, data in results.items():
        try:
            await db.market_regime.update_one(
                {"symbol": symbol},
                {"$set": data},
                upsert=True,
            )
            updated_count += 1
        except Exception as e:
            failed.append({"symbol": symbol, "error": str(e)})
            print(f"  ⚠️ DB update error {symbol}: {e}")
    
    return {
        "message": "Market data refreshed",
        "updated": updated_count,
        "total_requested": len(macro_symbols),
        "failed": failed,
        "source": "alpaca_iex",
        "refreshed_at": datetime.utcnow().isoformat(),
    }

@router.get("/tickers/list")
async def list_all_tickers():
    """
    🆕 v4.2 — Ritorna lista di tutti i ticker disponibili con nome azienda.
    Usato dal frontend per autocomplete search.
    """
    from app.services.stock_names import STOCK_NAMES, get_stock_name
    db = get_db()
    
    # Prendi tutti gli asset dal DB
    assets = await db.assets.find({}, {"ticker": 1, "sector_code": 1}).to_list(300)
    
    tickers = []
    for a in assets:
        ticker = a.get("ticker", "")
        if not ticker:
            continue
        tickers.append({
            "ticker": ticker,
            "name": get_stock_name(ticker),
            "sector": a.get("sector_code", ""),
        })
    
    # Aggiungi ETF e indici che sono in STOCK_NAMES ma potrebbero non essere in assets
    existing_tickers = {t["ticker"] for t in tickers}
    extra_symbols = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLV", "XLI", 
                     "XLY", "XLP", "XLE", "XLU", "XLB", "XLRE", "XLC", 
                     "VIXY", "FXE", "UUP", "TLT", "HYG", "LQD", "GLD", "USO",
                     "RSP", "IWO", "EEM", "IYT", "VXX"]
    for sym in extra_symbols:
        if sym not in existing_tickers and sym in STOCK_NAMES:
            tickers.append({
                "ticker": sym,
                "name": STOCK_NAMES[sym],
                "sector": "ETF",
            })
    
    # Sort by ticker
    tickers.sort(key=lambda x: x["ticker"])
    
    return {
        "total": len(tickers),
        "tickers": tickers,
    }

@router.get("/max-strategy-risk-shadow")
async def get_max_strategy_risk_shadow():
    db = get_db()
    shadow = await db.agent_state.find_one({"_id": "max_strategy_risk_shadow"})
    if not shadow:
        return {
            "mode": "SHADOW_SIZING",
            "live_execution_enabled": False,
            "strategy_version": "max_structure_v1_5_2",
            "decision_counts": {},
            "candidates": [],
            "message": "Risk shadow state not initialized",
        }
    shadow["_id"] = str(shadow["_id"])
    value = shadow.get("updated_at")
    if value and hasattr(value, "isoformat"):
        shadow["updated_at"] = value.isoformat()
    return shadow


@router.get("/max-strategy-shadow")
async def get_max_strategy_shadow():
    db = get_db()
    shadow = await db.agent_state.find_one({"_id": "max_strategy_shadow"})
    if not shadow:
        return {
            "mode": "SHADOW",
            "live_execution_enabled": False,
            "strategy_version": "max_structure_v1_5_2",
            "action_counts": {},
            "status_counts": {},
            "candidates": [],
            "message": "Shadow state not initialized",
        }
    shadow["_id"] = str(shadow["_id"])
    value = shadow.get("updated_at")
    if value and hasattr(value, "isoformat"):
        shadow["updated_at"] = value.isoformat()
    return shadow


@router.get("/max-strategy-validation")
async def get_max_strategy_validation(limit: int = Query(50, ge=1, le=500), ticker: str = None, include_legacy: bool = False):
    db = get_db()
    query = {} if include_legacy else {"lifecycle_version": {"$gte": 1}}
    if ticker:
        query["ticker"] = ticker.upper()
    total = await db.max_strategy_signals.count_documents(query)
    pipeline = [
        {"$match": query},
        {"$group": {"_id": "$status", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]
    status_rows = await db.max_strategy_signals.aggregate(pipeline).to_list(50)
    status_counts = {row["_id"] or "UNKNOWN": row["count"] for row in status_rows}
    triggered = await db.max_strategy_signals.count_documents({**query, "outcomes.trigger_reached": True})
    invalidated = await db.max_strategy_signals.count_documents({**query, "outcomes.invalidation_reached": True})
    exceeded = await db.max_strategy_signals.count_documents({**query, "outcomes.maximum_entry_exceeded": True})
    completed_5d = await db.max_strategy_signals.count_documents({**query, "outcomes.return_5d_pct": {"$ne": None}})
    completed_10d = await db.max_strategy_signals.count_documents({**query, "outcomes.return_10d_pct": {"$ne": None}})
    completed_20d = await db.max_strategy_signals.count_documents({**query, "outcomes.return_20d_pct": {"$ne": None}})
    averages_pipeline = [
        {"$match": query},
        {"$group": {
            "_id": None,
            "avg_mfe_pct": {"$avg": "$outcomes.mfe_pct"},
            "avg_mae_pct": {"$avg": "$outcomes.mae_pct"},
            "avg_return_5d_pct": {"$avg": "$outcomes.return_5d_pct"},
            "avg_return_10d_pct": {"$avg": "$outcomes.return_10d_pct"},
            "avg_return_20d_pct": {"$avg": "$outcomes.return_20d_pct"},
        }},
    ]
    average_rows = await db.max_strategy_signals.aggregate(averages_pipeline).to_list(1)
    averages_raw = average_rows[0] if average_rows else {}
    averages = {
        key: round(value, 2) if isinstance(value, (int, float)) else None
        for key, value in averages_raw.items()
        if key != "_id"
    }
    cursor = db.max_strategy_signals.find(query).sort("created_at", -1).limit(limit)
    signals = await cursor.to_list(limit)
    for signal in signals:
        signal["_id"] = str(signal["_id"])
        for key in ("created_at", "updated_at", "last_seen_at"):
            value = signal.get(key)
            if value and hasattr(value, "isoformat"):
                signal[key] = value.isoformat()
    return {
        "summary": {
            "total": total,
            "ticker_filter": ticker.upper() if ticker else None,
            "include_legacy": include_legacy,
            "legacy_total": await db.max_strategy_signals.count_documents({"lifecycle_version": 0}),
            "status_counts": status_counts,
            "trigger_reached": triggered,
            "invalidation_reached": invalidated,
            "maximum_entry_exceeded": exceeded,
            "completed_5d": completed_5d,
            "completed_10d": completed_10d,
            "completed_20d": completed_20d,
            "averages": averages,
        },
        "signals": signals,
    }


def _iso(value):
    return value.isoformat() if value is not None and hasattr(value, "isoformat") else value


def _spy_forward_return(spy_bars, start_date, horizon):
    if not spy_bars or not start_date:
        return None
    dates = [bar.get("date") for bar in spy_bars]
    start_index = next((index for index, date in enumerate(dates) if date and date >= start_date), None)
    if start_index is None or start_index + horizon - 1 >= len(spy_bars):
        return None
    start_price = spy_bars[start_index].get("c")
    end_price = spy_bars[start_index + horizon - 1].get("c")
    if not start_price or not end_price:
        return None
    return round((end_price - start_price) / start_price * 100, 2)


@router.get("/max-strategy-plans")
async def get_max_strategy_plans(
    limit: int = Query(100, ge=1, le=500),
    ticker: str = None,
    active_only: bool = False,
):
    db = get_db()
    query = {}
    if ticker:
        query["ticker"] = ticker.upper()
    if active_only:
        query["is_active"] = True

    state_rows = await db.max_strategy_plans.aggregate([
        {"$match": query},
        {"$group": {"_id": "$lifecycle_state", "count": {"$sum": 1}}},
    ]).to_list(20)
    state_counts = {row["_id"] or "UNKNOWN": row["count"] for row in state_rows}

    spy_doc = await db.stock_bars.find_one({"ticker": "SPY"}, {"bars": {"$slice": -400}})
    spy_bars = (spy_doc or {}).get("bars") or []

    triggered_plans = await db.max_strategy_plans.find(
        {**query, "triggered_at": {"$ne": None}},
        {"plan_key": 1, "ticker": 1, "triggered_at": 1, "outcomes": 1},
    ).to_list(500)

    horizons = {"5d": 5, "10d": 10, "20d": 20}
    stats = {}
    for label, horizon in horizons.items():
        rows = []
        for plan in triggered_plans:
            value = (plan.get("outcomes") or {}).get(f"return_{label}_pct")
            if value is None:
                continue
            spy_value = _spy_forward_return(spy_bars, plan.get("triggered_at"), horizon)
            rows.append((value, spy_value))
        if not rows:
            stats[label] = {"plans": 0}
            continue
        returns = [value for value, _ in rows]
        paired = [(value, spy_value) for value, spy_value in rows if spy_value is not None]
        excess = [value - spy_value for value, spy_value in paired]
        stats[label] = {
            "plans": len(rows),
            "avg_return_pct": round(sum(returns) / len(returns), 2),
            "win_rate_pct": round(sum(1 for value in returns if value > 0) / len(returns) * 100, 1),
            "avg_spy_return_pct": round(sum(spy for _, spy in paired) / len(paired), 2) if paired else None,
            "avg_excess_vs_spy_pct": round(sum(excess) / len(excess), 2) if excess else None,
            "beat_spy_pct": round(sum(1 for value in excess if value > 0) / len(excess) * 100, 1) if excess else None,
        }

    mfe = [p["outcomes"]["mfe_pct"] for p in triggered_plans if (p.get("outcomes") or {}).get("mfe_pct") is not None]
    mae = [p["outcomes"]["mae_pct"] for p in triggered_plans if (p.get("outcomes") or {}).get("mae_pct") is not None]

    plans = await db.max_strategy_plans.find(query).sort("updated_at", -1).limit(limit).to_list(limit)
    for plan in plans:
        plan["_id"] = str(plan["_id"])
        for key in ("created_at", "updated_at"):
            plan[key] = _iso(plan.get(key))
        for event in plan.get("events") or []:
            event["at"] = _iso(event.get("at"))

    return {
        "summary": {
            "total_plans": await db.max_strategy_plans.count_documents(query),
            "active_plans": await db.max_strategy_plans.count_documents({**query, "is_active": True}),
            "triggered_plans": len(triggered_plans),
            "state_counts": state_counts,
            "avg_mfe_pct": round(sum(mfe) / len(mfe), 2) if mfe else None,
            "avg_mae_pct": round(sum(mae) / len(mae), 2) if mae else None,
            "performance_vs_spy": stats,
            "benchmark": "SPY",
            "live_execution_enabled": False,
        },
        "plans": plans,
    }


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
async def reset_trader():
    """
    🔄 v2.1 — Reset completo.
    Il capitale iniziale viene preso automaticamente da Alpaca (equity attuale).
    Non serve più passare il parametro capital.
    """
    state = await reset_auto_trader(initial_capital=None)
    return {
        "message": "Reset complete (capital from Alpaca)",
        "state": state
    }


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

@router.delete("/reset-bars/{ticker}")
async def reset_ticker_bars(ticker: str):
    """Cancella le bars di UN singolo ticker per forzare re-download pulito."""
    db = get_db()
    result = await db.stock_bars.delete_one({"ticker": ticker.upper()})
    return {"deleted": result.deleted_count, "ticker": ticker.upper(),
            "message": "Bars deleted. Next refresh will bulk re-download."}

@router.delete("/assets/cleanup-search")
async def cleanup_search_assets():
    """Rimuove ticker da ricerche (SEARCH) + ETF macro dall'universo trade."""
    db = get_db()
    macro_etfs = ["SPY", "QQQ", "IWM", "DIA", "VXX", "VIXY", "TLT", "HYG",
                  "LQD", "GLD", "USO", "RSP", "IWO", "FXE", "UUP", "EEM", "IYT"]
    result = await db.assets.delete_many({
        "$or": [
            {"sector_code": "SEARCH"},
            {"ticker": {"$in": macro_etfs}},
        ]
    })
    return {"deleted": result.deleted_count, "message": "Universe cleaned (SEARCH + macro ETF removed)"}


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


# ============================================
# 🆕 v2.2 — STARTING CAPITAL FROM ALPACA (v2)
# ============================================

@router.get("/starting-capital")
async def get_starting_capital():
    """
    🆕 v2 — Ritorna il capitale iniziale REALE da Alpaca.
    
    Logica:
    1. Chiama Alpaca Portfolio History con periodo massimo
    2. Prende il PRIMO valore di equity nella history
    3. Quello è lo starting_capital vero
    4. Total P&L calcolato correttamente
    """
    from datetime import datetime
    from app.services.alpaca_trader import get_portfolio_history, get_account
    
    account = await get_account()
    if not account:
        return {"error": "Alpaca not connected", "starting_capital": 100000}
    
    current_equity = float(account.get("equity", 0))
    
    # Prende storia completa (period = tutto disponibile)
    history = await get_portfolio_history(period="1A", timeframe="1D")
    
    starting_capital = None
    first_date = None
    
    if history and history.get("equity"):
        equities = history.get("equity", [])
        timestamps = history.get("timestamp", [])
        
        # Trova il primo equity valido (>0)
        for i, eq in enumerate(equities):
            if eq and eq > 0:
                starting_capital = round(eq, 2)
                if i < len(timestamps):
                    first_date = datetime.fromtimestamp(timestamps[i]).strftime("%Y-%m-%d")
                break
    
    # Fallback: se non troviamo history, usa 100000 (default Alpaca paper)
    if starting_capital is None or starting_capital <= 0:
        starting_capital = 100000.0
        first_date = "unknown"
    
    total_pnl_dollar = round(current_equity - starting_capital, 2)
    total_pnl_pct = round((total_pnl_dollar / starting_capital * 100), 2) if starting_capital > 0 else 0
    
    return {
        "starting_capital": starting_capital,
        "current_equity": current_equity,
        "total_pnl_dollar": total_pnl_dollar,
        "total_pnl_pct": total_pnl_pct,
        "starting_date": first_date,
        "source": "alpaca_portfolio_history",
        "calculated_at": datetime.utcnow().isoformat(),
    }



# ============================================
# v4.6 — BACKFILL ADAPTIVE TARGETS
# ============================================

@router.post("/backfill/adaptive-targets")
async def backfill_adaptive_targets():
    """v4.6 One-shot: calcola adaptive targets per buy_trade esistenti."""
    from datetime import datetime as dt
    db = get_db()
    
    # Trova tutti i buy attivi senza adaptive_t1_pct
    buys = await db.trade_history.find({
        "side": "buy",
        "sell_linked": {"$ne": True},
        "adaptive_t1_pct": {"$exists": False}
    }).to_list(100)
    
    updated = 0
    skipped = []
    for buy in buys:
        entry_price = buy.get("entry_price", 0)
        target = buy.get("target", 0)
        stop = buy.get("stop_loss", 0)
        ticker = buy.get("ticker", "?")
        
        if entry_price <= 0 or target <= 0:
            skipped.append({"ticker": ticker, "reason": "invalid entry/target"})
            continue
        
        target_distance_pct = ((target - entry_price) / entry_price * 100)
        target_distance_pct = max(2.0, min(40.0, target_distance_pct))
        
        sl_distance_pct = ((entry_price - stop) / entry_price * 100) if stop > 0 else 4.0
        sl_distance_pct = max(1.0, min(15.0, sl_distance_pct))
        
        adaptive_t1_pct = round(target_distance_pct * 0.40, 2)
        adaptive_t2_pct = round(target_distance_pct * 0.70, 2)
        adaptive_t3_pct = round(target_distance_pct * 1.00, 2)
        
        await db.trade_history.update_one(
            {"_id": buy["_id"]},
            {"$set": {
                "adaptive_t1_pct": adaptive_t1_pct,
                "adaptive_t2_pct": adaptive_t2_pct,
                "adaptive_t3_pct": adaptive_t3_pct,
                "target_distance_pct": round(target_distance_pct, 2),
                "sl_distance_pct": round(sl_distance_pct, 2),
                "backfilled_at": dt.utcnow(),
            }}
        )
        updated += 1
    
    return {
        "message": "Backfill completed",
        "updated": updated,
        "total_buys": len(buys),
        "skipped": skipped,
    }

@router.post("/backfill-bars-history")
async def start_bars_history_backfill(target_bars: int = 750, max_concurrent: int = 4):
    import asyncio
    from app.services.data_fetcher import backfill_long_history
    target_bars = max(300, min(target_bars, 1000))
    db = get_db()
    job_id = f"stock_bars_{target_bars}"
    existing = await db.backfill_jobs.find_one({"_id": job_id})
    if existing and existing.get("status") == "running":
        existing["_id"] = str(existing["_id"])
        return existing
    asyncio.create_task(backfill_long_history(target_bars, max_concurrent))
    return {"job_id": job_id, "status": "started", "target_bars": target_bars}


@router.get("/backfill-bars-history/status")
async def bars_history_backfill_status(target_bars: int = 750):
    from app.services.data_fetcher import get_bars_coverage
    db = get_db()
    job_id = f"stock_bars_{target_bars}"
    job = await db.backfill_jobs.find_one({"_id": job_id}) or {"_id": job_id, "status": "not_started", "target_bars": target_bars}
    job["_id"] = str(job["_id"])
    job["coverage"] = await get_bars_coverage(target_bars)
    for key in ("started_at", "updated_at", "finished_at"):
        if job.get(key) and hasattr(job[key], "isoformat"):
            job[key] = job[key].isoformat()
    return job


_BACKTEST_LOCK = asyncio.Lock()
_BACKTEST_JOBS = {}
_BACKTEST_STATE = {"current_job": None}


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except Exception:
            return value
    if isinstance(value, float) and value != value:
        return None
    return value


async def _resolve_backtest_kwargs(
    days: int = 180,
    min_confluence: float = None,
    max_positions: int = None,
    position_size_pct: float = None,
    use_apm: bool = True,
    t1_ratio: float = 0.40,
    t2_ratio: float = 0.70,
    t3_ratio: float = 1.00,
    use_preset: bool = True,
    use_mtf: bool = True,
    use_momentum: bool = False,
    use_sector_bottom: bool = False,
    use_crash_deploy: bool = False,
    use_rotation: bool = False,
    use_sector_intelligence: bool = False,
    t1_size_pct: float = 30.0,
    t2_size_pct: float = 30.0,
    t3_size_pct: float = 25.0,
    floor_t1_pct: float = 0.0,
    floor_t2_pct: float = 3.0,
    floor_t3_pct: float = 8.0,
    min_holding_days: int = 1,
    use_dynamic_sizing: bool = False,
    use_apm_exit_proxy: bool = False,
    use_trend_leadership: bool = False,
    park_cash_in_spy: bool = False,
    trend_slots: int = 4,
    trend_max_per_sector: int = 3,
    trend_max_from_high_pct: float = 10.0,
    trend_max_rsi: float = 80.0,
    trend_rs_lookback: int = 126,
    core_spy_pct: float = 0.0,
    use_max_strategy: bool = False,
    max_slots: int = 3,
    max_stop_cap_pct: float = 15.0,
):
    from app.services.backtesting import run_backtest

    db = get_db()
    preset_name = None

    if use_preset:
        app_settings = await db.app_settings.find_one({"_id": "risk_params"}) or {}
        alpha_params = await db.agent_memory_alpha_strategist.find_one({"_id": "params"}) or {}
        risk_params = await db.agent_memory_risk_manager.find_one({"_id": "params"}) or {}
        preset_name = app_settings.get("active_preset")

        if max_positions is None:
            max_positions = app_settings.get("max_positions", 12)
        if position_size_pct is None:
            position_size_pct = app_settings.get("position_size_pct", 18.0)
        if min_confluence is None:
            min_confluence = alpha_params.get("min_confluence", 48)

    if not use_preset:
        risk_params = {}
        app_settings = {}
    max_positions = max_positions if max_positions is not None else 12
    position_size_pct = position_size_pct if position_size_pct is not None else 18.0
    min_confluence = min_confluence if min_confluence is not None else 48

    kwargs = dict(
        days=days,
        min_confluence=min_confluence,
        max_positions=max_positions,
        position_size_pct=position_size_pct,
        use_apm=use_apm,
        t1_ratio=t1_ratio,
        t2_ratio=t2_ratio,
        t3_ratio=t3_ratio,
        use_mtf=use_mtf,
        use_momentum=use_momentum,
        use_sector_bottom=use_sector_bottom,
        use_crash_deploy=use_crash_deploy,
        use_rotation=use_rotation,
        use_sector_intelligence=use_sector_intelligence,
        t1_size_pct=t1_size_pct,
        t2_size_pct=t2_size_pct,
        t3_size_pct=t3_size_pct,
        floor_t1_pct=floor_t1_pct,
        floor_t2_pct=floor_t2_pct,
        floor_t3_pct=floor_t3_pct,
        min_holding_days=min_holding_days,
        use_dynamic_sizing=use_dynamic_sizing,
        use_apm_exit_proxy=use_apm_exit_proxy,
        use_trend_leadership=use_trend_leadership,
        park_cash_in_spy=park_cash_in_spy,
        trend_slots=max(1, min(trend_slots, 8)),
        trend_max_per_sector=max(1, min(trend_max_per_sector, 6)),
        trend_max_from_high_pct=max(3.0, min(trend_max_from_high_pct, 25.0)),
        trend_max_rsi=max(60.0, min(trend_max_rsi, 90.0)),
        trend_rs_lookback=max(63, min(trend_rs_lookback, 200)),
        core_spy_pct=max(0.0, min(core_spy_pct, 90.0)),
        use_max_strategy=use_max_strategy,
        max_slots=max(1, min(max_slots, 6)),
        max_stop_cap_pct=max(5.0, min(max_stop_cap_pct, 30.0)),
        risk_pct_per_trade=risk_params.get("risk_pct_per_trade", app_settings.get("risk_pct_per_trade", 3.0) if use_preset else 3.0),
        max_position_pct=risk_params.get("max_position_pct", app_settings.get("max_position_pct", 25.0) if use_preset else 25.0),
        min_cash_reserve_pct=risk_params.get("min_cash_reserve_pct", app_settings.get("min_cash_reserve_pct", 5.0) if use_preset else 5.0),
        risk_params=risk_params,
    )
    return kwargs, preset_name


async def _execute_backtest_job(job_id, kwargs, preset_name):
    from app.services.backtesting import run_backtest
    db = get_db()
    job = _BACKTEST_JOBS[job_id]
    async with _BACKTEST_LOCK:
        _BACKTEST_STATE["current_job"] = job_id
        job["status"] = "running"
        job["started_at"] = _dt.utcnow().isoformat()
        print(f"[BACKTEST] job {job_id} started: days={kwargs.get('days')}")
        try:
            result = await run_backtest(**kwargs)
            result["active_preset"] = preset_name
            result = _plain(result)
            job["status"] = "error" if result.get("error") else "done"
            job["result"] = result
            job["error"] = result.get("error")
        except Exception as e:
            job["status"] = "error"
            job["error"] = str(e)
            print(f"[BACKTEST] job {job_id} error: {e}")
        finally:
            job["finished_at"] = _dt.utcnow().isoformat()
            _BACKTEST_STATE["current_job"] = None
            print(f"[BACKTEST] job {job_id} {job['status']}")
    try:
        await db.backtest_results.update_one(
            {"_id": job_id},
            {"$set": {k: v for k, v in job.items() if k != "_id"}},
            upsert=True,
        )
    except Exception as e:
        print(f"[BACKTEST] save error {job_id}: {e}")
    for old_id in list(_BACKTEST_JOBS.keys())[:-3]:
        if _BACKTEST_JOBS[old_id].get("status") in ("done", "error"):
            _BACKTEST_JOBS.pop(old_id, None)


@router.post("/backtest/start")
async def backtest_start(
    days: int = 180,
    min_confluence: float = None,
    max_positions: int = None,
    position_size_pct: float = None,
    use_apm: bool = True,
    t1_ratio: float = 0.40,
    t2_ratio: float = 0.70,
    t3_ratio: float = 1.00,
    use_preset: bool = True,
    use_mtf: bool = True,
    use_momentum: bool = False,
    use_sector_bottom: bool = False,
    use_crash_deploy: bool = False,
    use_rotation: bool = False,
    use_sector_intelligence: bool = False,
    t1_size_pct: float = 30.0,
    t2_size_pct: float = 30.0,
    t3_size_pct: float = 25.0,
    floor_t1_pct: float = 0.0,
    floor_t2_pct: float = 3.0,
    floor_t3_pct: float = 8.0,
    min_holding_days: int = 1,
    use_dynamic_sizing: bool = False,
    use_apm_exit_proxy: bool = False,
    use_trend_leadership: bool = False,
    park_cash_in_spy: bool = False,
    trend_slots: int = 4,
    trend_max_per_sector: int = 3,
    trend_max_from_high_pct: float = 10.0,
    trend_max_rsi: float = 80.0,
    trend_rs_lookback: int = 126,
    core_spy_pct: float = 0.0,
    use_max_strategy: bool = False,
    max_slots: int = 3,
    max_stop_cap_pct: float = 15.0,
):
    params = dict(locals())
    import uuid
    if _BACKTEST_LOCK.locked() or _BACKTEST_STATE["current_job"]:
        return {"status": "busy", "job_id": _BACKTEST_STATE["current_job"], "message": "Un backtest e' gia' in esecuzione"}
    kwargs, preset_name = await _resolve_backtest_kwargs(**params)
    job_id = uuid.uuid4().hex[:12]
    _BACKTEST_JOBS[job_id] = {"job_id": job_id, "status": "queued", "days": kwargs.get("days"), "created_at": _dt.utcnow().isoformat()}
    _BACKTEST_STATE["current_job"] = job_id
    asyncio.create_task(_execute_backtest_job(job_id, kwargs, preset_name))
    return {"status": "queued", "job_id": job_id}


@router.get("/backtest/job/{job_id}")
async def backtest_job(job_id: str):
    job = _BACKTEST_JOBS.get(job_id)
    if job is None:
        doc = await get_db().backtest_results.find_one({"_id": job_id})
        if not doc:
            return {"job_id": job_id, "status": "not_found"}
        doc.pop("_id", None)
        return _plain(doc)
    elapsed = None
    if job.get("started_at"):
        end = _dt.fromisoformat(job["finished_at"]) if job.get("finished_at") else _dt.utcnow()
        elapsed = round((end - _dt.fromisoformat(job["started_at"])).total_seconds(), 1)
    return {**job, "elapsed_s": elapsed}


@router.post("/backtest/run")
async def backtest_run(
    days: int = 180,
    min_confluence: float = None,
    max_positions: int = None,
    position_size_pct: float = None,
    use_apm: bool = True,
    t1_ratio: float = 0.40,
    t2_ratio: float = 0.70,
    t3_ratio: float = 1.00,
    use_preset: bool = True,
    use_mtf: bool = True,
    use_momentum: bool = False,
    use_sector_bottom: bool = False,
    use_crash_deploy: bool = False,
    use_rotation: bool = False,
    use_sector_intelligence: bool = False,
    t1_size_pct: float = 30.0,
    t2_size_pct: float = 30.0,
    t3_size_pct: float = 25.0,
    floor_t1_pct: float = 0.0,
    floor_t2_pct: float = 3.0,
    floor_t3_pct: float = 8.0,
    min_holding_days: int = 1,
    use_dynamic_sizing: bool = False,
    use_apm_exit_proxy: bool = False,
    use_trend_leadership: bool = False,
    park_cash_in_spy: bool = False,
    trend_slots: int = 4,
    trend_max_per_sector: int = 3,
    trend_max_from_high_pct: float = 10.0,
    trend_max_rsi: float = 80.0,
    trend_rs_lookback: int = 126,
    core_spy_pct: float = 0.0,
    use_max_strategy: bool = False,
    max_slots: int = 3,
    max_stop_cap_pct: float = 15.0,
):
    params = dict(locals())
    from app.services.backtesting import run_backtest
    if _BACKTEST_LOCK.locked():
        return {"error": "Un backtest e' gia' in esecuzione: usa /backtest/start"}
    kwargs, preset_name = await _resolve_backtest_kwargs(**params)
    async with _BACKTEST_LOCK:
        result = await run_backtest(**kwargs)
    result["active_preset"] = preset_name
    return _plain(result)

def _max_scan_busy():
    return _STOCKS_PIPELINE_LOCK.locked() or _BACKTEST_LOCK.locked()


@router.post("/max-scan/start")
async def max_scan_start(restart: bool = False, scan_days: int = 760):
    from app.services.max_backtest_scan import run_max_history_scan, is_running, get_scan_status
    if is_running():
        return {"status": "already_running", **(await get_scan_status())}
    scan_days = max(100, min(scan_days, 900))
    asyncio.create_task(run_max_history_scan(scan_days=scan_days, restart=restart, is_busy=_max_scan_busy))
    return {"status": "started", "restart": restart, "scan_days": scan_days}


@router.get("/max-scan/status")
async def max_scan_status():
    from app.services.max_backtest_scan import get_scan_status
    return await get_scan_status()


@router.post("/max-scan/stop")
async def max_scan_stop():
    from app.services.max_backtest_scan import request_stop
    return {"stop_requested": request_stop()}


@router.post("/load-spy-history")
async def load_spy_history_endpoint(years: int = 7):
    """One-shot: carica storico lungo SPY in spy_history (isolato)."""
    from app.services.spy_history import load_spy_history
    return await load_spy_history(years=years)


@router.post("/backtest-crash-spy")
async def backtest_crash_spy(start_date: str = None, end_date: str = None):
    """Mini-backtest crash deploy vs buy&hold su SPY storico."""
    from app.services.spy_history import backtest_crash_deploy_spy
    return await backtest_crash_deploy_spy(start_date=start_date, end_date=end_date)


@router.post("/load-sectors-history")
async def load_sectors_history_endpoint(years: int = 7):
    """One-shot: carica storico lungo degli 11 ETF settoriali (isolato)."""
    from app.services.spy_history import load_sectors_history
    return await load_sectors_history(years=years)


@router.post("/backtest-sector-rotation")
async def backtest_sector_rotation_endpoint(start_date: str, end_date: str):
    """Mini-backtest rotazione settoriale vs equal-weight su ETF storici."""
    from app.services.spy_history import backtest_sector_rotation
    return await backtest_sector_rotation(start_date=start_date, end_date=end_date)

@router.get("/crash-deploy/status")
async def crash_deploy_status():
    from app.services.crash_deploy_live import get_crash_deploy_state
    return await get_crash_deploy_state()


@router.post("/crash-deploy/flags")
async def crash_deploy_flags(enabled: bool = None, dry_run: bool = None):
    from app.services.crash_deploy_live import set_crash_deploy_flags
    return await set_crash_deploy_flags(enabled=enabled, dry_run=dry_run)


@router.post("/crash-deploy/check")
async def crash_deploy_check():
    """Valuta il crash deploy ORA (rispetta flag + dry_run)."""
    from app.services.crash_deploy_live import check_and_deploy
    return await check_and_deploy()


# Il MacroAnalyst salva tutto il contesto in db.market_context "latest",
# ma finora nessuna route lo esponeva: la Dashboard poteva leggere solo
# il riassunto nel SharedBrain, che non contiene focus_sectors,
# avoid_sectors, flusso intraday e forza relativa settoriale.


@router.get("/market-context")
async def get_market_context():
    """
    Contesto macro completo prodotto dal MacroAnalyst v3.3.

    Restituisce regime dettagliato, leadership settoriale, flusso
    intraday e forza relativa per settore su piu' finestre temporali.
    """
    db = get_db()
    doc = await db.market_context.find_one({"_id": "latest"})

    if not doc:
        return {"available": False, "message": "Nessun market context disponibile"}

    leadership = doc.get("leadership") or {}

    # Solo i campi che servono al frontend: il documento completo contiene
    # anche serie e dettagli che appesantirebbero inutilmente la risposta.
    sectors = [
        {
            "sector": s.get("sector"),
            "status": s.get("status"),
            "flow": s.get("flow"),
            "rs_intraday": s.get("rs_intraday", 0),
            "rs_short": s.get("rs_short", 0),
            "rs_swing": s.get("rs_swing", 0),
            "rs_structural": s.get("rs_structural", 0),
            "rank_swing": s.get("rank_swing"),
            "rank_structural": s.get("rank_structural"),
            "acceleration": s.get("acceleration", 0),
        }
        for s in (leadership.get("sectors") or [])
    ]

    return {
        "available": True,
        "analyzed_at": doc.get("analyzed_at"),
        "strategy_version": "macro_analyst_v3_3",

        "regime": doc.get("market_regime"),
        "regime_detail": doc.get("regime_detail"),
        "regime_detail_reason": doc.get("regime_detail_reason"),
        "confidence": doc.get("regime_confidence"),
        "raw_confidence": doc.get("regime_raw_confidence"),
        "regime_changed": doc.get("regime_changed", False),
        "exposure_multiplier": doc.get("exposure_multiplier"),

        "volatility": doc.get("volatility_regime"),
        "breadth_pct": doc.get("breadth_pct"),
        "market_breadth": doc.get("market_breadth"),
        "breadth_divergence": doc.get("breadth_divergence"),

        "leadership_state": leadership.get("state"),
        "leadership_description": leadership.get("description"),
        "rotation_state": doc.get("rotation_state"),
        "rotation_signal": doc.get("rotation_signal"),
        "concentration_swing": leadership.get("concentration_swing", 0),
        "participation_score": leadership.get("participation_score", 50),

        "focus_sectors": doc.get("focus_sectors", []),
        "avoid_sectors": doc.get("avoid_sectors", []),
        "emerging_leaders": doc.get("emerging_leaders", []),
        "fading_leaders": doc.get("fading_leaders", []),
        "inflow_sectors": doc.get("inflow_sectors", []),
        "outflow_sectors": doc.get("outflow_sectors", []),
        "spike_sectors": doc.get("spike_sectors", []),
        "flow_summary": doc.get("flow_summary"),
        "intraday_available": doc.get("intraday_available", False),
        "intraday_spy_move": leadership.get("intraday_spy_move", 0),

        "sectors": sectors,
        "top_sectors": leadership.get("top_sectors", []),
        "top_sectors_structural": leadership.get("top_sectors_structural", []),

        "timeframes": doc.get("timeframes", {}),
        "crash_radar": doc.get("crash_radar", {}),
        "sector_bottom": doc.get("sector_bottom", {}),

        "llm_reasoning": doc.get("llm_reasoning"),
    }

