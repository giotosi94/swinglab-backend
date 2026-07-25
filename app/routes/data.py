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
    
    async def _run_in_background():
        try:
            print(f"[ASYNC] Pipeline started at {datetime.utcnow().isoformat()}")
            results = await fetch_and_analyze_stocks()
            trader_result = await run_auto_trader()
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
    sectors = await fetch_and_analyze_sectors()
    stocks = await fetch_and_analyze_stocks()
    trader_result = await run_auto_trader()
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
):
    """v2.1 Backtest con APM + parametri dal preset di rischio attivo."""
    from app.services.backtesting import run_backtest
    db = get_db()

    # 🆕 v2.1 — Usa i parametri del preset attivo (se non forzati via query)
    preset_name = None
    if use_preset:
        settings = await db.app_settings.find_one({"_id": "risk_params"})
        if settings:
            preset_name = settings.get("active_preset")
            if max_positions is None:
                max_positions = settings.get("max_positions", 8)
            if position_size_pct is None:
                position_size_pct = settings.get("position_size_pct", 12.0)
            if min_confluence is None:
                # deriva soglia dal min_risk_reward: profili aggressivi = soglia più bassa
                mrr = settings.get("min_risk_reward", 1.5)
                min_confluence = 45 if mrr <= 1.3 else (50 if mrr <= 1.5 else 55)

    # Fallback finali
    max_positions = max_positions if max_positions is not None else 8
    position_size_pct = position_size_pct if position_size_pct is not None else 12.0
    min_confluence = min_confluence if min_confluence is not None else 55

    result = await run_backtest(
        days=days,
        min_confluence=min_confluence,
        max_positions=max_positions,
        position_size_pct=position_size_pct,
        use_apm=use_apm,
        t1_ratio=t1_ratio,
        t2_ratio=t2_ratio,
        t3_ratio=t3_ratio,
    )
    result["active_preset"] = preset_name
    return result
