from fastapi import APIRouter, Query
from typing import Optional
from app.db.mongodb import get_db

router = APIRouter()

# Ticker esclusi dall'universo trade (indici/ETF macro + residui ricerche)
EXCLUDED_TICKERS = [
    "SPY", "QQQ", "IWM", "DIA", "VXX", "VIXY", "TLT", "HYG",
    "LQD", "GLD", "USO", "RSP", "IWO", "FXE", "UUP", "EEM", "IYT",
]


def _base_query(sector: Optional[str], min_score: float) -> dict:
    query = {
        "sector_code": {"$ne": "SEARCH"},
        "ticker": {"$nin": EXCLUDED_TICKERS},
    }

    if sector:
        query["sector_code"] = sector.upper()

    if min_score > 0:
        query["setup_score"] = {"$gte": min_score}

    return query


@router.get("/")
async def get_assets(
    sector: Optional[str] = None,
    min_score: float = 0,
    sort_by: str = Query("setup_score"),
    limit: int = 50,
):
    """
    Lista asset alleggerita per dashboard e viste generali.
    Esclude i blocchi pesanti (Max Strategy, pattern, analisi LLM).
    """
    db = get_db()
    query = _base_query(sector, min_score)

    projection = {
        "price_history": 0,
        "vp_distribution": 0,
        "multi_tf_vp": 0,
        "fvg": 0,
        "history": 0,
        "max_strategy": 0,
        "llm_analysis": 0,
        "candlestick_patterns": 0,
        "wyckoff": 0,
        "accumulation": 0,
    }

    assets = await db.assets.find(query, projection).to_list(limit)

    for asset in assets:
        asset["_id"] = str(asset["_id"])

    assets.sort(key=lambda item: item.get(sort_by, 0) or 0, reverse=True)
    return assets


@router.get("/overview")
async def get_assets_overview(
    sector: Optional[str] = None,
    min_score: float = 0,
    sort_by: str = Query("setup_score"),
    limit: int = 305,
):
    """
    Overview compatta per la pagina Stock.
    Restituisce solo i campi necessari alla lista e allo stato Max Strategy.
    """
    db = get_db()
    query = _base_query(sector, min_score)

    projection = {
        "ticker": 1,
        "name": 1,
        "sector_code": 1,
        "price": 1,
        "change_pct": 1,
        "setup_score": 1,
        "setup_type": 1,
        "rsi": 1,
        "relative_volume": 1,
        "data_status": 1,
        "data_eligible": 1,
        "last_bar_date": 1,
        "calendar_days_old": 1,
        "max_strategy.version": 1,
        "max_strategy.max_score": 1,
        "max_strategy.strategy_type": 1,
        "max_strategy.strategy_eligible": 1,
        "max_strategy.trade_ready": 1,
        "max_strategy.market_phase.phase": 1,
        "max_strategy.market_phase.state": 1,
        "max_strategy.entry_plan.status": 1,
        "max_strategy.entry_plan.order_action": 1,
        "max_strategy.entry_plan.execution_mode": 1,
        "max_strategy.entry_plan.trigger_price": 1,
        "max_strategy.entry_plan.maximum_entry_price": 1,
        "max_strategy.entry_plan.invalidation_price": 1,
    }

    assets = await db.assets.find(query, projection).to_list(limit)

    for asset in assets:
        asset["_id"] = str(asset["_id"])

    assets.sort(key=lambda item: item.get(sort_by, 0) or 0, reverse=True)
    return assets


@router.get("/{ticker}")
async def get_asset(ticker: str):
    db = get_db()
    asset = await db.assets.find_one({"ticker": ticker.upper()})

    if asset:
        asset["_id"] = str(asset["_id"])

    return asset or {"error": "Asset not found"}
