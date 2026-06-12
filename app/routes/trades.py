from fastapi import APIRouter, Query
from app.db.mongodb import get_db
from datetime import datetime, timedelta

router = APIRouter()


@router.get("/history")
async def get_trade_history(limit: int = Query(default=50)):
    """Trade history con P&L in % e $."""
    db = get_db()
    trades = await db.trade_history.find(
        {"side": "sell"}
    ).sort("date", -1).to_list(limit)
    for t in trades:
        t["_id"] = str(t["_id"])
        # Calcola P&L in $
        entry = t.get("entry_price", 0)
        exit_p = t.get("exit_price", 0)
        shares = t.get("shares", 0)
        # Se non abbiamo shares nel sell, cercalo nel buy corrispondente
        if not shares:
            buy = await db.trade_history.find_one(
                {"ticker": t.get("ticker"), "side": "buy"},
                sort=[("date", -1)]
            )
            if buy:
                shares = buy.get("shares", 0)
                if not entry:
                    entry = buy.get("entry_price", 0)
        pnl_pct = t.get("pnl_pct", 0)
        pnl_dollar = round((exit_p - entry) * shares, 2) if entry and exit_p and shares else 0
        t["pnl_dollar"] = pnl_dollar
        t["shares"] = shares
        t["entry_price"] = round(entry, 2) if entry else 0
        t["exit_price"] = round(exit_p, 2) if exit_p else 0
    return trades


@router.get("/daily")
async def get_daily_summary():
    """P&L di oggi."""
    db = get_db()
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    trades_today = await db.trade_history.find(
        {"date": {"$gte": today}}
    ).to_list(100)

    buys = [t for t in trades_today if t.get("side") == "buy"]
    sells = [t for t in trades_today if t.get("side") == "sell"]

    total_pnl_pct = sum(t.get("pnl_pct", 0) for t in sells)
    wins = sum(1 for t in sells if t.get("pnl_pct", 0) > 0)

    return {
        "date": today.strftime("%Y-%m-%d"),
        "buys": len(buys),
        "sells": len(sells),
        "wins": wins,
        "losses": len(sells) - wins,
        "total_pnl_pct": round(total_pnl_pct, 2),
        "trades": [{
            "ticker": t.get("ticker"),
            "side": t.get("side"),
            "pnl_pct": t.get("pnl_pct", 0),
            "reason": t.get("reason", ""),
            "setup_type": t.get("setup_type", ""),
            "date": t.get("date").isoformat() if t.get("date") else "",
        } for t in trades_today],
    }


@router.get("/open")
async def get_open_trades():
    """Trade aperti (buy senza sell corrispondente)."""
    db = get_db()
    buys = await db.trade_history.find(
        {"side": "buy"}
    ).sort("date", -1).to_list(100)

    open_trades = []
    for b in buys:
        ticker = b.get("ticker")
        # Check if there is a corresponding sell
        sell = await db.trade_history.find_one(
            {"ticker": ticker, "side": "sell", "date": {"$gt": b.get("date")}},
            sort=[("date", 1)]
        )
        if not sell:
            b["_id"] = str(b["_id"])
            open_trades.append(b)

    return open_trades
@router.delete("/clear-all")
async def clear_all_trades():
    db = get_db()
    result = await db.trade_history.delete_many({"side": "sell"})
    return {"deleted": result.deleted_count}
