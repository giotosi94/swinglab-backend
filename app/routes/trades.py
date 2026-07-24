from fastapi import APIRouter, Query
from app.db.mongodb import get_db
from datetime import datetime, timedelta
import math

router = APIRouter()


async def _get_starting_capital():
    """Recupera il capitale iniziale reale da Alpaca (fallback 100000)."""
    starting_capital = 100000.0
    try:
        from app.services.alpaca_trader import get_portfolio_history
        history = await get_portfolio_history(period="1A", timeframe="1D")
        if history and history.get("equity"):
            for eq in history.get("equity", []):
                if eq and eq > 0:
                    starting_capital = round(eq, 2)
                    break
    except Exception as e:
        print(f"  starting_capital fetch error: {e}")
    return starting_capital


@router.get("/history")
async def get_trade_history(limit: int = Query(default=50)):
    """Trade history con P&L in % e $."""
    db = get_db()
    trades = await db.trade_history.find(
        {"side": "sell"}
    ).sort("date", -1).to_list(limit)
    for t in trades:
        t["_id"] = str(t["_id"])
        entry = t.get("entry_price", 0)
        exit_p = t.get("exit_price", 0)
        shares = t.get("shares", 0)
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
    """P&L di oggi (return reale in $ e % sul capitale)."""
    db = get_db()
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    trades_today = await db.trade_history.find(
        {"date": {"$gte": today}}
    ).to_list(100)

    buys = [t for t in trades_today if t.get("side") == "buy"]
    sells = [t for t in trades_today if t.get("side") == "sell"]

    starting_capital = await _get_starting_capital()
    total_pnl_dollar = sum(t.get("pnl_dollar", 0) for t in sells)
    total_pnl_pct = round((total_pnl_dollar / starting_capital * 100), 2) if starting_capital > 0 else 0
    wins = sum(1 for t in sells if t.get("pnl_pct", 0) > 0)

    return {
        "date": today.strftime("%Y-%m-%d"),
        "buys": len(buys),
        "sells": len(sells),
        "wins": wins,
        "losses": len(sells) - wins,
        "total_pnl_pct": total_pnl_pct,
        "total_pnl_dollar": round(total_pnl_dollar, 2),
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
    result = await db.trade_history.delete_many({})
    return {"deleted": result.deleted_count}


from pydantic import BaseModel


class TradeInsert(BaseModel):
    ticker: str
    side: str
    entry_price: float = 0
    shares: int = 0
    setup_type: str = "unknown"
    sector: str = "unknown"
    rsi_at_entry: float = 50
    market_regime: str = "NEUTRAL"
    agent: str = "manual"


@router.post("/insert")
async def insert_trade(trade: TradeInsert):
    db = get_db()
    doc = trade.dict()
    doc["date"] = datetime.utcnow()
    await db.trade_history.insert_one(doc)
    return {"inserted": trade.ticker}


# ============================================
# FASE 23: Advanced Analytics (v2.0 — equity-based)
# ============================================
@router.get("/analytics")
async def get_analytics():
    """
    Sharpe, Sortino, Drawdown, Monthly P&L, Regime/Day/Sector/Setup breakdown.
    v2.0 — Tutte le metriche cumulative sono calcolate su equity reale in $,
    NON sommando percentuali (che era matematicamente errato).
    """
    db = get_db()
    trades = await db.trade_history.find(
        {"side": "sell"}
    ).sort("date", 1).to_list(1000)

    if not trades or len(trades) < 2:
        return {"error": "Not enough trades", "count": len(trades)}

    starting_capital = await _get_starting_capital()

    # ============================================
    # BASIC STATS
    # ============================================
    wins = [t for t in trades if (t.get("pnl_pct") or 0) > 0]
    losses = [t for t in trades if (t.get("pnl_pct") or 0) <= 0]
    total = len(trades)
    win_rate = (len(wins) / total * 100) if total > 0 else 0

    win_pcts = [t.get("pnl_pct", 0) for t in wins]
    loss_pcts = [abs(t.get("pnl_pct", 0)) for t in losses]
    all_pcts = [t.get("pnl_pct", 0) for t in trades]

    avg_win = sum(win_pcts) / len(win_pcts) if win_pcts else 0
    avg_loss = sum(loss_pcts) / len(loss_pcts) if loss_pcts else 0

    # Total P&L: return REALE in $ diviso capitale iniziale (NO somma %)
    total_pnl_dollar = sum(t.get("pnl_dollar", 0) for t in trades)
    total_pnl_pct = round((total_pnl_dollar / starting_capital * 100), 2) if starting_capital > 0 else 0

    # ============================================
    # EXPECTANCY (edge medio per trade — si legge per trade)
    # ============================================
    wr = win_rate / 100
    lr = 1 - wr
    expectancy = round((wr * avg_win) - (lr * avg_loss), 3)

    # ============================================
    # PROFIT FACTOR ($ vinti / $ persi — dollar-based)
    # ============================================
    gross_profit_usd = sum(t.get("pnl_dollar", 0) for t in wins)
    gross_loss_usd = abs(sum(t.get("pnl_dollar", 0) for t in losses))
    profit_factor = round(gross_profit_usd / gross_loss_usd, 2) if gross_loss_usd > 0 else 999

    # ============================================
    # SHARPE RATIO (su distribuzione return per-trade)
    # ============================================
    if len(all_pcts) >= 5:
        mean_return = sum(all_pcts) / len(all_pcts)
        variance = sum((r - mean_return) ** 2 for r in all_pcts) / len(all_pcts)
        std_return = math.sqrt(variance) if variance > 0 else 0.001
        avg_days = sum(t.get("days_held", 1) for t in trades) / total
        trades_per_year = 252 / max(avg_days, 1)
        sharpe = round((mean_return / std_return) * math.sqrt(trades_per_year), 2)
    else:
        sharpe = 0
        mean_return = 0
        std_return = 0

    # ============================================
    # SORTINO RATIO (penalizza solo le discese)
    # ============================================
    if len(all_pcts) >= 5:
        negative_returns = [r for r in all_pcts if r < 0]
        if negative_returns:
            downside_variance = sum(r ** 2 for r in negative_returns) / len(all_pcts)
            downside_std = math.sqrt(downside_variance) if downside_variance > 0 else 0.001
            avg_days = sum(t.get("days_held", 1) for t in trades) / total
            trades_per_year = 252 / max(avg_days, 1)
            sortino = round((mean_return / downside_std) * math.sqrt(trades_per_year), 2)
        else:
            sortino = 999
    else:
        sortino = 0

    # ============================================
    # EQUITY CURVE + DRAWDOWN (su capitale reale in $)
    # ============================================
    equity = starting_capital
    peak = starting_capital
    drawdown_series = []
    max_drawdown = 0
    max_drawdown_date = ""

    for t in trades:
        equity += t.get("pnl_dollar", 0)
        if equity > peak:
            peak = equity
        dd_pct = ((equity - peak) / peak * 100) if peak > 0 else 0
        cum_return_pct = ((equity - starting_capital) / starting_capital * 100) if starting_capital > 0 else 0
        if dd_pct < max_drawdown:
            max_drawdown = dd_pct
            max_drawdown_date = str(t.get("date", ""))[:10]
        drawdown_series.append({
            "date": str(t.get("date", ""))[:10],
            "ticker": t.get("ticker", ""),
            "cum_pnl": round(cum_return_pct, 2),
            "drawdown": round(dd_pct, 2),
            "peak": round(peak, 2),
            "equity": round(equity, 2),
        })

    # ============================================
    # MAX CONSECUTIVE WINS / LOSSES
    # ============================================
    max_consec_wins = 0
    max_consec_losses = 0
    current_wins = 0
    current_losses = 0

    for t in trades:
        if (t.get("pnl_pct") or 0) > 0:
            current_wins += 1
            current_losses = 0
            max_consec_wins = max(max_consec_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_consec_losses = max(max_consec_losses, current_losses)

    # ============================================
    # WIN RATE PER REGIME (pnl % = contributo $ sul capitale)
    # ============================================
    regime_stats = {}
    for t in trades:
        regime = t.get("market_regime", "UNKNOWN")
        if regime not in regime_stats:
            regime_stats[regime] = {"total": 0, "wins": 0, "pnl_dollar": 0}
        regime_stats[regime]["total"] += 1
        regime_stats[regime]["pnl_dollar"] += t.get("pnl_dollar", 0)
        if (t.get("pnl_pct") or 0) > 0:
            regime_stats[regime]["wins"] += 1

    regime_breakdown = []
    for regime, s in regime_stats.items():
        regime_breakdown.append({
            "regime": regime,
            "total": s["total"],
            "wins": s["wins"],
            "win_rate": round(s["wins"] / s["total"] * 100, 1) if s["total"] > 0 else 0,
            "pnl": round((s["pnl_dollar"] / starting_capital * 100), 2) if starting_capital > 0 else 0,
            "pnl_dollar": round(s["pnl_dollar"], 2),
        })
    regime_breakdown.sort(key=lambda x: x["total"], reverse=True)

    # ============================================
    # WIN RATE PER DAY OF WEEK
    # ============================================
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    day_stats = {i: {"total": 0, "wins": 0, "pnl_dollar": 0} for i in range(7)}

    for t in trades:
        d = t.get("date")
        if d:
            dow = d.weekday() if hasattr(d, 'weekday') else 0
            day_stats[dow]["total"] += 1
            day_stats[dow]["pnl_dollar"] += t.get("pnl_dollar", 0)
            if (t.get("pnl_pct") or 0) > 0:
                day_stats[dow]["wins"] += 1

    day_breakdown = []
    for i in range(5):
        s = day_stats[i]
        day_breakdown.append({
            "day": day_names[i],
            "total": s["total"],
            "wins": s["wins"],
            "win_rate": round(s["wins"] / s["total"] * 100, 1) if s["total"] > 0 else 0,
            "pnl": round((s["pnl_dollar"] / starting_capital * 100), 2) if starting_capital > 0 else 0,
        })

    # ============================================
    # AVG HOLDING PERIOD: WINNERS vs LOSERS
    # ============================================
    win_days = [t.get("days_held", 0) for t in wins]
    loss_days = [t.get("days_held", 0) for t in losses]
    avg_hold_winners = round(sum(win_days) / len(win_days), 1) if win_days else 0
    avg_hold_losers = round(sum(loss_days) / len(loss_days), 1) if loss_days else 0
    avg_hold_all = round(sum(t.get("days_held", 0) for t in trades) / total, 1) if total > 0 else 0

    # ============================================
    # SECTOR P&L HEATMAP
    # ============================================
    sector_stats = {}
    for t in trades:
        sec = t.get("sector", t.get("setup_type", "unknown"))
        if sec not in sector_stats:
            sector_stats[sec] = {"total": 0, "wins": 0, "pnl_dollar": 0}
        sector_stats[sec]["total"] += 1
        sector_stats[sec]["pnl_dollar"] += t.get("pnl_dollar", 0)
        if (t.get("pnl_pct") or 0) > 0:
            sector_stats[sec]["wins"] += 1

    sector_breakdown = []
    for sec, s in sector_stats.items():
        sector_breakdown.append({
            "sector": sec,
            "total": s["total"],
            "wins": s["wins"],
            "win_rate": round(s["wins"] / s["total"] * 100, 1) if s["total"] > 0 else 0,
            "pnl": round((s["pnl_dollar"] / starting_capital * 100), 2) if starting_capital > 0 else 0,
            "pnl_dollar": round(s["pnl_dollar"], 2),
        })
    sector_breakdown.sort(key=lambda x: x["pnl_dollar"], reverse=True)

    # ============================================
    # MONTHLY P&L TABLE
    # ============================================
    monthly = {}
    for t in trades:
        d = t.get("date")
        if d:
            key = d.strftime("%Y-%m") if hasattr(d, 'strftime') else str(d)[:7]
            if key not in monthly:
                monthly[key] = {"pnl_dollar": 0, "trades": 0, "wins": 0}
            monthly[key]["pnl_dollar"] += t.get("pnl_dollar", 0)
            monthly[key]["trades"] += 1
            if (t.get("pnl_pct") or 0) > 0:
                monthly[key]["wins"] += 1

    monthly_table = []
    for month, s in sorted(monthly.items()):
        monthly_table.append({
            "month": month,
            "pnl": round((s["pnl_dollar"] / starting_capital * 100), 2) if starting_capital > 0 else 0,
            "pnl_dollar": round(s["pnl_dollar"], 2),
            "trades": s["trades"],
            "wins": s["wins"],
            "win_rate": round(s["wins"] / s["trades"] * 100, 1) if s["trades"] > 0 else 0,
        })

    # ============================================
    # SETUP TYPE STATS
    # ============================================
    setup_stats = {}
    for t in trades:
        st = t.get("setup_type", "unknown")
        if st not in setup_stats:
            setup_stats[st] = {"total": 0, "wins": 0, "pnl_dollar": 0}
        setup_stats[st]["total"] += 1
        setup_stats[st]["pnl_dollar"] += t.get("pnl_dollar", 0)
        if (t.get("pnl_pct") or 0) > 0:
            setup_stats[st]["wins"] += 1

    setup_breakdown = []
    for st, s in setup_stats.items():
        setup_breakdown.append({
            "setup": st,
            "total": s["total"],
            "wins": s["wins"],
            "win_rate": round(s["wins"] / s["total"] * 100, 1) if s["total"] > 0 else 0,
            "pnl": round((s["pnl_dollar"] / starting_capital * 100), 2) if starting_capital > 0 else 0,
        })
    setup_breakdown.sort(key=lambda x: x["pnl_dollar"], reverse=True)

    # ============================================
    # RETURN
    # ============================================
    return {
        "total_trades": total,
        "win_rate": round(win_rate, 1),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "sharpe": sharpe,
        "sortino": sortino,
        "total_pnl_pct": total_pnl_pct,
        "total_pnl_dollar": round(total_pnl_dollar, 2),
        "starting_capital": starting_capital,
        "current_equity": round(equity, 2),
        "max_drawdown": round(max_drawdown, 2),
        "max_drawdown_date": max_drawdown_date,
        "max_consec_wins": max_consec_wins,
        "max_consec_losses": max_consec_losses,
        "avg_hold_winners": avg_hold_winners,
        "avg_hold_losers": avg_hold_losers,
        "avg_hold_all": avg_hold_all,
        "drawdown_series": drawdown_series,
        "regime_breakdown": regime_breakdown,
        "day_breakdown": day_breakdown,
        "sector_breakdown": sector_breakdown,
        "monthly_table": monthly_table,
        "setup_breakdown": setup_breakdown,
    }
