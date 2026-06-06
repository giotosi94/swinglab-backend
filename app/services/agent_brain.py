from datetime import datetime
from app.db.mongodb import get_db


async def analyze_performance():
    """Analyze all closed trades and learn what works"""
    db = get_db()
    trades = await db.trade_history.find({"side": "sell"}).to_list(500)

    if len(trades) < 3:
        return get_default_params()

    wins = [t for t in trades if t.get("pnl_pct", 0) > 0]
    losses = [t for t in trades if t.get("pnl_pct", 0) <= 0]
    total = len(trades)
    win_rate = len(wins) / total * 100 if total > 0 else 50

    # Analyze by setup type
    setup_stats = {}
    for t in trades:
        st = t.get("setup_type", "unknown")
        if st not in setup_stats:
            setup_stats[st] = {"wins": 0, "losses": 0, "total_pnl": 0}
        if t.get("pnl_pct", 0) > 0:
            setup_stats[st]["wins"] += 1
        else:
            setup_stats[st]["losses"] += 1
        setup_stats[st]["total_pnl"] += t.get("pnl_pct", 0)

    # Best and worst setups
    best_setups = []
    worst_setups = []
    for st, stats in setup_stats.items():
        total_st = stats["wins"] + stats["losses"]
        if total_st >= 2:
            wr = stats["wins"] / total_st * 100
            if wr >= 60:
                best_setups.append(st)
            elif wr < 40:
                worst_setups.append(st)

    # Analyze by sector
    sector_stats = {}
    for t in trades:
        sec = t.get("sector", "unknown")
        if sec not in sector_stats:
            sector_stats[sec] = {"wins": 0, "losses": 0}
        if t.get("pnl_pct", 0) > 0:
            sector_stats[sec]["wins"] += 1
        else:
            sector_stats[sec]["losses"] += 1

    weak_sectors = []
    for sec, stats in sector_stats.items():
        total_sec = stats["wins"] + stats["losses"]
        if total_sec >= 3 and stats["wins"] / total_sec < 0.35:
            weak_sectors.append(sec)

    # Analyze by confluence level
    conf_stats = {"high": {"wins": 0, "losses": 0}, "mid": {"wins": 0, "losses": 0}, "low": {"wins": 0, "losses": 0}}
    for t in trades:
        conf = t.get("confluence", 5)
        bucket = "high" if conf >= 7 else ("mid" if conf >= 5.5 else "low")
        if t.get("pnl_pct", 0) > 0:
            conf_stats[bucket]["wins"] += 1
        else:
            conf_stats[bucket]["losses"] += 1

    # Determine optimal confluence threshold
    min_confluence = 5.5
    low_total = conf_stats["low"]["wins"] + conf_stats["low"]["losses"]
    if low_total >= 3:
        low_wr = conf_stats["low"]["wins"] / low_total
        if low_wr < 0.4:
            min_confluence = 6.0
        elif low_wr < 0.3:
            min_confluence = 6.5

    mid_total = conf_stats["mid"]["wins"] + conf_stats["mid"]["losses"]
    if mid_total >= 3:
        mid_wr = conf_stats["mid"]["wins"] / mid_total
        if mid_wr > 0.6:
            min_confluence = 5.0

    # Analyze holding period
    avg_win_days = 0
    avg_loss_days = 0
    if wins:
        days_list = [t.get("days_held", 5) for t in wins if t.get("days_held")]
        avg_win_days = sum(days_list) / len(days_list) if days_list else 5
    if losses:
        days_list = [t.get("days_held", 5) for t in losses if t.get("days_held")]
        avg_loss_days = sum(days_list) / len(days_list) if days_list else 10

    # Optimal max hold
    max_hold_days = 15
    if avg_loss_days > 10 and avg_win_days < 7:
        max_hold_days = 10
    elif avg_win_days > 10:
        max_hold_days = 20

    # RSI analysis
    rsi_losses = [t.get("rsi_at_entry", 50) for t in losses if t.get("rsi_at_entry")]
    max_rsi = 68
    if rsi_losses:
        avg_loss_rsi = sum(rsi_losses) / len(rsi_losses)
        if avg_loss_rsi > 62:
            max_rsi = 60
        elif avg_loss_rsi > 55:
            max_rsi = 65

    # Build learned parameters
    params = {
        "min_confluence": round(min_confluence, 1),
        "max_rsi_entry": max_rsi,
        "max_hold_days": max_hold_days,
        "best_setups": best_setups if best_setups else ["pullback_to_poc", "ema_bounce", "breakout"],
        "worst_setups": worst_setups,
        "weak_sectors": weak_sectors,
        "win_rate": round(win_rate, 1),
        "total_trades": total,
        "avg_win_pnl": round(sum(t.get("pnl_pct", 0) for t in wins) / len(wins), 2) if wins else 0,
        "avg_loss_pnl": round(sum(t.get("pnl_pct", 0) for t in losses) / len(losses), 2) if losses else 0,
        "setup_stats": {k: {"win_rate": round(v["wins"]/(v["wins"]+v["losses"])*100, 1) if (v["wins"]+v["losses"]) > 0 else 0, "trades": v["wins"]+v["losses"]} for k, v in setup_stats.items()},
        "sector_stats": {k: {"win_rate": round(v["wins"]/(v["wins"]+v["losses"])*100, 1) if (v["wins"]+v["losses"]) > 0 else 0, "trades": v["wins"]+v["losses"]} for k, v in sector_stats.items()},
        "conf_stats": {k: {"win_rate": round(v["wins"]/(v["wins"]+v["losses"])*100, 1) if (v["wins"]+v["losses"]) > 0 else 0, "trades": v["wins"]+v["losses"]} for k, v in conf_stats.items()},
        "updated_at": datetime.utcnow(),
    }

    # Save learned params
    await db.agent_brain.update_one(
        {"_id": "learned_params"},
        {"$set": params},
        upsert=True
    )

    print("AGENT BRAIN: Win rate {:.1f}%, min_conf={}, max_rsi={}, max_hold={}d".format(
        win_rate, min_confluence, max_rsi, max_hold_days))
    if best_setups:
        print("  Best setups: {}".format(best_setups))
    if worst_setups:
        print("  Worst setups (avoiding): {}".format(worst_setups))
    if weak_sectors:
        print("  Weak sectors (avoiding): {}".format(weak_sectors))

    return params


def get_default_params():
    return {
        "min_confluence": 5.5,
        "max_rsi_entry": 68,
        "max_hold_days": 15,
        "best_setups": ["pullback_to_poc", "ema_bounce", "breakout"],
        "worst_setups": [],
        "weak_sectors": [],
        "win_rate": 50,
        "total_trades": 0,
    }


async def get_learned_params():
    db = get_db()
    params = await db.agent_brain.find_one({"_id": "learned_params"})
    if params:
        params["_id"] = str(params["_id"])
        return params
    return get_default_params()


async def log_trade_decision(ticker, action, reason, details):
    db = get_db()
    await db.agent_decisions.insert_one({
        "ticker": ticker,
        "action": action,
        "reason": reason,
        "details": details,
        "date": datetime.utcnow(),
    })
