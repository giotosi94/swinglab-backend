from datetime import datetime
from app.db.mongodb import get_db


async def run_auto_trader():
    db = get_db()

    # Get or create auto-trader state
    state = await db.auto_trader.find_one({"_id": "state"})
    if not state:
        state = {
            "_id": "state",
            "capital": 10000,
            "initial_capital": 10000,
            "cash": 10000,
            "open_positions": [],
            "closed_trades": [],
            "equity_history": [],
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "created_at": datetime.utcnow(),
        }
        await db.auto_trader.insert_one(state)

    assets = await db.assets.find().to_list(200)
    sectors = await db.sectors.find().sort("composite_score", -1).to_list(20)

    if not assets:
        return {"message": "No assets data"}

    capital = state.get("capital", 10000)
    cash = state.get("cash", 10000)
    open_positions = state.get("open_positions", [])
    closed_trades = state.get("closed_trades", [])
    equity_history = state.get("equity_history", [])
    total_trades = state.get("total_trades", 0)
    wins = state.get("wins", 0)
    losses = state.get("losses", 0)

    asset_map = {a["ticker"]: a for a in assets}
    sector_codes = [s["code"] for s in sectors]

    actions = []

    # ============================================
    # STEP 1: CHECK SELLS
    # ============================================
    new_open = []
    for pos in open_positions:
        ticker = pos["ticker"]
        asset = asset_map.get(ticker)
        if not asset:
            new_open.append(pos)
            continue

        current_price = asset.get("price", pos["entry_price"])
        entry_price = pos["entry_price"]
        shares = pos["shares"]
        days_held = (datetime.utcnow() - datetime.fromisoformat(pos["entry_date"])).days
        pnl_pct = ((current_price - entry_price) / entry_price) * 100
        rsi = asset.get("rsi", 50)
        setup_score = asset.get("setup_score", 50)
        target = pos.get("target_price", entry_price * 1.08)
        stop = pos.get("stop_loss", entry_price * 0.96)

        sell_reason = None

        # Target hit
        if current_price >= target:
            sell_reason = "TARGET_HIT"
        # Stop loss hit
        elif current_price <= stop:
            sell_reason = "STOP_LOSS"
        # RSI overbought
        elif rsi > 75 and pnl_pct > 2:
            sell_reason = "RSI_OVERBOUGHT"
        # Score collapsed
        elif setup_score < 25 and pnl_pct < 0:
            sell_reason = "SCORE_COLLAPSED"
        # Max hold time
        elif days_held > 15:
            sell_reason = "MAX_HOLD_TIME"
        # Trailing stop: if was up 5%+ and now dropping
        elif pnl_pct < -4:
            sell_reason = "HARD_STOP"

        if sell_reason:
            pnl = (current_price - entry_price) * shares
            pnl_pct_final = ((current_price - entry_price) / entry_price) * 100
            cash += current_price * shares

            closed_trade = {
                "ticker": ticker,
                "entry_price": entry_price,
                "exit_price": round(current_price, 2),
                "shares": shares,
                "entry_date": pos["entry_date"],
                "exit_date": datetime.utcnow().isoformat(),
                "days_held": days_held,
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct_final, 2),
                "reason": sell_reason,
                "sector": pos.get("sector", ""),
            }
            closed_trades.append(closed_trade)
            total_trades += 1
            if pnl > 0:
                wins += 1
            else:
                losses += 1

            actions.append({
                "action": "SELL",
                "ticker": ticker,
                "price": round(current_price, 2),
                "shares": shares,
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct_final, 2),
                "reason": sell_reason,
            })
        else:
            pos["current_price"] = round(current_price, 2)
            pos["pnl"] = round((current_price - entry_price) * shares, 2)
            pos["pnl_pct"] = round(pnl_pct, 2)
            pos["days_held"] = days_held
            pos["rsi"] = round(rsi, 1)
            pos["setup_score"] = setup_score
            new_open.append(pos)

    open_positions = new_open

    # ============================================
    # STEP 2: CHECK BUYS
    # ============================================
    max_positions = 5
    max_per_position = capital * 0.20
    risk_per_trade = capital * 0.02

    if len(open_positions) < max_positions:
        # Score all assets for buying
        candidates = []
        open_tickers = [p["ticker"] for p in open_positions]
        open_sectors = [p.get("sector", "") for p in open_positions]

        for a in assets:
            ticker = a.get("ticker", "")
            if ticker in open_tickers:
                continue

            score = a.get("setup_score", 0)
            rsi = a.get("rsi", 50)
            stype = a.get("setup_type", "")
            sector = a.get("sector_code", "")
            price = a.get("price", 0)
            poc = a.get("poc_price")
            va_high = a.get("value_area_high")
            va_low = a.get("value_area_low")
            rel_vol = a.get("relative_volume", 1)
            macd_hist = a.get("macd", {}).get("histogram", 0)
            ema10 = a.get("ema10", 0)
            ema20 = a.get("ema20", 0)
            ema50 = a.get("ema50", 0)
            patterns = a.get("candlestick_patterns", [])
            bullish_patterns = [p for p in patterns if p.get("type") == "bullish"]

            # Calculate confluence
            confluence = 0
            if poc and price and abs(price - poc) / price * 100 <= 2:
                confluence += 2
            if bullish_patterns:
                confluence += 1.5
            if 40 <= rsi <= 60:
                confluence += 1
            if macd_hist > 0:
                confluence += 1
            if price > ema10 > ema20 > ema50:
                confluence += 1.5
            elif price > ema20 > ema50:
                confluence += 0.75
            if rel_vol >= 1.5:
                confluence += 1
            sector_rank = sector_codes.index(sector) + 1 if sector in sector_codes else 11
            if sector_rank <= 5:
                confluence += 1
            pct_from_high = a.get("pct_from_high", -50)
            if pct_from_high and pct_from_high >= -10:
                confluence += 0.5
            change = a.get("change_pct", 0)
            if 0 < change <= 5:
                confluence += 0.5

            # Filter conditions
            if confluence < 6:
                continue
            if rsi > 65 or rsi < 25:
                continue
            if stype not in ("breakout", "pullback_to_poc", "ema_bounce"):
                continue
            if price <= 0:
                continue
            # Max 2 per sector
            sector_count = len([p for p in open_positions if p.get("sector") == sector])
            if sector_count >= 2:
                continue

            # Calculate position
            stop_loss = va_low if va_low and va_low < price else price * 0.96
            target_price = va_high if va_high and va_high > price else price * 1.08
            risk_per_share = abs(price - stop_loss)
            if risk_per_share <= 0:
                continue

            shares = min(
                int(risk_per_trade / risk_per_share),
                int(max_per_position / price),
                int(cash / price)
            )
            if shares <= 0:
                continue

            candidates.append({
                "ticker": ticker,
                "price": price,
                "confluence": confluence,
                "score": score,
                "shares": shares,
                "stop_loss": round(stop_loss, 2),
                "target_price": round(target_price, 2),
                "risk_per_share": round(risk_per_share, 2),
                "sector": sector,
                "setup_type": stype,
                "rsi": rsi,
            })

        # Sort by confluence, take top ones
        candidates.sort(key=lambda x: x["confluence"], reverse=True)

        for c in candidates:
            if len(open_positions) >= max_positions:
                break
            if cash < c["price"] * c["shares"]:
                continue

            cost = c["price"] * c["shares"]
            cash -= cost

            new_position = {
                "ticker": c["ticker"],
                "entry_price": round(c["price"], 2),
                "current_price": round(c["price"], 2),
                "shares": c["shares"],
                "stop_loss": c["stop_loss"],
                "target_price": c["target_price"],
                "entry_date": datetime.utcnow().isoformat(),
                "sector": c["sector"],
                "setup_type": c["setup_type"],
                "confluence": round(c["confluence"], 1),
                "pnl": 0,
                "pnl_pct": 0,
                "days_held": 0,
                "rsi": round(c["rsi"], 1),
                "setup_score": c.get("score", 0),
            }
            open_positions.append(new_position)
            total_trades += 1

            actions.append({
                "action": "BUY",
                "ticker": c["ticker"],
                "price": round(c["price"], 2),
                "shares": c["shares"],
                "stop_loss": c["stop_loss"],
                "target": c["target_price"],
                "confluence": round(c["confluence"], 1),
                "setup": c["setup_type"],
            })

    # ============================================
    # STEP 3: CALCULATE EQUITY
    # ============================================
    positions_value = sum(p.get("current_price", p["entry_price"]) * p["shares"] for p in open_positions)
    total_equity = cash + positions_value

    equity_point = {
        "date": datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
        "equity": round(total_equity, 2),
        "cash": round(cash, 2),
        "positions_value": round(positions_value, 2),
        "open_count": len(open_positions),
        "pnl_pct": round(((total_equity - state.get("initial_capital", 10000)) / state.get("initial_capital", 10000)) * 100, 2),
    }
    equity_history.append(equity_point)

    # Keep last 200 equity points
    if len(equity_history) > 200:
        equity_history = equity_history[-200:]

    # Keep last 100 closed trades
    if len(closed_trades) > 100:
        closed_trades = closed_trades[-100:]

    # ============================================
    # STEP 4: SAVE STATE
    # ============================================
    win_rate = round((wins / total_trades * 100), 1) if total_trades > 0 else 0
    avg_pnl = round(sum(t.get("pnl", 0) for t in closed_trades) / len(closed_trades), 2) if closed_trades else 0
    total_pnl = round(sum(t.get("pnl", 0) for t in closed_trades), 2)
    best_trade = max(closed_trades, key=lambda t: t.get("pnl", 0)) if closed_trades else None
    worst_trade = min(closed_trades, key=lambda t: t.get("pnl", 0)) if closed_trades else None

    update = {
        "capital": state.get("initial_capital", 10000),
        "initial_capital": state.get("initial_capital", 10000),
        "cash": round(cash, 2),
        "equity": round(total_equity, 2),
        "open_positions": open_positions,
        "closed_trades": closed_trades,
        "equity_history": equity_history,
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "total_pnl": total_pnl,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "last_run": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow(),
    }

    await db.auto_trader.update_one({"_id": "state"}, {"$set": update}, upsert=True)

    return {
        "equity": round(total_equity, 2),
        "cash": round(cash, 2),
        "open_positions": len(open_positions),
        "actions": actions,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
    }


async def reset_auto_trader(initial_capital=10000):
    db = get_db()
    state = {
        "_id": "state",
        "capital": initial_capital,
        "initial_capital": initial_capital,
        "cash": initial_capital,
        "equity": initial_capital,
        "open_positions": [],
        "closed_trades": [],
        "equity_history": [{
            "date": datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
            "equity": initial_capital,
            "cash": initial_capital,
            "positions_value": 0,
            "open_count": 0,
            "pnl_pct": 0,
        }],
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": 0,
        "avg_pnl": 0,
        "total_pnl": 0,
        "best_trade": None,
        "worst_trade": None,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    await db.auto_trader.delete_many({})
    await db.auto_trader.insert_one(state)
    return state


async def get_auto_trader_state():
    db = get_db()
    state = await db.auto_trader.find_one({"_id": "state"})
    if state:
        state["_id"] = str(state["_id"])
    return state
