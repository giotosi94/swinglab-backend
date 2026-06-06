from datetime import datetime
from app.db.mongodb import get_db
from app.services.alpaca_trader import (
    get_account, get_positions, place_bracket_order,
    place_order, close_position, get_latest_price
)
from app.services.agent_brain import analyze_performance, get_learned_params, log_trade_decision


async def run_auto_trader():
    db = get_db()

    account = await get_account()
    if not account:
        print("AutoTrader: Alpaca not connected")
        return {"error": "Alpaca not connected"}

    equity = float(account.get("equity", 0))
    cash = float(account.get("cash", 0))
    buying_power = float(account.get("buying_power", 0))

    # Learn from past trades
    brain = await analyze_performance()
    min_confluence = brain.get("min_confluence", 5.5)
    max_rsi = brain.get("max_rsi_entry", 68)
    max_hold = brain.get("max_hold_days", 15)
    best_setups = brain.get("best_setups", ["pullback_to_poc", "ema_bounce", "breakout"])
    worst_setups = brain.get("worst_setups", [])
    weak_sectors = brain.get("weak_sectors", [])

    # Market context
    spy_doc = await db.market_regime.find_one({"symbol": "SPY"})
    vix_doc = await db.market_regime.find_one({"symbol": "VIXY"})

    spy_rsi = spy_doc.get("rsi", 50) if spy_doc else 50
    spy_price = spy_doc.get("price", 0) if spy_doc else 0
    spy_ema50 = spy_doc.get("ema50", 0) if spy_doc else 0
    vix_price = vix_doc.get("price", 20) if vix_doc else 20

    # Market regime
    if spy_price > spy_ema50 and spy_rsi > 50:
        market_regime = "BULL"
        regime_multiplier = 1.0
    elif spy_price > spy_ema50:
        market_regime = "NEUTRAL"
        regime_multiplier = 0.6
    elif spy_rsi > 35:
        market_regime = "BEAR"
        regime_multiplier = 0.3
    else:
        market_regime = "CRASH"
        regime_multiplier = 0.0

    # VIX adjustment
    if vix_price > 30:
        regime_multiplier *= 0.5
    elif vix_price > 25:
        regime_multiplier *= 0.7

    print("=" * 50)
    print("AUTO-TRADER RUNNING (AI Agent)")
    print("Equity: ${:.2f} | Cash: ${:.2f}".format(equity, cash))
    print("Market: {} | SPY ${} RSI {} | VIX {} | Size: {:.0f}%".format(
        market_regime, spy_price, round(spy_rsi), vix_price, regime_multiplier * 100))
    print("Brain: min_conf={}, max_rsi={}, max_hold={}d".format(min_confluence, max_rsi, max_hold))
    if worst_setups:
        print("  Avoiding setups: {}".format(worst_setups))
    if weak_sectors:
        print("  Avoiding sectors: {}".format(weak_sectors))
    print("=" * 50)

    positions = await get_positions() or []
    open_tickers = [p.get("symbol") for p in positions]
    open_sectors = []

    assets = await db.assets.find().to_list(300)
    sectors = await db.sectors.find().sort("composite_score", -1).to_list(20)

    if not assets:
        print("No assets data")
        return {"error": "No assets data"}

    sector_codes = [s["code"] for s in sectors]
    asset_map = {a["ticker"]: a for a in assets}

    for t in open_tickers:
        a = asset_map.get(t)
        if a:
            open_sectors.append(a.get("sector_code", ""))

    actions = []

    # ============================================
    # STEP 1: CHECK SELLS
    # ============================================
    for p in positions:
        symbol = p.get("symbol")
        asset = asset_map.get(symbol)
        if not asset:
            continue

        current_price = float(p.get("current_price", 0))
        entry_price = float(p.get("avg_entry_price", 0))
        pnl_pct = float(p.get("unrealized_plpc", 0)) * 100
        rsi = asset.get("rsi", 50)
        setup_score = asset.get("setup_score", 50)

        sell_reason = None

        if rsi > 78 and pnl_pct > 3:
            sell_reason = "RSI_EXTREME"
        elif setup_score < 20 and pnl_pct < -2:
            sell_reason = "SCORE_COLLAPSED"
        elif pnl_pct < -1:
            bearish = [pat for pat in asset.get("candlestick_patterns", []) if pat.get("type") == "bearish" and pat.get("strength") == "strong"]
            if bearish:
                sell_reason = "BEARISH_PATTERN"
        elif asset.get("wyckoff", {}).get("phase") in ("distribution", "markdown") and pnl_pct < 0:
            sell_reason = "WYCKOFF_BEARISH"

        if sell_reason:
            print("  SELL {}: {} (P&L {:.1f}%)".format(symbol, sell_reason, pnl_pct))
            result = await close_position(symbol)
            if result is not None:
                actions.append({"action": "SELL", "ticker": symbol, "reason": sell_reason, "pnl_pct": round(pnl_pct, 2)})

                await db.trade_history.insert_one({
                    "ticker": symbol, "side": "sell", "entry_price": entry_price,
                    "exit_price": current_price, "pnl_pct": round(pnl_pct, 2),
                    "reason": sell_reason, "rsi_at_entry": rsi,
                    "setup_type": asset.get("setup_type", "unknown"),
                    "sector": asset.get("sector_code", "unknown"),
                    "market_regime": market_regime,
                    "date": datetime.utcnow(),
                })

                await log_trade_decision(symbol, "SELL", sell_reason,
                    {"pnl_pct": round(pnl_pct, 2), "rsi": rsi, "score": setup_score, "regime": market_regime})

    positions = await get_positions() or []
    open_tickers = [p.get("symbol") for p in positions]
    num_positions = len(positions)

    # ============================================
    # STEP 2: CHECK BUYS
    # ============================================
    max_positions = 5
    max_per_position = equity * 0.20 * regime_multiplier
    risk_per_trade = equity * 0.02 * regime_multiplier

    if market_regime == "CRASH":
        print("CRASH MODE - no new buys, staying in cash")
        num_positions = max_positions

    if num_positions >= max_positions:
        print("Max positions reached ({}/{})".format(num_positions, max_positions))
    else:
        candidates = []

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
            wyckoff = a.get("wyckoff", {})
            accum = a.get("accumulation", {})

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

            wyckoff_signal = wyckoff.get("signal", "neutral")
            if wyckoff_signal in ("strong_bullish", "bullish_soon"):
                confluence += 1.5
            elif wyckoff_signal == "bullish":
                confluence += 0.5
            elif wyckoff_signal in ("bearish", "bearish_soon"):
                confluence -= 2

            accum_score = accum.get("score", 0)
            if accum_score >= 70:
                confluence += 1
            elif accum_score >= 40:
                confluence += 0.5

            # AI Agent filters
            if confluence < min_confluence:
                continue
            if rsi > max_rsi or rsi < 25:
                continue
            if stype not in best_setups:
                continue
            if stype in worst_setups:
                continue
            if sector in weak_sectors:
                confluence -= 1
            if price <= 1:
                continue
            if rel_vol >= 3.0:
                print("    SKIP {} - extreme volume {:.1f}x (possible earnings)".format(ticker, rel_vol))
                continue
            sector_count = open_sectors.count(sector)
            if sector_count >= 2:
                continue

            stop_loss = va_low if va_low and va_low < price else round(price * 0.96, 2)
            target_price = va_high if va_high and va_high > price else round(price * 1.08, 2)
            risk_per_share = abs(price - stop_loss)
            if risk_per_share <= 0.01:
                continue

            shares = min(
                int(risk_per_trade / risk_per_share),
                int(max_per_position / price) if max_per_position > 0 else 0,
                int(buying_power * 0.9 / price) if buying_power > 0 else 0
            )
            if shares <= 0:
                continue

            candidates.append({
                "ticker": ticker, "price": price, "confluence": round(confluence, 1),
                "score": score, "shares": shares, "stop_loss": round(stop_loss, 2),
                "target_price": round(target_price, 2), "sector": sector,
                "setup_type": stype, "rsi": rsi,
            })

        candidates.sort(key=lambda x: x["confluence"], reverse=True)

        for c in candidates:
            if num_positions >= max_positions:
                break

            ticker = c["ticker"]
            shares = c["shares"]
            target = c["target_price"]
            stop = c["stop_loss"]

            print("  BUY {}: {} shares, target ${}, stop ${}, confluence {}".format(
                ticker, shares, target, stop, c["confluence"]))

            result = await place_bracket_order(
                symbol=ticker, qty=shares,
                limit_price=c["price"] * 1.005,
                take_profit=target, stop_loss=stop
            )

            if result:
                num_positions += 1
                open_tickers.append(ticker)
                open_sectors.append(c["sector"])
                buying_power -= c["price"] * shares

                actions.append({
                    "action": "BUY", "ticker": ticker, "shares": shares,
                    "price": c["price"], "target": target, "stop": stop,
                    "confluence": c["confluence"], "setup": c["setup_type"],
                })

                await db.trade_history.insert_one({
                    "ticker": ticker, "side": "buy", "entry_price": c["price"],
                    "shares": shares, "target": target, "stop_loss": stop,
                    "confluence": c["confluence"], "setup_type": c["setup_type"],
                    "sector": c["sector"], "rsi_at_entry": c["rsi"],
                    "market_regime": market_regime,
                    "date": datetime.utcnow(),
                })

                await log_trade_decision(ticker, "BUY",
                    "Confluence {}, setup {}, RSI {}, regime {}".format(c["confluence"], c["setup_type"], c["rsi"], market_regime),
                    {"confluence": c["confluence"], "setup": c["setup_type"], "sector": c["sector"],
                     "rsi": c["rsi"], "regime": market_regime, "vix": vix_price,
                     "brain_params": {"min_conf": min_confluence, "max_rsi": max_rsi}})
            else:
                print("  FAILED to buy {}".format(ticker))

    # ============================================
    # STEP 3: SAVE STATE
    # ============================================
    state = {
        "last_run": datetime.utcnow().isoformat(),
        "equity": equity,
        "cash": cash,
        "positions": len(open_tickers),
        "actions": actions,
        "market": {
            "regime": market_regime,
            "spy_price": spy_price,
            "spy_rsi": round(spy_rsi),
            "vix": vix_price,
            "size_multiplier": round(regime_multiplier * 100),
        },
        "brain": {
            "min_confluence": min_confluence,
            "max_rsi": max_rsi,
            "max_hold_days": max_hold,
            "best_setups": best_setups,
            "worst_setups": worst_setups,
            "weak_sectors": weak_sectors,
            "win_rate": brain.get("win_rate", 50),
        },
        "updated_at": datetime.utcnow(),
    }
    await db.auto_trader.update_one(
        {"_id": "alpaca_state"}, {"$set": state}, upsert=True
    )

    print("\nAUTO-TRADER DONE: {} actions".format(len(actions)))
    for a in actions:
        print("  {} {} {}".format(a["action"], a["ticker"], a.get("reason", "")))

    return {
        "equity": equity, "cash": cash, "positions": num_positions,
        "actions": actions, "regime": market_regime,
    }


async def reset_auto_trader(initial_capital=10000):
    from app.services.alpaca_trader import close_all_positions, cancel_all_orders
    await cancel_all_orders()
    await close_all_positions()
    db = get_db()
    await db.trade_history.delete_many({})
    await db.agent_brain.delete_many({})
    await db.agent_decisions.delete_many({})
    return {"message": "All positions closed, history and brain cleared"}


async def get_auto_trader_state():
    db = get_db()
    state = await db.auto_trader.find_one({"_id": "alpaca_state"})
    if state:
        state["_id"] = str(state["_id"])
    return state
