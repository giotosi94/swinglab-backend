"""
Backtesting Engine v2.0 — Simula strategia SwingLab COMPLETA.
Include: APM Adaptive Targets, Scale-Out multi-target, Break-even SL,
Trailing stops, gestione posizioni realistica.
"""

import numpy as np
from datetime import datetime
from app.db.mongodb import get_db

_USE_MTF_BT = True


def _calc_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def _calc_ema(prices, period):
    if len(prices) < period:
        return prices[-1] if len(prices) else 0
    k = 2 / (period + 1)
    ema = np.mean(prices[:period])
    for p in prices[period:]:
        ema = p * k + ema * (1 - k)
    return ema


def _weekly_trend_bt(bars_slice):
    """MTF Light per backtest: raggruppa le daily in settimane (blocchi di 5)."""
    if len(bars_slice) < 60:
        return "UNKNOWN", "flat"
    closes = [b["c"] for b in bars_slice]
    # Weekly = ogni 5 barre daily, prendo l'ultima chiusura
    weekly = closes[4::5]
    if len(weekly) < 12:
        return "UNKNOWN", "flat"
    wprice = weekly[-1]
    wema10 = _calc_ema(weekly, 10)
    wema20 = _calc_ema(weekly, 20)
    wema50 = _calc_ema(weekly, min(50, len(weekly)))
    # Slope EMA20 sulle ultime 4 settimane
    wema20_prev = _calc_ema(weekly[:-4], 20) if len(weekly) > 24 else wema20
    if wema20 > wema20_prev * 1.005:
        slope = "rising"
    elif wema20 < wema20_prev * 0.995:
        slope = "falling"
    else:
        slope = "flat"
    if wprice > wema10 > wema20 > wema50 and wema50 > 0:
        trend = "BULL"
    elif wprice > wema20 > wema50 and wema50 > 0:
        trend = "BULL"
    elif wprice > wema50 and wema50 > 0:
        trend = "NEUTRAL"
    elif wprice > wema20:
        trend = "NEUTRAL"
    else:
        trend = "BEAR"
    return trend, slope


def _confluence_and_target(bars_slice):
    """
    Calcola confluence + target dinamico (come Alpha reale).
    Ritorna: (confluence, target_price, stop_price, setup_type)
    """
    if len(bars_slice) < 50:
        return 0, 0, 0, "none"

    closes = [b["c"] for b in bars_slice]
    highs = [b["h"] for b in bars_slice]
    lows = [b["l"] for b in bars_slice]
    volumes = [b["v"] for b in bars_slice]

    price = closes[-1]
    rsi = _calc_rsi(closes)
    ema10 = _calc_ema(closes, 10)
    ema20 = _calc_ema(closes, 20)
    ema50 = _calc_ema(closes, 50)

    avg_vol = np.mean(volumes[-20:])
    rel_vol = volumes[-1] / avg_vol if avg_vol > 0 else 1
    ret_20d = ((price - closes[-21]) / closes[-21] * 100) if len(closes) >= 21 else 0

    # ATR per volatility-aware targets
    atr_period = 14
    trs = []
    for i in range(max(1, len(bars_slice) - atr_period), len(bars_slice)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1]),
        )
        trs.append(tr)
    atr = np.mean(trs) if trs else price * 0.02
    atr_pct = (atr / price * 100) if price > 0 else 2.0

    # Confluence score
    score = 0
    if price > ema10 > ema20 > ema50:
        score += 25
    elif price > ema20 > ema50:
        score += 15
    elif price > ema50:
        score += 5
    if 40 <= rsi <= 60:
        score += 20
    elif 30 <= rsi < 40:
        score += 12
    elif rsi < 30:
        score += 8
    if ret_20d > 5:
        score += 15
    elif ret_20d > 0:
        score += 8
    if rel_vol >= 1.5:
        score += 15
    elif rel_vol >= 1.0:
        score += 8
    if closes[-1] > closes[-5]:
        score += 10

    # 🆕 MTF Weekly Alignment (proporzionale al peso live: ~11% del totale)
    wtrend, wslope = _weekly_trend_bt(bars_slice) if _USE_MTF_BT else ("UNKNOWN", "flat")
    if wtrend == "BULL" and wslope == "rising":
        score += 12
    elif wtrend == "BULL":
        score += 7
    elif wtrend == "NEUTRAL":
        score += 2
    elif wtrend == "BEAR":
        score -= 10

    score = max(0, min(score, 100))

    # Setup type
    if price > ema10 > ema20 > ema50:
        setup = "breakout"
    elif abs(price - ema20) / price < 0.02:
        setup = "ema_bounce"
    elif rsi < 40:
        setup = "pullback_to_poc"
    else:
        setup = "neutral"

    # 🎯 Target dinamico basato su ATR + setup (come Alpha reale)
    # Volatile (ATR alto) → target largo. Stabile → target stretto.
    target_multiplier = {
        "breakout": 4.0,
        "ema_bounce": 3.0,
        "pullback_to_poc": 3.5,
        "neutral": 3.0,
    }.get(setup, 3.0)

    target_distance_pct = min(40, max(4, atr_pct * target_multiplier))
    sl_distance_pct = min(12, max(3, atr_pct * 1.5))

    target_price = price * (1 + target_distance_pct / 100)
    stop_price = price * (1 - sl_distance_pct / 100)

    return score, target_price, stop_price, setup


def _calc_metrics(equity_curve, trades):
    if len(equity_curve) < 2:
        return {}
    equities = [e["equity"] for e in equity_curve]
    returns = np.diff(equities) / equities[:-1]
    total_return = (equities[-1] - equities[0]) / equities[0] * 100

    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
    else:
        sharpe = 0
    downside = returns[returns < 0]
    if len(downside) > 1 and np.std(downside) > 0:
        sortino = (np.mean(returns) / np.std(downside)) * np.sqrt(252)
    else:
        sortino = 0

    peak = equities[0]
    max_dd = 0
    for eq in equities:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd:
            max_dd = dd

    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
    avg_loss = abs(np.mean([t["pnl_pct"] for t in losses])) if losses else 0
    gross_win = sum(t["pnl_dollar"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl_dollar"] for t in losses)) if losses else 1
    profit_factor = gross_win / gross_loss if gross_loss > 0 else 0
    expectancy = (win_rate / 100 * avg_win) - ((100 - win_rate) / 100 * avg_loss)

    return {
        "total_return_pct": round(total_return, 2),
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "win_rate": round(win_rate, 1),
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "expectancy_pct": round(expectancy, 2),
    }


async def run_backtest(
    days: int = 180,
    min_confluence: float = 55,
    max_positions: int = 8,
    position_size_pct: float = 12.0,
    starting_capital: float = 100000,
    use_apm: bool = True,
    t1_ratio: float = 0.40,
    t2_ratio: float = 0.70,
    t3_ratio: float = 1.00,
    use_mtf: bool = True,
):
    """
    Backtest v2.0 con APM COMPLETO:
    - Adaptive targets (T1/T2/T3 = ratio × target Alpha)
    - Scale-out 50%/30%/20%
    - Break-even SL dopo T1
    - Trailing stop dopo T2
    """
    global _USE_MTF_BT
    _USE_MTF_BT = use_mtf
    db = get_db()
    all_bars = await db.stock_bars.find({}).to_list(300)
    if not all_bars:
        return {"error": "No stock_bars data available"}

    ticker_bars = {}
    all_dates = set()
    for doc in all_bars:
        ticker = doc.get("ticker")
        bars = doc.get("bars", [])
        if len(bars) >= 60:
            ticker_bars[ticker] = bars
            for b in bars:
                all_dates.add(b["date"])

    if not ticker_bars:
        return {"error": "Not enough bars"}

    sorted_dates = sorted(all_dates)
    backtest_dates = sorted_dates[-days:] if len(sorted_dates) > days else sorted_dates

    cash = starting_capital
    positions = {}
    trades = []
    equity_curve = []
    scale_out_events = 0

    for date in backtest_dates:
        # ===== 1. GESTIONE POSIZIONI APERTE (APM logic) =====
        for ticker in list(positions.keys()):
            pos = positions[ticker]
            bars = ticker_bars.get(ticker, [])
            bar = next((b for b in bars if b["date"] == date), None)
            if not bar:
                continue

            high = bar["h"]
            low = bar["l"]
            close = bar["c"]
            entry = pos["entry_price"]
            pos["last_price"] = close

            if use_apm:
                # APM Adaptive scale-out multi-target
                pnl_pct_high = (high - entry) / entry * 100

                # T1 hit (chiude 50%, SL → break-even)
                if pos["last_target_hit"] < 1 and pnl_pct_high >= pos["t1_pct"]:
                    exit_price = entry * (1 + pos["t1_pct"] / 100)
                    qty_close = pos["shares"] * 0.50
                    pnl_d = (exit_price - entry) * qty_close
                    cash += exit_price * qty_close
                    pos["shares"] -= qty_close
                    pos["last_target_hit"] = 1
                    pos["sl"] = entry  # break-even
                    scale_out_events += 1
                    trades.append({
                        "ticker": ticker, "entry_date": pos["entry_date"], "exit_date": date,
                        "entry_price": round(entry, 2), "exit_price": round(exit_price, 2),
                        "pnl_pct": round(pos["t1_pct"], 2), "pnl_dollar": round(pnl_d, 2),
                        "reason": "APM_SCALE_T1",
                    })

                # T2 hit (chiude 30%, SL → entry+3%)
                elif pos["last_target_hit"] == 1 and pnl_pct_high >= pos["t2_pct"]:
                    exit_price = entry * (1 + pos["t2_pct"] / 100)
                    qty_close = pos["shares"] * 0.60  # 30% del totale = 60% del residuo
                    pnl_d = (exit_price - entry) * qty_close
                    cash += exit_price * qty_close
                    pos["shares"] -= qty_close
                    pos["last_target_hit"] = 2
                    pos["sl"] = entry * 1.03
                    scale_out_events += 1
                    trades.append({
                        "ticker": ticker, "entry_date": pos["entry_date"], "exit_date": date,
                        "entry_price": round(entry, 2), "exit_price": round(exit_price, 2),
                        "pnl_pct": round(pos["t2_pct"], 2), "pnl_dollar": round(pnl_d, 2),
                        "reason": "APM_SCALE_T2",
                    })

                # T3 hit (chiude tutto il residuo)
                elif pos["last_target_hit"] == 2 and pnl_pct_high >= pos["t3_pct"]:
                    exit_price = entry * (1 + pos["t3_pct"] / 100)
                    pnl_d = (exit_price - entry) * pos["shares"]
                    cash += exit_price * pos["shares"]
                    trades.append({
                        "ticker": ticker, "entry_date": pos["entry_date"], "exit_date": date,
                        "entry_price": round(entry, 2), "exit_price": round(exit_price, 2),
                        "pnl_pct": round(pos["t3_pct"], 2), "pnl_dollar": round(pnl_d, 2),
                        "reason": "APM_SCALE_T3",
                    })
                    del positions[ticker]
                    continue

                # SL hit (su qualsiasi residuo)
                if low <= pos["sl"]:
                    exit_price = pos["sl"]
                    pnl_pct = (exit_price - entry) / entry * 100
                    pnl_d = (exit_price - entry) * pos["shares"]
                    cash += exit_price * pos["shares"]
                    reason = "BREAK_EVEN_SL" if pos["last_target_hit"] >= 1 else "STOP_LOSS"
                    trades.append({
                        "ticker": ticker, "entry_date": pos["entry_date"], "exit_date": date,
                        "entry_price": round(entry, 2), "exit_price": round(exit_price, 2),
                        "pnl_pct": round(pnl_pct, 2), "pnl_dollar": round(pnl_d, 2),
                        "reason": reason,
                    })
                    del positions[ticker]
                    continue

            else:
                # Modalità semplice (no APM): SL/TP fissi
                if low <= pos["sl"]:
                    exit_price = pos["sl"]
                    reason = "STOP_LOSS"
                elif high >= pos["tp"]:
                    exit_price = pos["tp"]
                    reason = "TAKE_PROFIT"
                else:
                    continue
                pnl_pct = (exit_price - entry) / entry * 100
                pnl_d = (exit_price - entry) * pos["shares"]
                cash += exit_price * pos["shares"]
                trades.append({
                    "ticker": ticker, "entry_date": pos["entry_date"], "exit_date": date,
                    "entry_price": round(entry, 2), "exit_price": round(exit_price, 2),
                    "pnl_pct": round(pnl_pct, 2), "pnl_dollar": round(pnl_d, 2),
                    "reason": reason,
                })
                del positions[ticker]

        # ===== 2. ENTRIES =====
        if len(positions) < max_positions:
            candidates = []
            for ticker, bars in ticker_bars.items():
                if ticker in positions:
                    continue
                idx = next((i for i, b in enumerate(bars) if b["date"] == date), None)
                if idx is None or idx < 50:
                    continue
                bars_slice = bars[:idx + 1]
                conf, target_price, stop_price, setup = _confluence_and_target(bars_slice)
                if conf >= min_confluence:
                    entry_price = bars_slice[-1]["c"]
                    candidates.append((ticker, conf, entry_price, target_price, stop_price, setup))

            candidates.sort(key=lambda x: x[1], reverse=True)
            slots = max_positions - len(positions)

            for ticker, conf, entry_price, target_price, stop_price, setup in candidates[:slots]:
                notional = cash * (position_size_pct / 100)
                if notional < 100 or notional > cash:
                    continue
                shares = notional / entry_price
                cash -= notional

                # Adaptive targets basati su target Alpha
                target_dist = (target_price - entry_price) / entry_price * 100

                positions[ticker] = {
                    "entry_price": entry_price,
                    "shares": shares,
                    "sl": stop_price,
                    "tp": target_price,  # per modalità semplice
                    "t1_pct": round(target_dist * t1_ratio, 2),
                    "t2_pct": round(target_dist * t2_ratio, 2),
                    "t3_pct": round(target_dist * t3_ratio, 2),
                    "last_target_hit": 0,
                    "entry_date": date,
                    "confluence": conf,
                    "setup": setup,
                    "last_price": entry_price,
                }

        # ===== 3. EQUITY =====
        positions_value = 0
        for ticker, pos in positions.items():
            bars = ticker_bars.get(ticker, [])
            bar = next((b for b in bars if b["date"] == date), None)
            if bar:
                positions_value += bar["c"] * pos["shares"]
                pos["last_price"] = bar["c"]
            else:
                positions_value += pos.get("last_price", pos["entry_price"]) * pos["shares"]
        total_equity = cash + positions_value
        equity_curve.append({"date": date, "equity": round(total_equity, 2)})

    # Chiudi posizioni residue
    last_date = backtest_dates[-1]
    for ticker, pos in positions.items():
        bars = ticker_bars.get(ticker, [])
        bar = next((b for b in bars if b["date"] == last_date), None)
        exit_price = bar["c"] if bar else pos.get("last_price", pos["entry_price"])
        pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"] * 100
        pnl_d = (exit_price - pos["entry_price"]) * pos["shares"]
        trades.append({
            "ticker": ticker, "entry_date": pos["entry_date"], "exit_date": last_date,
            "entry_price": round(pos["entry_price"], 2), "exit_price": round(exit_price, 2),
            "pnl_pct": round(pnl_pct, 2), "pnl_dollar": round(pnl_d, 2),
            "reason": "END_OF_BACKTEST",
        })

    metrics = _calc_metrics(equity_curve, trades)

    # SPY benchmark
    spy_return = 0
    spy_doc = await db.stock_bars.find_one({"ticker": "SPY"})
    if spy_doc:
        spy_bars = spy_doc.get("bars", [])
        spy_slice = [b for b in spy_bars if b["date"] in set(backtest_dates)]
        if len(spy_slice) >= 2:
            spy_return = (spy_slice[-1]["c"] - spy_slice[0]["c"]) / spy_slice[0]["c"] * 100

    return {
        "config": {
            "days": days,
            "min_confluence": min_confluence,
            "max_positions": max_positions,
            "position_size_pct": position_size_pct,
            "use_apm": use_apm,
            "t1_ratio": t1_ratio,
            "t2_ratio": t2_ratio,
            "t3_ratio": t3_ratio,
            "starting_capital": starting_capital,
        },
        "metrics": metrics,
        "benchmark": {
            "spy_return_pct": round(spy_return, 2),
            "alpha": round(metrics.get("total_return_pct", 0) - spy_return, 2),
        },
        "apm_stats": {
            "scale_out_events": scale_out_events,
            "apm_enabled": use_apm,
        },
        "equity_curve": equity_curve[::max(1, len(equity_curve) // 100)],
        "trades": sorted(trades, key=lambda x: x["exit_date"], reverse=True)[:60],
        "total_trades": len(trades),
        "period": {
            "start": backtest_dates[0] if backtest_dates else "",
            "end": backtest_dates[-1] if backtest_dates else "",
        },
    }
