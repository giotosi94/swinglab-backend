"""
Backtesting Engine v1.0 — Simula strategia SwingLab su dati storici.
Riusa stock_bars MongoDB + logica confluence semplificata.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from app.db.mongodb import get_db


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


def _simple_confluence(bars_slice):
    """
    Confluence semplificata su bars storici (0-100).
    Usa RSI, EMA alignment, momentum, volume.
    """
    if len(bars_slice) < 50:
        return 0, {}

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

    # Return 20d
    ret_20d = ((price - closes[-21]) / closes[-21] * 100) if len(closes) >= 21 else 0

    score = 0
    # EMA alignment
    if price > ema10 > ema20 > ema50:
        score += 25
    elif price > ema20 > ema50:
        score += 15
    elif price > ema50:
        score += 5
    # RSI sweet spot
    if 40 <= rsi <= 60:
        score += 20
    elif 30 <= rsi < 40:
        score += 12
    elif rsi < 30:
        score += 8
    # Momentum
    if ret_20d > 5:
        score += 15
    elif ret_20d > 0:
        score += 8
    # Volume
    if rel_vol >= 1.5:
        score += 15
    elif rel_vol >= 1.0:
        score += 8
    # Trend positivo recente
    if closes[-1] > closes[-5]:
        score += 10

    details = {
        "price": round(price, 2),
        "rsi": rsi,
        "ema50": round(ema50, 2),
        "ret_20d": round(ret_20d, 2),
        "rel_vol": round(rel_vol, 2),
    }
    return min(score, 100), details


def _calc_metrics(equity_curve, trades):
    """Calcola metriche performance da equity curve + trades."""
    if len(equity_curve) < 2:
        return {}

    equities = [e["equity"] for e in equity_curve]
    returns = np.diff(equities) / equities[:-1]

    total_return = (equities[-1] - equities[0]) / equities[0] * 100

    # Sharpe (annualizzato, risk-free 0)
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
    else:
        sharpe = 0

    # Sortino (solo downside deviation)
    downside = returns[returns < 0]
    if len(downside) > 1 and np.std(downside) > 0:
        sortino = (np.mean(returns) / np.std(downside)) * np.sqrt(252)
    else:
        sortino = 0

    # Max Drawdown
    peak = equities[0]
    max_dd = 0
    for eq in equities:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd:
            max_dd = dd

    # Trade stats
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
    stop_loss_pct: float = 6.0,
    take_profit_pct: float = 12.0,
    max_positions: int = 8,
    position_size_pct: float = 12.0,
    starting_capital: float = 100000,
):
    """
    Esegue backtest della strategia su dati storici.

    Simula: per ogni giorno, scansiona stock, entra se confluence >= soglia,
    esce a SL/TP. Traccia equity curve e metriche.
    """
    db = get_db()

    # Carica tutti i bars
    all_bars = await db.stock_bars.find({}).to_list(300)
    if not all_bars:
        return {"error": "No stock_bars data available"}

    # Costruisci mappa ticker -> bars ordinati
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
        return {"error": "Not enough bars (need 60+ per stock)"}

    # Date ordinate, ultimi N giorni
    sorted_dates = sorted(all_dates)
    backtest_dates = sorted_dates[-days:] if len(sorted_dates) > days else sorted_dates

    # Simulazione
    cash = starting_capital
    positions = {}  # ticker -> {entry_price, shares, sl, tp, entry_date}
    trades = []
    equity_curve = []

    for date in backtest_dates:
        # 1. Check exits su posizioni aperte
        for ticker in list(positions.keys()):
            pos = positions[ticker]
            bars = ticker_bars.get(ticker, [])
            bar = next((b for b in bars if b["date"] == date), None)
            if not bar:
                continue

            high = bar["h"]
            low = bar["l"]
            close = bar["c"]
            exit_price = None
            reason = None

            # SL hit
            if low <= pos["sl"]:
                exit_price = pos["sl"]
                reason = "STOP_LOSS"
            # TP hit
            elif high >= pos["tp"]:
                exit_price = pos["tp"]
                reason = "TAKE_PROFIT"

            if exit_price:
                pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"] * 100
                pnl_dollar = (exit_price - pos["entry_price"]) * pos["shares"]
                cash += exit_price * pos["shares"]
                trades.append({
                    "ticker": ticker,
                    "entry_date": pos["entry_date"],
                    "exit_date": date,
                    "entry_price": round(pos["entry_price"], 2),
                    "exit_price": round(exit_price, 2),
                    "pnl_pct": round(pnl_pct, 2),
                    "pnl_dollar": round(pnl_dollar, 2),
                    "reason": reason,
                })
                del positions[ticker]

        # 2. Check entries (se c'è spazio)
        if len(positions) < max_positions:
            candidates = []
            for ticker, bars in ticker_bars.items():
                if ticker in positions:
                    continue
                # bars fino a questa data (inclusa)
                idx = next((i for i, b in enumerate(bars) if b["date"] == date), None)
                if idx is None or idx < 50:
                    continue
                bars_slice = bars[:idx + 1]
                conf, det = _simple_confluence(bars_slice)
                if conf >= min_confluence:
                    candidates.append((ticker, conf, det, bars_slice[-1]["c"]))

            # Ordina per confluence, prendi i migliori
            candidates.sort(key=lambda x: x[1], reverse=True)
            slots = max_positions - len(positions)

            for ticker, conf, det, entry_price in candidates[:slots]:
                notional = cash * (position_size_pct / 100)
                if notional < 100 or notional > cash:
                    continue
                shares = notional / entry_price
                cash -= notional
                positions[ticker] = {
                    "entry_price": entry_price,
                    "shares": shares,
                    "sl": entry_price * (1 - stop_loss_pct / 100),
                    "tp": entry_price * (1 + take_profit_pct / 100),
                    "entry_date": date,
                    "confluence": conf,
                }

        # 3. Calcola equity totale (cash + posizioni aperte a close)
        positions_value = 0
        for ticker, pos in positions.items():
            bars = ticker_bars.get(ticker, [])
            bar = next((b for b in bars if b["date"] == date), None)
            if bar:
                positions_value += bar["c"] * pos["shares"]
        total_equity = cash + positions_value
        equity_curve.append({"date": date, "equity": round(total_equity, 2)})

    # Chiudi posizioni rimaste all'ultima data
    last_date = backtest_dates[-1]
    for ticker, pos in positions.items():
        bars = ticker_bars.get(ticker, [])
        bar = next((b for b in bars if b["date"] == last_date), None)
        if bar:
            exit_price = bar["c"]
            pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"] * 100
            pnl_dollar = (exit_price - pos["entry_price"]) * pos["shares"]
            trades.append({
                "ticker": ticker,
                "entry_date": pos["entry_date"],
                "exit_date": last_date,
                "entry_price": round(pos["entry_price"], 2),
                "exit_price": round(exit_price, 2),
                "pnl_pct": round(pnl_pct, 2),
                "pnl_dollar": round(pnl_dollar, 2),
                "reason": "END_OF_BACKTEST",
            })

    metrics = _calc_metrics(equity_curve, trades)

    # SPY benchmark
    spy_return = 0
    spy_bars = ticker_bars.get("SPY")
    if not spy_bars:
        spy_doc = await db.stock_bars.find_one({"ticker": "SPY"})
        spy_bars = spy_doc.get("bars", []) if spy_doc else []
    if spy_bars:
        spy_slice = [b for b in spy_bars if b["date"] in set(backtest_dates)]
        if len(spy_slice) >= 2:
            spy_return = (spy_slice[-1]["c"] - spy_slice[0]["c"]) / spy_slice[0]["c"] * 100

    return {
        "config": {
            "days": days,
            "min_confluence": min_confluence,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
            "max_positions": max_positions,
            "position_size_pct": position_size_pct,
            "starting_capital": starting_capital,
        },
        "metrics": metrics,
        "benchmark": {
            "spy_return_pct": round(spy_return, 2),
            "alpha": round(metrics.get("total_return_pct", 0) - spy_return, 2),
        },
        "equity_curve": equity_curve[::max(1, len(equity_curve) // 100)],
        "trades": sorted(trades, key=lambda x: x["exit_date"], reverse=True)[:50],
        "total_trades": len(trades),
        "period": {
            "start": backtest_dates[0] if backtest_dates else "",
            "end": backtest_dates[-1] if backtest_dates else "",
        },
    }
