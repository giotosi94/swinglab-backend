"""
Backtest Data Collector — genera training data ML da dati storici.
Simula entry su stock_bars, calcola le 15 features, traccia outcome WIN/LOSS.
Salva in ml_training_data per il training ibrido.
"""

import numpy as np
from datetime import datetime
from app.db.mongodb import get_db
from app.ml.features import features_to_array, SETUP_ENCODE, REGIME_ENCODE


def _rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50.0
    d = np.diff(prices)
    g = np.where(d > 0, d, 0)
    l = np.where(d < 0, -d, 0)
    ag = np.mean(g[-period:])
    al = np.mean(l[-period:])
    if al == 0:
        return 100.0
    return round(100 - (100 / (1 + ag / al)), 2)


def _ema(prices, period):
    if len(prices) < period:
        return prices[-1] if len(prices) else 0
    k = 2 / (period + 1)
    e = np.mean(prices[:period])
    for p in prices[period:]:
        e = p * k + e * (1 - k)
    return e


def _macd_hist(prices):
    if len(prices) < 26:
        return 0
    ema12 = _ema(prices, 12)
    ema26 = _ema(prices, 26)
    macd = ema12 - ema26
    # signal semplificato
    return round(macd * 0.2, 4)


def _build_features(bars_slice, sector_code):
    """Estrae le 15 features nel formato features.py da bars storici."""
    closes = [b["c"] for b in bars_slice]
    highs = [b["h"] for b in bars_slice]
    lows = [b["l"] for b in bars_slice]
    volumes = [b["v"] for b in bars_slice]

    price = closes[-1]
    rsi = _rsi(closes)
    ema10 = _ema(closes, 10)
    ema20 = _ema(closes, 20)
    ema50 = _ema(closes, 50)

    # EMA alignment
    if price > ema10 > ema20 > ema50:
        ema_align = 2
    elif price > ema20 > ema50:
        ema_align = 1
    else:
        ema_align = 0

    avg_vol = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
    rel_vol = round(volumes[-1] / avg_vol, 2) if avg_vol > 0 else 1
    ret_20d = ((price - closes[-21]) / closes[-21] * 100) if len(closes) >= 21 else 0
    change_pct = ((price - closes[-2]) / closes[-2] * 100) if len(closes) >= 2 else 0

    high_52w = max(highs)
    pct_from_high = ((price - high_52w) / high_52w * 100) if high_52w > 0 else -50
    low_52w = min(lows)
    range_pos = ((price - low_52w) / (high_52w - low_52w) * 100) if (high_52w - low_52w) > 0 else 50

    # setup type
    if price > ema10 > ema20 > ema50:
        setup = "breakout"
    elif abs(price - ema20) / price < 0.02:
        setup = "ema_bounce"
    elif rsi < 40:
        setup = "pullback_to_poc"
    else:
        setup = "neutral"

    # confluence semplificata (0-100)
    conf = 0
    if ema_align == 2:
        conf += 25
    elif ema_align == 1:
        conf += 15
    if 40 <= rsi <= 60:
        conf += 20
    elif 30 <= rsi < 40:
        conf += 12
    if ret_20d > 5:
        conf += 15
    elif ret_20d > 0:
        conf += 8
    if rel_vol >= 1.5:
        conf += 15
    elif rel_vol >= 1.0:
        conf += 8
    if closes[-1] > closes[-5]:
        conf += 10
    conf = min(conf, 100)

    # POC distance semplificato (usa VA proxy = media 20d)
    poc_proxy = np.mean(closes[-20:])
    poc_dist = round(abs(price - poc_proxy) / price * 100, 2) if price > 0 else 50

    features = {
        "rsi": rsi,
        "macd_histogram": _macd_hist(closes),
        "ema_alignment": ema_align,
        "relative_volume": rel_vol,
        "poc_distance_pct": poc_dist,
        "setup_type_encoded": SETUP_ENCODE.get(setup, 5),
        "sector_rank": 6,  # neutro, non disponibile storicamente
        "wyckoff_encoded": 5,  # neutro
        "accumulation_score": 0,
        "range_position": round(range_pos, 1),
        "change_pct": round(change_pct, 2),
        "regime_encoded": REGIME_ENCODE.get("NEUTRAL", 1),
        "confluence_score": round(conf, 1),
        "has_bullish_patterns": 0,
        "pct_from_high": round(pct_from_high, 2),
    }
    return features, conf, price


async def collect_backtest_training_data(
    days: int = 250,
    min_confluence: float = 55,
    stop_loss_pct: float = 6.0,
    take_profit_pct: float = 12.0,
    max_hold_days: int = 30,
):
    """
    Simula trade su dati storici e salva features + outcome in ml_training_data.
    Ogni posizione = 1 training sample (no scale-out duplication).
    """
    db = get_db()

    all_bars = await db.stock_bars.find({}).to_list(300)
    ticker_bars = {}
    all_dates = set()

    # Mappa sector
    assets_meta = await db.assets.find({}, {"ticker": 1, "sector_code": 1}).to_list(300)
    ticker_sector = {a["ticker"]: a.get("sector_code", "") for a in assets_meta}

    for doc in all_bars:
        ticker = doc.get("ticker")
        bars = doc.get("bars", [])
        if len(bars) >= 60:
            ticker_bars[ticker] = bars
            for b in bars:
                all_dates.add(b["date"])

    if not ticker_bars:
        return {"error": "No stock_bars data"}

    sorted_dates = sorted(all_dates)
    bt_dates = sorted_dates[-days:] if len(sorted_dates) > days else sorted_dates

    samples = []
    open_positions = {}  # ticker -> {entry_idx, entry_price, sl, tp, features, entry_date}

    for date in bt_dates:
        # Check exits
        for ticker in list(open_positions.keys()):
            pos = open_positions[ticker]
            bars = ticker_bars.get(ticker, [])
            bar = next((b for b in bars if b["date"] == date), None)
            if not bar:
                continue

            high = bar["h"]
            low = bar["l"]
            days_held = (
                sorted_dates.index(date) - sorted_dates.index(pos["entry_date"])
            )

            outcome = None
            if high >= pos["tp"]:
                outcome = 1  # WIN
            elif low <= pos["sl"]:
                outcome = 0  # LOSS
            elif days_held >= max_hold_days:
                # Chiusura per tempo: WIN se close > entry
                outcome = 1 if bar["c"] > pos["entry_price"] else 0

            if outcome is not None:
                samples.append({
                    "ticker": ticker,
                    "entry_date": pos["entry_date"],
                    "exit_date": date,
                    "features": pos["features"],
                    "features_array": features_to_array(pos["features"]),
                    "label": outcome,
                    "confluence": pos["confluence"],
                    "source": "backtest_collector",
                    "created_at": datetime.utcnow(),
                })
                del open_positions[ticker]

        # Check entries
        for ticker, bars in ticker_bars.items():
            if ticker in open_positions:
                continue
            idx = next((i for i, b in enumerate(bars) if b["date"] == date), None)
            if idx is None or idx < 55:
                continue
            bars_slice = bars[:idx + 1]
            features, conf, price = _build_features(
                bars_slice, ticker_sector.get(ticker, "")
            )
            if conf >= min_confluence:
                open_positions[ticker] = {
                    "entry_date": date,
                    "entry_price": price,
                    "sl": price * (1 - stop_loss_pct / 100),
                    "tp": price * (1 + take_profit_pct / 100),
                    "features": features,
                    "confluence": conf,
                }

    # Chiudi posizioni rimaste (label su ultimo close)
    last_date = bt_dates[-1]
    for ticker, pos in open_positions.items():
        bars = ticker_bars.get(ticker, [])
        bar = next((b for b in bars if b["date"] == last_date), None)
        if bar:
            outcome = 1 if bar["c"] > pos["entry_price"] else 0
            samples.append({
                "ticker": ticker,
                "entry_date": pos["entry_date"],
                "exit_date": last_date,
                "features": pos["features"],
                "features_array": features_to_array(pos["features"]),
                "label": outcome,
                "confluence": pos["confluence"],
                "source": "backtest_collector",
                "created_at": datetime.utcnow(),
            })

    # Wipe vecchi dati e salva nuovi
    await db.ml_training_data.delete_many({"source": "backtest_collector"})
    if samples:
        await db.ml_training_data.insert_many(samples)

    wins = sum(1 for s in samples if s["label"] == 1)
    losses = len(samples) - wins

    return {
        "message": "Backtest training data collected",
        "total_samples": len(samples),
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / len(samples) * 100, 1) if samples else 0,
        "period": {"start": bt_dates[0], "end": bt_dates[-1]},
    }
