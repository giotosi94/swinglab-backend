"""
Backtest Data Collector v2.1 — genera training data ML da dati storici.
Calcola TUTTE le 15 features (incluse regime, wyckoff, accumulation, bullish patterns).
Ogni posizione = 1 training sample (no scale-out duplication).
"""

import numpy as np
from datetime import datetime
from app.db.mongodb import get_db
from app.ml.features import features_to_array, SETUP_ENCODE, REGIME_ENCODE, WYCKOFF_ENCODE


# ============================================
# INDICATOR HELPERS
# ============================================

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
    return round(macd * 0.2, 4)


def _detect_wyckoff(closes, volumes, highs, lows):
    """Wyckoff phase encoded. 0=accum 1=markup 2=spring 3=distrib 4=markdown 5=neutral."""
    if len(closes) < 30:
        return 5
    price = closes[-1]
    ema20 = _ema(closes, 20)
    ema50 = _ema(closes, 50)
    pc_20 = ((price / closes[-20]) - 1) * 100
    rng_20 = ((max(closes[-20:]) - min(closes[-20:])) / closes[-20]) * 100

    vol_old = np.mean(volumes[-40:-20]) if len(volumes) >= 40 else np.mean(volumes[:20])
    vol_new = np.mean(volumes[-20:])
    vol_chg = ((vol_new - vol_old) / vol_old * 100) if vol_old > 0 else 0

    if rng_20 < 10 and vol_chg < -10:
        return 0  # accumulation
    elif pc_20 > 5 and price > ema20 > ema50:
        return 1  # markup
    elif rng_20 < 10 and price > ema50:
        return 3  # distribution
    elif pc_20 < -5 and price < ema20:
        return 4  # markdown

    if len(lows) >= 10:
        recent_low = min(lows[-10:])
        prev_low = min(lows[-30:-10]) if len(lows) >= 30 else min(lows[:20])
        pc_10 = ((price / closes[-10]) - 1) * 100
        if recent_low < prev_low and pc_10 > 3:
            return 2  # spring
    return 5


def _accumulation_score(closes, volumes, highs, lows):
    """Accumulation score 0-100 da volume + price action."""
    if len(closes) < 20:
        return 0
    price = closes[-1]
    poc_proxy = np.mean(closes[-20:])
    score = 0
    if price < poc_proxy:
        dist = abs(price - poc_proxy) / poc_proxy * 100
        if dist <= 5:
            score += 25
        elif dist <= 15:
            score += 15
    low_20 = min(lows[-20:])
    if price <= low_20 * 1.02:
        score += 20
    if len(volumes) >= 20:
        v1 = np.mean(volumes[-20:-10])
        v2 = np.mean(volumes[-10:])
        if v1 > 0 and (v2 - v1) / v1 * 100 < -15:
            score += 15
    return min(score, 100)


def _bullish_pattern(closes, opens, highs, lows):
    """Detect bullish candlestick pattern (1 se presente)."""
    if len(closes) < 2:
        return 0
    o, c, h, l = opens[-1], closes[-1], highs[-1], lows[-1]
    body = abs(c - o)
    rng = h - l
    if rng == 0:
        return 0
    lower_shadow = min(o, c) - l
    upper_shadow = h - max(o, c)
    body_pct = body / rng
    if body_pct < 0.35 and lower_shadow >= body * 2 and upper_shadow < body * 0.5:
        return 1
    if len(closes) >= 2:
        o1, c1 = opens[-2], closes[-2]
        if c > o and c1 < o1 and c > o1 and o < c1:
            return 1
    return 0


def _regime_from_spy(spy_closes, date_idx):
    """Regime encoded da SPY trend. 0=BULL 1=NEUTRAL 2=BEAR 3=CRASH."""
    if not spy_closes or date_idx < 50:
        return 1
    window = spy_closes[max(0, date_idx - 50):date_idx + 1]
    if len(window) < 20:
        return 1
    price = window[-1]
    ema50 = _ema(window, 50) if len(window) >= 50 else _ema(window, 20)
    ret_20 = ((price / window[-20]) - 1) * 100 if len(window) >= 20 else 0
    if ret_20 > 5 and price > ema50:
        return 0  # BULL
    elif ret_20 < -10:
        return 3  # CRASH
    elif ret_20 < -3 or price < ema50:
        return 2  # BEAR
    return 1  # NEUTRAL


# ============================================
# FEATURE BUILDER
# ============================================

def _build_features(bars_slice, sector_code, sector_rank=6, regime_enc=1):
    """Estrae le 15 features nel formato features.py da bars storici."""
    closes = [b["c"] for b in bars_slice]
    opens = [b.get("o", b["c"]) for b in bars_slice]
    highs = [b["h"] for b in bars_slice]
    lows = [b["l"] for b in bars_slice]
    volumes = [b["v"] for b in bars_slice]

    price = closes[-1]
    rsi = _rsi(closes)
    ema10 = _ema(closes, 10)
    ema20 = _ema(closes, 20)
    ema50 = _ema(closes, 50)

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

    if price > ema10 > ema20 > ema50:
        setup = "breakout"
    elif abs(price - ema20) / price < 0.02:
        setup = "ema_bounce"
    elif rsi < 40:
        setup = "pullback_to_poc"
    else:
        setup = "neutral"

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

    poc_proxy = np.mean(closes[-20:])
    poc_dist = round(abs(price - poc_proxy) / price * 100, 2) if price > 0 else 50

    features = {
        "rsi": rsi,
        "macd_histogram": _macd_hist(closes),
        "ema_alignment": ema_align,
        "relative_volume": rel_vol,
        "poc_distance_pct": poc_dist,
        "setup_type_encoded": SETUP_ENCODE.get(setup, 5),
        "sector_rank": sector_rank,
        "wyckoff_encoded": _detect_wyckoff(closes, volumes, highs, lows),
        "accumulation_score": _accumulation_score(closes, volumes, highs, lows),
        "range_position": round(range_pos, 1),
        "change_pct": round(change_pct, 2),
        "regime_encoded": regime_enc,
        "confluence_score": round(conf, 1),
        "has_bullish_patterns": _bullish_pattern(closes, opens, highs, lows),
        "pct_from_high": round(pct_from_high, 2),
    }
    return features, conf, price


# ============================================
# MAIN COLLECTOR
# ============================================

async def collect_backtest_training_data(
    days: int = 250,
    min_confluence: float = 55,
    stop_loss_pct: float = 6.0,
    take_profit_pct: float = 12.0,
    max_hold_days: int = 30,
):
    """
    Simula trade su dati storici e salva features + outcome in ml_training_data.
    v2.1: tutte le 15 features calcolate realmente.
    """
    db = get_db()

    all_bars = await db.stock_bars.find({}).to_list(300)

    # SPY per regime detection
    spy_doc = await db.stock_bars.find_one({"ticker": "SPY"})
    spy_bars = spy_doc.get("bars", []) if spy_doc else []
    spy_date_close = {b["date"]: b["c"] for b in spy_bars}
    spy_sorted_closes = [b["c"] for b in spy_bars]

    ticker_bars = {}
    all_dates = set()

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
    open_positions = {}

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
                outcome = 1
            elif low <= pos["sl"]:
                outcome = 0
            elif days_held >= max_hold_days:
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

            # Regime da SPY alla data corrente
            regime_enc = 1
            if date in spy_date_close:
                spy_idx = next((i for i, b in enumerate(spy_bars) if b["date"] == date), None)
                if spy_idx is not None:
                    regime_enc = _regime_from_spy(spy_sorted_closes, spy_idx)

            features, conf, price = _build_features(
                bars_slice, ticker_sector.get(ticker, ""),
                sector_rank=6, regime_enc=regime_enc
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

    # Chiudi posizioni rimaste
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

    # Wipe vecchi e salva
    await db.ml_training_data.delete_many({"source": "backtest_collector"})
    if samples:
        await db.ml_training_data.insert_many(samples)

    wins = sum(1 for s in samples if s["label"] == 1)
    losses = len(samples) - wins

    return {
        "message": "Backtest training data collected (v2.1 - 15 features)",
        "total_samples": len(samples),
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / len(samples) * 100, 1) if samples else 0,
        "period": {"start": bt_dates[0], "end": bt_dates[-1]},
    }
