"""
Backtest Data Collector v2.0 — genera training data ML da dati storici.
v2.0: calcola TUTTE le 15 features (incluse regime, sector, wyckoff, accumulation, patterns).
"""

import numpy as np
from datetime import datetime
from app.db.mongodb import get_db
from app.ml.features import features_to_array, SETUP_ENCODE, REGIME_ENCODE, WYCKOFF_ENCODE


def _rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50.0
    d = np.diff(prices)
    g = np.where(d > 0, d, 0)
    l = np.where(d < 0, -d, 0)
    ag = np.mean(g[-period:])🎯 **Vai Giovanni! Miglioro il collector con le 5 features reali. Questo porta da 10 → 15 features attive.**

## 📄 Sostituisci `_build_features` in `backtest_collector.py`

Vai su GitHub → `app/ml/backtest_collector.py` → matita.

### 🔍 MODIFICA 1 — Aggiungi funzioni helper

**Ctrl+F cerca**: `def _build_features(bars_slice, sector_code):`

**IMMEDIATAMENTE PRIMA** di questa riga, incolla le nuove funzioni:

```python
def _detect_wyckoff(closes, volumes, highs, lows):
    """Wyckoff phase semplificato da price/volume action."""
    if len(closes) < 30:
        return 5  # neutro
    price = closes[-1]
    ema20 = _ema(closes, 20)
    ema50 = _ema(closes, 50)
    pc_20 = ((price / closes[-20]) - 1) * 100
    rng_20 = ((max(closes[-20:]) - min(closes[-20:])) / closes[-20]) * 100

    vol_old = np.mean(volumes[-40:-20]) if len(volumes) >= 40 else np.mean(volumes[:20])
    vol_new = np.mean(volumes[-20:])
    vol_chg = ((vol_new - vol_old) / vol_old * 100) if vol_old > 0 else 0

    # accumulation: 0 | markup: 1 | spring: 2 | distribution: 3 | markdown: 4
    if rng_20 < 10 and vol_chg < -10:
        return 0  # accumulation
    elif pc_20 > 5 and price > ema20 > ema50:
        return 1  # markup
    elif rng_20 < 10 and price > ema50:
        return 3  # distribution
    elif pc_20 < -5 and price < ema20:
        return 4  # markdown
    # spring: recent low breaks then recovers
    if len(lows) >= 10:
        recent_low = min(lows[-10:])
        prev_low = min(lows[-30:-10]) if len(lows) >= 30 else min(lows[:20])
        pc_10 = ((price / closes[-10]) - 1) * 100
        if recent_low < prev_low and pc_10 > 3:
            return 2  # spring
    return 5


def _accumulation_score(closes, volumes, highs, lows):
    """Accumulation score 0-100 da volume + price."""
    if len(closes) < 20:
        return 0
    price = closes[-1]
    poc_proxy = np.mean(closes[-20:])
    score = 0
    # below POC proxy
    if price < poc_proxy:
        dist = abs(price - poc_proxy) / poc_proxy * 100
        if dist <= 5:
            score += 25
        elif dist <= 15:
            score += 15
    # near low of range
    low_20 = min(lows[-20:])
    if price <= low_20 * 1.02:
        score += 20
    # volume decreasing (accumulation)
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
    # Hammer: small body, long lower shadow
    if body_pct < 0.35 and lower_shadow >= body * 2 and upper_shadow < body * 0.5:
        return 1
    # Bullish engulfing
    if len(closes) >= 2:
        o1, c1 = opens[-2], closes[-2]
        if c > o and c1 < o1 and c > o1 and o < c1:
            return 1
    return 0


def _regime_from_spy(spy_closes, date_idx):
    """Regime encoded da SPY trend alla data. 0=BULL 1=NEUTRAL 2=BEAR 3=CRASH."""
    if not spy_closes or date_idx < 50:
        return 1  # neutral
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
