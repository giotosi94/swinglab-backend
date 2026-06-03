import httpx
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime
from app.db.mongodb import get_db
from app.config import settings
import traceback

TD_BASE = "https://api.twelvedata.com"

SECTOR_MAP = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}

SECTOR_STOCKS = {
    "XLK": ["AAPL", "MSFT", "NVDA", "AVGO", "AMD", "CRM", "ADBE", "INTC", "CSCO", "ORCL",
            "PLTR", "NOW", "SNOW", "CRWD", "PANW"],
    "XLF": ["JPM", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "AXP", "C", "USB",
            "V", "MA", "PYPL", "COF", "ICE"],
    "XLV": ["UNH", "JNJ", "PFE", "ABBV", "MRK", "TMO", "ABT", "LLY", "BMY", "AMGN",
            "ISRG", "DXCM", "VRTX", "REGN", "ZTS"],
    "XLI": ["CAT", "DE", "UNP", "HON", "BA", "RTX", "LMT", "GE", "MMM", "FDX",
            "UPS", "WM", "ETN", "ITW", "EMR"],
    "XLY": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "BKNG", "CMG",
            "LULU", "ROST", "DHI", "LEN", "ABNB"],
    "XLP": ["PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "CL", "MDLZ", "KHC",
            "STZ", "SYY", "HSY", "K", "GIS"],
    "XLE": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HAL",
            "DVN", "FANG", "PXD", "WMB", "KMI"],
    "XLU": ["NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "ED", "WEC",
            "AWK", "ES", "ATO", "CMS", "PNW"],
    "XLB": ["LIN", "APD", "SHW", "FCX", "NEM", "ECL", "DOW", "NUE", "VMC", "MLM",
            "CF", "MOS", "BALL", "PKG", "IFF"],
    "XLRE": ["PLD", "AMT", "CCI", "EQIX", "SPG", "PSA", "O", "WELL", "DLR", "AVB",
             "VICI", "MAA", "EXR", "ARE", "UDR"],
    "XLC": ["META", "GOOGL", "GOOG", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "EA",
            "SPOT", "RBLX", "TTWO", "WBD", "PARA"],
}


async def fetch_td(client, symbol):
    url = f"{TD_BASE}/time_series"
    params = {
        "symbol": symbol,
        "interval": "1day",
        "outputsize": 70,
        "apikey": settings.TWELVEDATA_API_KEY,
    }
    try:
        r = await client.get(url, params=params)
        if r.status_code != 200:
            print(f"  {symbol}: HTTP {r.status_code}")
            return None
        data = r.json()
        if "code" in data and data["code"] != 200:
            print(f"  {symbol}: API error - {data.get('message', 'unknown')}")
            return None
        values = data.get("values", [])
        if not values:
            print(f"  {symbol}: no values")
            return None
        df = pd.DataFrame(values)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
        df = df.dropna()
        return df
    except Exception as e:
        print(f"  {symbol} error: {e}")
        return None

# ============================================
# FAIR VALUE GAP (FVG) DETECTION
# ============================================

def detect_fvg(df):
    """Detect Fair Value Gaps - inefficiency zones"""
    if df is None or len(df) < 3:
        return []

    fvgs = []
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values

    for i in range(2, len(df)):
        # Bullish FVG: candle 3 low > candle 1 high (gap up)
        if low[i] > high[i-2]:
            fvg_top = float(low[i])
            fvg_bottom = float(high[i-2])
            fvg_size = fvg_top - fvg_bottom
            mid_price = (fvg_top + fvg_bottom) / 2
            # Check if FVG has been filled
            filled = False
            for j in range(i+1, len(df)):
                if low[j] <= fvg_bottom:
                    filled = True
                    break
            if not filled and fvg_size > 0:
                fvgs.append({
                    "type": "bullish",
                    "top": round(fvg_top, 2),
                    "bottom": round(fvg_bottom, 2),
                    "mid": round(mid_price, 2),
                    "size_pct": round((fvg_size / mid_price) * 100, 2),
                    "age_days": len(df) - i,
                    "filled": False,
                })

        # Bearish FVG: candle 3 high < candle 1 low (gap down)
        if high[i] < low[i-2]:
            fvg_top = float(low[i-2])
            fvg_bottom = float(high[i])
            fvg_size = fvg_top - fvg_bottom
            mid_price = (fvg_top + fvg_bottom) / 2
            filled = False
            for j in range(i+1, len(df)):
                if high[j] >= fvg_top:
                    filled = True
                    break
            if not filled and fvg_size > 0:
                fvgs.append({
                    "type": "bearish",
                    "top": round(fvg_top, 2),
                    "bottom": round(fvg_bottom, 2),
                    "mid": round(mid_price, 2),
                    "size_pct": round((fvg_size / mid_price) * 100, 2),
                    "age_days": len(df) - i,
                    "filled": False,
                })

    # Keep only recent unfilled FVGs (last 5)
    return sorted(fvgs, key=lambda x: x["age_days"])[:5]


# ============================================
# WYCKOFF PHASE DETECTION
# ============================================

def detect_wyckoff_phase(df):
    """Detect Wyckoff market phase from daily data"""
    if df is None or len(df) < 30:
        return {"phase": "unknown", "description": "Not enough data"}

    close = df["Close"].values
    volume = df["Volume"].values
    high = df["High"].values
    low = df["Low"].values

    n = len(df)
    current_price = float(close[-1])

    # Recent periods
    last_20_close = close[-20:]
    last_20_vol = volume[-20:]
    last_10_close = close[-10:]
    last_10_vol = volume[-10:]
    prev_20_close = close[-40:-20] if n >= 40 else close[:20]
    prev_20_vol = volume[-40:-20] if n >= 40 else volume[:20]

    # Calculations
    avg_vol_recent = float(np.mean(last_20_vol))
    avg_vol_prev = float(np.mean(prev_20_vol)) if len(prev_20_vol) > 0 else avg_vol_recent
    vol_change = (avg_vol_recent - avg_vol_prev) / avg_vol_prev * 100 if avg_vol_prev > 0 else 0

    price_range_20 = (float(max(last_20_close)) - float(min(last_20_close))) / float(np.mean(last_20_close)) * 100
    price_change_20 = (float(last_20_close[-1]) - float(last_20_close[0])) / float(last_20_close[0]) * 100
    price_change_10 = (float(last_10_close[-1]) - float(last_10_close[0])) / float(last_10_close[0]) * 100

    # Trend direction
    ema20 = float(pd.Series(close).ewm(span=20).mean().iloc[-1])
    ema50 = float(pd.Series(close).ewm(span=50).mean().iloc[-1])

    # Volume trend in last 10 days
    vol_trend_10 = (float(np.mean(last_10_vol)) - float(np.mean(last_20_vol[:10]))) / float(np.mean(last_20_vol[:10])) * 100 if float(np.mean(last_20_vol[:10])) > 0 else 0

    # Higher lows detection (accumulation sign)
    lows_5 = [float(min(low[max(0,i-5):i+1])) for i in range(n-20, n, 5)]
    higher_lows = all(lows_5[i] >= lows_5[i-1] for i in range(1, len(lows_5))) if len(lows_5) >= 2 else False

    # Lower highs detection (distribution sign)
    highs_5 = [float(max(high[max(0,i-5):i+1])) for i in range(n-20, n, 5)]
    lower_highs = all(highs_5[i] <= highs_5[i-1] for i in range(1, len(highs_5))) if len(highs_5) >= 2 else False

    # Phase detection
    phase = "unknown"
    confidence = 0
    description = ""
    signal = "neutral"

    # ACCUMULATION: price range tight, volume decreasing, near lows, higher lows forming
    if price_range_20 < 10 and current_price < ema50 and higher_lows:
        phase = "accumulation"
        confidence = 70
        if vol_change < -10:
            confidence += 15
        if price_change_10 > 0:
            confidence += 10
        description = "Price consolidating near lows with higher lows forming. Smart money may be accumulating. Volume is {}. Watch for breakout above range.".format("decreasing (bullish)" if vol_change < 0 else "increasing")
        signal = "bullish_soon"

    # MARKUP: price rising, above EMAs, volume confirming
    elif price_change_20 > 5 and current_price > ema20 > ema50:
        phase = "markup"
        confidence = 75
        if vol_change > 10:
            confidence += 10
        if price_change_10 > 2:
            confidence += 10
        description = "Strong uptrend. Price above EMA20 and EMA50. Trend is confirmed by {}. Look for pullbacks to EMA20 for entries.".format("increasing volume" if vol_change > 0 else "momentum")
        signal = "bullish"

    # DISTRIBUTION: price range tight near highs, volume increasing, lower highs
    elif price_range_20 < 10 and current_price > ema50 and lower_highs:
        phase = "distribution"
        confidence = 65
        if vol_change > 15:
            confidence += 15
        description = "Price consolidating near highs with lower highs forming. Smart money may be distributing. Watch for breakdown below range."
        signal = "bearish_soon"

    # MARKDOWN: price falling, below EMAs
    elif price_change_20 < -5 and current_price < ema20 and current_price < ema50:
        phase = "markdown"
        confidence = 75
        if vol_change > 10:
            confidence += 10
        description = "Downtrend. Price below EMA20 and EMA50. Avoid buying until accumulation phase begins. Watch for selling climax (volume spike + strong reversal)."
        signal = "bearish"

    # SPRING (Wyckoff): false breakdown then recovery (accumulation phase C)
    elif n >= 10:
        recent_low = float(min(low[-10:]))
        prev_low = float(min(low[-30:-10])) if n >= 30 else float(min(low[:20]))
        if recent_low < prev_low and price_change_10 > 3:
            phase = "spring"
            confidence = 60
            if vol_change > 20:
                confidence += 15
            description = "Potential Wyckoff Spring! Price broke below support then recovered strongly. This is often a bear trap before a significant move up."
            signal = "strong_bullish"

    # SELLING CLIMAX: huge volume + big drop
    elif n >= 5:
        last_5_vol_avg = float(np.mean(volume[-5:]))
        overall_vol_avg = float(np.mean(volume[-30:])) if n >= 30 else float(np.mean(volume))
        vol_spike = last_5_vol_avg / overall_vol_avg if overall_vol_avg > 0 else 1
        if vol_spike > 2 and price_change_10 < -8:
            phase = "selling_climax"
            confidence = 55
            description = "Potential Selling Climax! Extreme volume with sharp price drop. This often marks the end of a downtrend. Watch for reversal patterns."
            signal = "reversal_possible"

    else:
        phase = "transition"
        confidence = 30
        description = "Market is in transition. No clear Wyckoff phase detected. Wait for clearer signals."
        signal = "neutral"

    confidence = min(confidence, 95)

    return {
        "phase": phase,
        "confidence": confidence,
        "description": description,
        "signal": signal,
        "metrics": {
            "price_change_20d": round(price_change_20, 2),
            "price_change_10d": round(price_change_10, 2),
            "price_range_20d": round(price_range_20, 2),
            "vol_change_pct": round(vol_change, 2),
            "higher_lows": higher_lows,
            "lower_highs": lower_highs,
            "above_ema20": current_price > ema20,
            "above_ema50": current_price > ema50,
        }
    }


# ============================================
# ACCUMULATION SCORE
# ============================================

def calc_accumulation_score(df, poc_price, va_low, va_high):
    """Score how likely smart money is accumulating"""
    if df is None or len(df) < 20:
        return {"score": 0, "level": "unknown", "factors": []}

    close = df["Close"].values
    volume = df["Volume"].values
    low = df["Low"].values

    current_price = float(close[-1])
    factors = []
    score = 0

    # 1. Price below POC (+25) - buying below fair value
    if poc_price and current_price < poc_price:
        dist = abs(current_price - poc_price) / poc_price * 100
        if dist <= 5:
            score += 25
            factors.append({"name": "Below POC", "score": 25, "detail": "Price {:.1f}% below POC - buying zone".format(dist), "pass": True})
        elif dist <= 15:
            score += 15
            factors.append({"name": "Below POC", "score": 15, "detail": "Price {:.1f}% below POC".format(dist), "pass": True})
        else:
            factors.append({"name": "Below POC", "score": 0, "detail": "Price {:.1f}% below POC - too far".format(dist), "pass": False})
    else:
        factors.append({"name": "Below POC", "score": 0, "detail": "Price above POC", "pass": False})

    # 2. Near or below VA Low (+20) - deep value
    if va_low and current_price <= va_low * 1.02:
        score += 20
        factors.append({"name": "Near VA Low", "score": 20, "detail": "Price near/below Value Area Low", "pass": True})
    else:
        factors.append({"name": "Near VA Low", "score": 0, "detail": "Price above VA Low", "pass": False})

    # 3. Volume pattern: decreasing vol = accumulation (+15)
    if len(volume) >= 20:
        vol_first_half = float(np.mean(volume[-20:-10]))
        vol_second_half = float(np.mean(volume[-10:]))
        if vol_first_half > 0:
            vol_decrease = (vol_second_half - vol_first_half) / vol_first_half * 100
            if vol_decrease < -15:
                score += 15
                factors.append({"name": "Volume Decreasing", "score": 15, "detail": "Volume down {:.0f}% - typical accumulation".format(vol_decrease), "pass": True})
            else:
                factors.append({"name": "Volume Decreasing", "score": 0, "detail": "Volume change {:.0f}%".format(vol_decrease), "pass": False})

    # 4. Higher lows in last 20 days (+20)
    lows_weekly = []
    for i in range(0, min(20, len(low)), 5):
        end = min(i + 5, len(low))
        if end > i:
            lows_weekly.append(float(min(low[-end:][:5] if end <= len(low) else low[-5:])))
    if len(lows_weekly) >= 3:
        hl = all(lows_weekly[j] >= lows_weekly[j-1] * 0.99 for j in range(1, len(lows_weekly)))
        if hl:
            score += 20
            factors.append({"name": "Higher Lows", "score": 20, "detail": "Forming higher lows - accumulation pattern", "pass": True})
        else:
            factors.append({"name": "Higher Lows", "score": 0, "detail": "No higher lows pattern", "pass": False})

    # 5. Tight range (consolidation) (+10)
    if len(close) >= 10:
        range_pct = (float(max(close[-10:])) - float(min(close[-10:]))) / float(np.mean(close[-10:])) * 100
        if range_pct < 8:
            score += 10
            factors.append({"name": "Tight Range", "score": 10, "detail": "Price range {:.1f}% - consolidation".format(range_pct), "pass": True})
        else:
            factors.append({"name": "Tight Range", "score": 0, "detail": "Price range {:.1f}%".format(range_pct), "pass": False})

    # 6. POC recovery direction (+10)
    if poc_price and len(close) >= 5:
        approaching = float(close[-1]) > float(close[-5]) and current_price < poc_price
        if approaching:
            score += 10
            factors.append({"name": "Approaching POC", "score": 10, "detail": "Price moving toward POC - recovery", "pass": True})
        else:
            factors.append({"name": "Approaching POC", "score": 0, "detail": "Not approaching POC", "pass": False})

    score = min(score, 100)
    level = "strong" if score >= 70 else "moderate" if score >= 40 else "weak" if score >= 20 else "none"

    return {"score": score, "level": level, "factors": factors}

# ============================================
# CANDLESTICK PATTERN DETECTION
# ============================================

def detect_candlestick_patterns(df):
    """Detect candlestick patterns from last 3 candles"""
    if df is None or len(df) < 3:
        return []

    patterns = []
    o = df["Open"].values
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values

    # Last 3 candles
    i = len(df) - 1  # today
    i1 = i - 1       # yesterday
    i2 = i - 2       # 2 days ago

    body = abs(c[i] - o[i])
    range_total = h[i] - l[i]
    upper_shadow = h[i] - max(o[i], c[i])
    lower_shadow = min(o[i], c[i]) - l[i]
    is_bullish = c[i] > o[i]
    is_bearish = c[i] < o[i]

    body1 = abs(c[i1] - o[i1])
    is_bullish1 = c[i1] > o[i1]
    is_bearish1 = c[i1] < o[i1]

    body2 = abs(c[i2] - o[i2])
    is_bullish2 = c[i2] > o[i2]

    if range_total == 0:
        return []

    body_pct = body / range_total

    # 1. HAMMER (bullish reversal)
    # Small body at top, long lower shadow (2x body), small upper shadow
    if body_pct < 0.35 and lower_shadow >= body * 2 and upper_shadow < body * 0.5:
        patterns.append({
            "name": "Hammer",
            "type": "bullish",
            "strength": "strong",
            "description": "Bullish reversal - buyers pushed price up from lows"
        })

    # 2. INVERTED HAMMER (bullish reversal)
    # Small body at bottom, long upper shadow, small lower shadow
    if body_pct < 0.35 and upper_shadow >= body * 2 and lower_shadow < body * 0.5:
        patterns.append({
            "name": "Inverted Hammer",
            "type": "bullish",
            "strength": "moderate",
            "description": "Potential bullish reversal - needs confirmation"
        })

    # 3. BULLISH ENGULFING
    # Previous bearish, current bullish, current body engulfs previous body
    if is_bearish1 and is_bullish and c[i] > o[i1] and o[i] < c[i1] and body > body1:
        patterns.append({
            "name": "Bullish Engulfing",
            "type": "bullish",
            "strength": "strong",
            "description": "Strong bullish reversal - buyers overwhelmed sellers"
        })

    # 4. BEARISH ENGULFING
    # Previous bullish, current bearish, current body engulfs previous body
    if is_bullish1 and is_bearish and o[i] > c[i1] and c[i] < o[i1] and body > body1:
        patterns.append({
            "name": "Bearish Engulfing",
            "type": "bearish",
            "strength": "strong",
            "description": "Strong bearish reversal - sellers overwhelmed buyers"
        })

    # 5. DOJI (indecision)
    # Very small body relative to range
    if body_pct < 0.1 and range_total > 0:
        patterns.append({
            "name": "Doji",
            "type": "neutral",
            "strength": "moderate",
            "description": "Market indecision - watch for breakout direction"
        })

    # 6. MORNING STAR (bullish reversal - 3 candle pattern)
    # Day 1: big bearish, Day 2: small body (gap down), Day 3: big bullish closes above day1 midpoint
    day1_mid = (o[i2] + c[i2]) / 2
    if is_bearish1 is False and body2 > 0 and (c[i2] < o[i2]) and body_pct < 0.3 is False:
        pass
    if len(df) >= 3:
        if c[i2] < o[i2] and body2 > range_total * 0.15:  # day1 bearish with decent body
            if body1 < body2 * 0.4:  # day2 small body
                if is_bullish and c[i] > day1_mid:  # day3 bullish, closes above day1 midpoint
                    patterns.append({
                        "name": "Morning Star",
                        "type": "bullish",
                        "strength": "strong",
                        "description": "3-candle bullish reversal - high reliability pattern"
                    })

    # 7. EVENING STAR (bearish reversal - 3 candle pattern)
    day1_mid2 = (o[i2] + c[i2]) / 2
    if len(df) >= 3:
        if c[i2] > o[i2] and body2 > range_total * 0.15:  # day1 bullish
            if body1 < body2 * 0.4:  # day2 small body
                if is_bearish and c[i] < day1_mid2:  # day3 bearish
                    patterns.append({
                        "name": "Evening Star",
                        "type": "bearish",
                        "strength": "strong",
                        "description": "3-candle bearish reversal - high reliability pattern"
                    })

    # 8. THREE WHITE SOLDIERS (bullish continuation)
    if len(df) >= 3:
        if (c[i2] > o[i2]) and (c[i1] > o[i1]) and (c[i] > o[i]):
            if c[i1] > c[i2] and c[i] > c[i1]:
                if body1 > range_total * 0.15 and body2 > range_total * 0.15 and body > range_total * 0.15:
                    patterns.append({
                        "name": "Three White Soldiers",
                        "type": "bullish",
                        "strength": "strong",
                        "description": "3 consecutive bullish candles - strong buying pressure"
                    })

    # 9. THREE BLACK CROWS (bearish continuation)
    if len(df) >= 3:
        if (c[i2] < o[i2]) and (c[i1] < o[i1]) and (c[i] < o[i]):
            if c[i1] < c[i2] and c[i] < c[i1]:
                if body1 > range_total * 0.15 and body2 > range_total * 0.15 and body > range_total * 0.15:
                    patterns.append({
                        "name": "Three Black Crows",
                        "type": "bearish",
                        "strength": "strong",
                        "description": "3 consecutive bearish candles - strong selling pressure"
                    })

    # 10. SHOOTING STAR (bearish reversal at top)
    if body_pct < 0.3 and upper_shadow >= body * 2 and lower_shadow < body * 0.3 and is_bearish:
        patterns.append({
            "name": "Shooting Star",
            "type": "bearish",
            "strength": "moderate",
            "description": "Bearish reversal at resistance - sellers rejected higher prices"
        })

    return patterns


def get_pattern_score_bonus(patterns):
    """Calculate bonus/penalty points from candlestick patterns"""
    bonus = 0
    for p in patterns:
        if p["type"] == "bullish" and p["strength"] == "strong":
            bonus += 8
        elif p["type"] == "bullish" and p["strength"] == "moderate":
            bonus += 4
        elif p["type"] == "bearish" and p["strength"] == "strong":
            bonus -= 6
        elif p["type"] == "bearish" and p["strength"] == "moderate":
            bonus -= 3
    return bonus


# ============================================
# INDICATORI TECNICI
# ============================================

def calc_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1] if not rsi.empty else 50
    return 50 if pd.isna(val) else val


def calc_ema(prices, period):
    ema = prices.ewm(span=period, adjust=False).mean()
    val = ema.iloc[-1] if not ema.empty else 0
    return 0 if pd.isna(val) else val


def calc_macd(prices):
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal
    return {
        "macd": round(float(macd_line.iloc[-1]), 4),
        "signal": round(float(signal.iloc[-1]), 4),
        "histogram": round(float(histogram.iloc[-1]), 4),
    }


def calc_volume_profile(highs, lows, volumes, bins=30):
    try:
        price_min = float(lows.min())
        price_max = float(highs.max())
        if price_max <= price_min:
            return None, None, None, []
        bin_edges = np.linspace(price_min, price_max, bins + 1)
        volume_per_level = np.zeros(bins)
        for idx in range(len(highs)):
            row_low = float(lows.iloc[idx])
            row_high = float(highs.iloc[idx])
            row_vol = float(volumes.iloc[idx])
            spread = max(1, int((row_high - row_low) / ((price_max - price_min) / bins)))
            for i in range(bins):
                if row_low <= bin_edges[i + 1] and row_high >= bin_edges[i]:
                    volume_per_level[i] += row_vol / spread
        poc_idx = int(np.argmax(volume_per_level))
        poc_price = round((bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2, 2)
        total_vol = volume_per_level.sum()
        target_vol = total_vol * 0.70
        sorted_idx = np.argsort(volume_per_level)[::-1]
        cumulative = 0
        va_indices = []
        for idx in sorted_idx:
            cumulative += volume_per_level[idx]
            va_indices.append(idx)
            if cumulative >= target_vol:
                break
        va_low = round(float(bin_edges[min(va_indices)]), 2)
        va_high = round(float(bin_edges[max(va_indices) + 1]), 2)

        # Build distribution for frontend chart
        max_vol = float(volume_per_level.max()) if volume_per_level.max() > 0 else 1
        distribution = []
        for i in range(bins):
            price_level = round((bin_edges[i] + bin_edges[i + 1]) / 2, 2)
            vol_pct = round(float(volume_per_level[i] / max_vol * 100), 1)
            is_poc = i == poc_idx
            in_va = i in va_indices
            distribution.append({
                "price": price_level,
                "volume_pct": vol_pct,
                "is_poc": is_poc,
                "in_value_area": in_va,
            })
        return poc_price, va_high, va_low, distribution
    except Exception:
        return None, None, None, []


def calc_multi_tf_vp(df):
    """Calculate Volume Profile for multiple timeframes"""
    results = {}
    for label, days in [("short", 20), ("medium", 50), ("full", len(df))]:
        subset = df.tail(days)
        if len(subset) < 10:
            continue
        poc, va_h, va_l, dist = calc_volume_profile(
            subset["High"], subset["Low"], subset["Volume"], bins=30
        )
        if poc:
            results[label] = {
                "period_days": days,
                "poc": poc,
                "va_high": va_h,
                "va_low": va_l,
                "distribution": dist,
            }
    return results


def calc_setup_score(data):
    score = 0
    rsi = data.get("rsi", 50)
    if 40 <= rsi <= 60:
        score += 15
    elif 30 <= rsi <= 70:
        score += 10
    else:
        score += 5
    hist = data.get("macd_histogram", 0)
    if hist > 0:
        score += 15
    elif hist > -0.5:
        score += 8
    price = data.get("price", 0)
    ema10 = data.get("ema10", 0)
    ema20 = data.get("ema20", 0)
    ema50 = data.get("ema50", 0)
    if price > ema10 > ema20 > ema50:
        score += 15
    elif price > ema20 > ema50:
        score += 10
    elif price > ema50:
        score += 5
    rel_vol = data.get("relative_volume", 1)
    if rel_vol >= 2.0:
        score += 15
    elif rel_vol >= 1.5:
        score += 12
    elif rel_vol >= 1.0:
        score += 8
    poc = data.get("poc_price", 0)
    if poc and price:
        dist = abs(price - poc) / price * 100
        if dist <= 2:
            score += 15
        elif dist <= 5:
            score += 10
        elif dist <= 10:
            score += 5
    sector_score = data.get("sector_strength", 50)
    score += int(sector_score / 100 * 15)
    change = data.get("change_pct", 0)
    if 0.5 <= change <= 5:
        score += 10
    elif 0 < change <= 0.5:
        score += 6
    elif change > 5:
        score += 4

    # Candlestick pattern bonus
    pattern_bonus = data.get("pattern_bonus", 0)
    score += pattern_bonus

    return max(0, min(score, 100))


def detect_setup_type(data):
    price = data.get("price", 0)
    poc = data.get("poc_price", 0)
    va_high = data.get("va_high", 0)
    ema20 = data.get("ema20", 0)
    rsi = data.get("rsi", 50)
    rel_vol = data.get("relative_volume", 1)
    if va_high and price > va_high and rel_vol >= 1.5:
        return "breakout"
    if poc and price and abs(price - poc) / price * 100 <= 2:
        return "pullback_to_poc"
    if ema20 and price and abs(price - ema20) / price * 100 <= 1.5:
        return "ema_bounce"
    if rsi <= 30:
        return "oversold_reversal"
    if rsi >= 70:
        return "overbought_warning"
    return "neutral"


# ============================================
# FETCH & ANALYZE SECTORS
# ============================================

async def fetch_and_analyze_sectors():
    db = get_db()
    if not settings.TWELVEDATA_API_KEY:
        print("ERROR: TWELVEDATA_API_KEY not set!")
        return []
    print("=" * 50)
    print("STARTING SECTOR REFRESH (Twelve Data)")
    print("=" * 50)
    async with httpx.AsyncClient(timeout=60) as client:
        print("Fetching SPY...")
        spy_df = await fetch_td(client, "SPY")
        spy_return = 0
        if spy_df is not None and len(spy_df) >= 20:
            spy_return = ((float(spy_df["Close"].iloc[-1]) / float(spy_df["Close"].iloc[-20])) - 1) * 100
            print(f"SPY 20d return: {spy_return:.2f}%")
        else:
            print("WARNING: SPY not available")
        results = []
        # Save SPY data for market regime
        if spy_df is not None and len(spy_df) >= 50:
            spy_close = spy_df["Close"]
            spy_ema20 = float(spy_close.ewm(span=20, adjust=False).mean().iloc[-1])
            spy_ema50 = float(spy_close.ewm(span=50, adjust=False).mean().iloc[-1])
            spy_ema200 = float(spy_close.ewm(span=50, adjust=False).mean().iloc[-1])  # approx with 50 data
            spy_rsi_delta = spy_close.diff()
            spy_gain = spy_rsi_delta.where(spy_rsi_delta > 0, 0).rolling(14).mean()
            spy_loss = (-spy_rsi_delta.where(spy_rsi_delta < 0, 0)).rolling(14).mean()
            spy_rs = spy_gain / spy_loss
            spy_rsi_val = float((100 - (100 / (1 + spy_rs))).iloc[-1])
            spy_price = float(spy_close.iloc[-1])
            spy_change = float(((spy_close.iloc[-1] / spy_close.iloc[-2]) - 1) * 100)

            spy_doc = {
                "symbol": "SPY",
                "price": round(spy_price, 2),
                "change_pct": round(spy_change, 2),
                "ema20": round(spy_ema20, 2),
                "ema50": round(spy_ema50, 2),
                "rsi": round(spy_rsi_val, 1),
                "return_20d": round(spy_return, 2),
                "updated_at": datetime.utcnow(),
            }
            await db.market_regime.update_one({"symbol": "SPY"}, {"$set": spy_doc}, upsert=True)
            print(f"SPY regime saved: ${spy_price:.2f}, RSI {spy_rsi_val:.1f}")
        for etf, name in SECTOR_MAP.items():
            try:
                await asyncio.sleep(8)
                print(f"  Fetching {etf} ({name})...")
                df = await fetch_td(client, etf)
                if df is None or len(df) < 20:
                    print(f"  SKIP {etf}")
                    continue
                close = df["Close"]
                volume = df["Volume"]
                ret_20d = ((float(close.iloc[-1]) / float(close.iloc[-20])) - 1) * 100
                strength = round(float(ret_20d - spy_return), 2)
                rsi = round(float(calc_rsi(close)), 2)
                ema10 = float(calc_ema(close, 10))
                ema20_val = float(calc_ema(close, 20))
                ema50 = float(calc_ema(close, 50))
                price = float(close.iloc[-1])
                avg_vol = float(volume.rolling(20).mean().iloc[-1])
                curr_vol = float(volume.iloc[-1])
                rel_vol = round(curr_vol / avg_vol, 2) if avg_vol > 0 else 1
                trend = 0
                if price > ema10 > ema20_val > ema50:
                    trend = 90
                elif price > ema20_val > ema50:
                    trend = 70
                elif price > ema50:
                    trend = 50
                else:
                    trend = 30
                composite = round((strength * 2 + trend + rsi) / 4, 2)
                # Build daily history with indicators
                history = []
                rsi_series = pd.Series(dtype=float)
                delta = close.diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi_series = 100 - (100 / (1 + rs))

                ema10_series = close.ewm(span=10, adjust=False).mean()
                ema20_series = close.ewm(span=20, adjust=False).mean()
                ema50_series = close.ewm(span=50, adjust=False).mean()

                for idx in range(20, len(df)):
                    day_close = float(close.iloc[idx])
                    day_rsi = float(rsi_series.iloc[idx]) if not pd.isna(rsi_series.iloc[idx]) else 50
                    day_ema10 = float(ema10_series.iloc[idx])
                    day_ema20 = float(ema20_series.iloc[idx])
                    day_ema50 = float(ema50_series.iloc[idx])
                    day_vol = float(volume.iloc[idx])
                    avg_vol_20 = float(volume.iloc[max(0,idx-20):idx].mean())
                    day_rvol = round(day_vol / avg_vol_20, 2) if avg_vol_20 > 0 else 1

                    # Daily health
                    day_trend = 90 if day_close > day_ema10 > day_ema20 > day_ema50 else (70 if day_close > day_ema20 > day_ema50 else (50 if day_close > day_ema50 else 30))

                    # Zone
                    if day_rsi <= 30:
                        zone = "oversold"
                    elif day_rsi <= 40:
                        zone = "weak"
                    elif day_rsi >= 70:
                        zone = "overbought"
                    elif day_rsi >= 60:
                        zone = "strong"
                    else:
                        zone = "neutral"

                    history.append({
                        "date": df.index[idx].strftime("%Y-%m-%d") if hasattr(df.index[idx], 'strftime') else str(df["datetime"].iloc[idx])[:10] if "datetime" in df.columns else f"day_{idx}",
                        "close": round(day_close, 2),
                        "rsi": round(day_rsi, 1),
                        "ema10": round(day_ema10, 2),
                        "ema20": round(day_ema20, 2),
                        "ema50": round(day_ema50, 2),
                        "rvol": day_rvol,
                        "trend": day_trend,
                        "zone": zone,
                    })

                sector_doc = {
                    "code": etf, "name": name, "etf_ticker": etf,
                    "price": round(price, 2),
                    "return_20d": round(float(ret_20d), 2),
                    "strength_score": strength, "trend_score": trend,
                    "volume_score": round(rel_vol * 30, 2),
                    "rsi": rsi, "composite_score": composite,
                    "history": history,
                    "updated_at": datetime.utcnow(),
                }
                await db.sectors.update_one({"code": etf}, {"$set": sector_doc}, upsert=True)
                results.append(sector_doc)
                print(f"  OK {etf}: ${price:.2f} score={composite:.2f}")
            except Exception as e:
                print(f"  ERROR {etf}: {e}")
                traceback.print_exc()
    print(f"\nSECTORS DONE: {len(results)}/11")
    return results


# ============================================
# FETCH & ANALYZE STOCKS
# ============================================

async def fetch_and_analyze_stocks():
    db = get_db()
    if not settings.TWELVEDATA_API_KEY:
        print("ERROR: TWELVEDATA_API_KEY not set!")
        return []
    print("=" * 50)
    print("STARTING STOCKS REFRESH (Twelve Data + Candlestick)")
    print("=" * 50)
    sector_scores = {}
    async for s in db.sectors.find():
        sector_scores[s["code"]] = s.get("composite_score", 50)
    print(f"Sector scores: {sector_scores}")
    results = []
    async with httpx.AsyncClient(timeout=60) as client:
        for sector_code, tickers in SECTOR_STOCKS.items():
            print(f"\n--- {sector_code} ---")
            for ticker in tickers:
                try:
                    await asyncio.sleep(8)
                    df = await fetch_td(client, ticker)
                    if df is None or len(df) < 20:
                        print(f"    SKIP {ticker}")
                        continue
                    close = df["Close"]
                    volume = df["Volume"]
                    high = df["High"]
                    low = df["Low"]
                    price = float(close.iloc[-1])
                    prev_close = float(close.iloc[-2])
                    change_pct = round(((price - prev_close) / prev_close) * 100, 2)
                    avg_vol = float(volume.rolling(20).mean().iloc[-1])
                    curr_vol = float(volume.iloc[-1])
                    rel_vol = round(curr_vol / avg_vol, 2) if avg_vol > 0 else 1
                    rsi = round(float(calc_rsi(close)), 2)
                    macd = calc_macd(close)
                    ema10 = round(float(calc_ema(close, 10)), 2)
                    ema20 = round(float(calc_ema(close, 20)), 2)
                    ema50 = round(float(calc_ema(close, 50)), 2)
                    poc_result = calc_volume_profile(high, low, volume)
                    # 52-week high/low (from available data)
                    high_52w = round(float(high.max()), 2)
                    low_52w = round(float(low.min()), 2)
                    pct_from_high = round(((price - high_52w) / high_52w) * 100, 2) if high_52w > 0 else 0
                    pct_from_low = round(((price - low_52w) / low_52w) * 100, 2) if low_52w > 0 else 0
                    range_position = round(((price - low_52w) / (high_52w - low_52w)) * 100, 1) if (high_52w - low_52w) > 0 else 50
                    poc = poc_result[0]
                    va_high = poc_result[1]
                    va_low = poc_result[2]
                    vp_distribution = poc_result[3]

                    # Multi-timeframe VP
                    multi_tf_vp = calc_multi_tf_vp(df)

                    # Candlestick patterns
                    patterns = detect_candlestick_patterns(df)
                    # Advanced analysis
                    fvgs = detect_fvg(df)
                    wyckoff = detect_wyckoff_phase(df)
                    accumulation = calc_accumulation_score(df, poc, va_low, va_high)
                    pattern_bonus = get_pattern_score_bonus(patterns)
                    patterns_list = [{"name": p["name"], "type": p["type"], "strength": p["strength"], "description": p["description"]} for p in patterns]

                    ind_data = {
                        "price": price, "rsi": rsi, "macd_histogram": macd["histogram"],
                        "ema10": ema10, "ema20": ema20, "ema50": ema50,
                        "relative_volume": rel_vol, "poc_price": poc, "va_high": va_high,
                        "change_pct": change_pct,
                        "sector_strength": sector_scores.get(sector_code, 50),
                        "pattern_bonus": pattern_bonus,
                    }
                    setup_score = calc_setup_score(ind_data)
                    setup_type = detect_setup_type(ind_data)
                    asset_doc = {
                        "ticker": ticker, "name": ticker, "sector_code": sector_code,
                        "price": round(price, 2), "change_pct": change_pct,
                        "avg_volume": round(avg_vol, 0), "relative_volume": rel_vol,
                        "rsi": rsi, "macd": macd,
                        "ema10": ema10, "ema20": ema20, "ema50": ema50,
                        "momentum_score": rsi, "volume_score": round(rel_vol * 30, 2),
                        "poc_price": poc, "value_area_high": va_high, "value_area_low": va_low,
                        "setup_score": setup_score, "setup_type": setup_type,
                        "vp_distribution": vp_distribution,
                        "multi_tf_vp": multi_tf_vp,
                        "candlestick_patterns": patterns_list,
                        "fvg": fvgs,
                        "wyckoff": wyckoff,
                        "accumulation": accumulation,
                        "pattern_bonus": pattern_bonus,
                        "high_52w": high_52w,
                        "low_52w": low_52w,
                        "pct_from_high": pct_from_high,
                        "pct_from_low": pct_from_low,
                        "range_position": range_position,
                        "updated_at": datetime.utcnow(),
                    }
                    await db.assets.update_one({"ticker": ticker}, {"$set": asset_doc}, upsert=True)
                    results.append(asset_doc)
                    pat_str = ", ".join([p["name"] for p in patterns]) if patterns else "none"
                    print(f"    OK {ticker}: ${price:.2f} score={setup_score} [{setup_type}] patterns=[{pat_str}]")
                except Exception as e:
                    print(f"    ERROR {ticker}: {e}")
    print(f"\nSTOCKS DONE: {len(results)}/110")
    return results
