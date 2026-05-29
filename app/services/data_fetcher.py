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
    "XLK": ["AAPL", "MSFT", "NVDA", "AVGO", "AMD", "CRM", "ADBE", "INTC", "CSCO", "ORCL"],
    "XLF": ["JPM", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "AXP", "C", "USB"],
    "XLV": ["UNH", "JNJ", "PFE", "ABBV", "MRK", "TMO", "ABT", "LLY", "BMY", "AMGN"],
    "XLI": ["CAT", "DE", "UNP", "HON", "BA", "RTX", "LMT", "GE", "MMM", "FDX"],
    "XLY": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "BKNG", "CMG"],
    "XLP": ["PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "CL", "MDLZ", "KHC"],
    "XLE": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HAL"],
    "XLU": ["NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "ED", "WEC"],
    "XLB": ["LIN", "APD", "SHW", "FCX", "NEM", "ECL", "DOW", "NUE", "VMC", "MLM"],
    "XLRE": ["PLD", "AMT", "CCI", "EQIX", "SPG", "PSA", "O", "WELL", "DLR", "AVB"],
    "XLC": ["META", "GOOGL", "GOOG", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "EA"],
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
                sector_doc = {
                    "code": etf, "name": name, "etf_ticker": etf,
                    "price": round(price, 2),
                    "return_20d": round(float(ret_20d), 2),
                    "strength_score": strength, "trend_score": trend,
                    "volume_score": round(rel_vol * 30, 2),
                    "rsi": rsi, "composite_score": composite,
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
                    poc = poc_result[0]
                    va_high = poc_result[1]
                    va_low = poc_result[2]
                    vp_distribution = poc_result[3]

                    # Multi-timeframe VP
                    multi_tf_vp = calc_multi_tf_vp(df)

                    # Candlestick patterns
                    patterns = detect_candlestick_patterns(df)
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
                        "pattern_bonus": pattern_bonus,
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
