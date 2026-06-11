import httpx
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta
from app.db.mongodb import get_db
from app.config import settings
import traceback
import time

SECTOR_MAP = {
    "XLK": "Technology", "XLF": "Financials", "XLV": "Health Care",
    "XLI": "Industrials", "XLY": "Consumer Discretionary", "XLP": "Consumer Staples",
    "XLE": "Energy", "XLU": "Utilities", "XLB": "Materials",
    "XLRE": "Real Estate", "XLC": "Communication Services",
}

SECTOR_STOCKS = {
    "XLK": ["AAPL","MSFT","NVDA","AVGO","AMD","CRM","ADBE","INTC","CSCO","ORCL",
            "PLTR","NOW","SNOW","CRWD","PANW","MNDY","SHOP","SQ","UBER","DDOG"],
    "XLF": ["JPM","BAC","WFC","GS","MS","BLK","SCHW","AXP","C","USB",
            "V","MA","PYPL","COF","ICE","SPGI","MCO","MMC","AON","TFC"],
    "XLV": ["UNH","JNJ","PFE","ABBV","MRK","TMO","ABT","LLY","BMY","AMGN",
            "ISRG","DXCM","VRTX","REGN","ZTS","HCA","CI","ELV","HUM","SYK"],
    "XLI": ["CAT","DE","UNP","HON","BA","RTX","LMT","GE","MMM","FDX",
            "UPS","WM","ETN","ITW","EMR","NSC","CSX","PCAR","ROK","IR"],
    "XLY": ["AMZN","TSLA","HD","MCD","NKE","SBUX","LOW","TJX","BKNG","CMG",
            "LULU","ROST","DHI","LEN","ABNB","DASH","EBAY","MAR","HLT","YUM"],
    "XLP": ["PG","KO","PEP","COST","WMT","PM","MO","CL","MDLZ","KHC",
            "STZ","SYY","HSY","GIS","ADM","MNST","KDP","CHD","CLX","SJM"],
    "XLE": ["XOM","CVX","COP","SLB","EOG","MPC","PSX","VLO","OXY","HAL",
            "DVN","FANG","WMB","KMI","TRGP","BKR","CTRA","MRO","APA","AR"],
    "XLU": ["NEE","DUK","SO","D","AEP","SRE","EXC","XEL","ED","WEC",
            "AWK","ES","ATO","CMS","PNW","PPL","FE","DTE","AES","ETR"],
    "XLB": ["LIN","APD","SHW","FCX","NEM","ECL","DOW","NUE","VMC","MLM",
            "CF","MOS","BALL","PKG","IFF","EMN","CE","RPM","SEE","AVY"],
    "XLRE": ["PLD","AMT","CCI","EQIX","SPG","PSA","O","WELL","DLR","AVB",
             "VICI","MAA","EXR","ARE","UDR","ESS","REG","HST","KIM","CPT"],
    "XLC": ["META","GOOGL","GOOG","NFLX","DIS","CMCSA","T","VZ","TMUS","EA",
            "SPOT","RBLX","TTWO","WBD","PARA","MTCH","ZM","PINS","SNAP","LYV"],
}

ALPACA_HEADERS = {
    "APCA-API-KEY-ID": settings.ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": settings.ALPACA_SECRET_KEY,
}

ALPACA_DATA_URL = "https://data.alpaca.markets"
MAX_STORED_BARS = 300


# ============================================
# INDICATOR FUNCTIONS
# ============================================

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return float((100 - (100 / (1 + rs))).iloc[-1])

def calc_ema(series, span):
    return series.ewm(span=span).mean().iloc[-1]

def calc_macd(series):
    ema12 = series.ewm(span=12).mean()
    ema26 = series.ewm(span=26).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9).mean()
    histogram = macd_line - signal
    return {
        "macd": round(float(macd_line.iloc[-1]), 4),
        "signal": round(float(signal.iloc[-1]), 4),
        "histogram": round(float(histogram.iloc[-1]), 4),
    }

def calc_volume_profile(high, low, volume, bins=50):
    if len(high) < 10:
        return None, None, None, []
    price_min = float(low.min())
    price_max = float(high.max())
    if price_max <= price_min:
        return None, None, None, []
    bin_size = (price_max - price_min) / bins
    vp = {}
    for i in range(len(high)):
        h, l, v = float(high.iloc[i]), float(low.iloc[i]), float(volume.iloc[i])
        mid = (h + l) / 2
        bin_idx = int((mid - price_min) / bin_size)
        bin_idx = min(bin_idx, bins - 1)
        price_level = round(price_min + bin_idx * bin_size + bin_size / 2, 2)
        vp[price_level] = vp.get(price_level, 0) + v
    if not vp:
        return None, None, None, []
    poc_price = max(vp, key=vp.get)
    total_vol = sum(vp.values())
    sorted_levels = sorted(vp.items(), key=lambda x: x[1], reverse=True)
    cumulative = 0
    value_area = []
    for price_level, vol in sorted_levels:
        cumulative += vol
        value_area.append(price_level)
        if cumulative >= total_vol * 0.7:
            break
    va_high = max(value_area) if value_area else poc_price
    va_low = min(value_area) if value_area else poc_price
    distribution = [{"price": p, "volume": int(v)} for p, v in sorted(vp.items())]
    return round(poc_price, 2), round(va_high, 2), round(va_low, 2), distribution

def calc_setup_score(data):
    score = 0
    price = data.get("price", 0)
    rsi = data.get("rsi", 50)
    macd_hist = data.get("macd_histogram", 0)
    ema10 = data.get("ema10", 0)
    ema20 = data.get("ema20", 0)
    ema50 = data.get("ema50", 0)
    rel_vol = data.get("relative_volume", 1)
    poc = data.get("poc_price")
    sector_str = data.get("sector_strength", 50)
    pattern_bonus = data.get("pattern_bonus", 0)
    if price > ema10 > ema20 > ema50 and ema50 > 0: score += 25
    elif price > ema20 > ema50 and ema50 > 0: score += 15
    elif price > ema50 and ema50 > 0: score += 5
    if 40 <= rsi <= 60: score += 15
    elif 30 <= rsi < 40: score += 10
    elif rsi < 30: score += 5
    if macd_hist > 0: score += 10
    if rel_vol >= 2: score += 10
    elif rel_vol >= 1.5: score += 5
    if poc and price:
        dist = abs(price - poc) / price * 100
        if dist <= 2: score += 15
        elif dist <= 5: score += 8
    if sector_str >= 60: score += 10
    elif sector_str >= 40: score += 5
    score += pattern_bonus
    return min(score, 100)

def detect_setup_type(data):
    price = data.get("price", 0)
    rsi = data.get("rsi", 50)
    ema20 = data.get("ema20", 0)
    ema50 = data.get("ema50", 0)
    poc = data.get("poc_price")
    va_high = data.get("va_high")
    if poc and price and abs(price - poc) / price * 100 <= 3: return "pullback_to_poc"
    if va_high and price and price > va_high and rsi < 70: return "breakout"
    if ema20 > 0 and ema50 > 0 and price > ema50:
        if abs(price - ema20) / price * 100 <= 2: return "ema_bounce"
    if rsi < 35: return "oversold_reversal"
    if rsi > 70: return "overbought_warning"
    return "neutral"

def detect_fvg(df):
    if df is None or len(df) < 3: return []
    fvgs = []
    high = df["High"].values
    low = df["Low"].values
    for i in range(2, len(df)):
        if low[i] > high[i-2]:
            fvg_top = float(low[i]); fvg_bottom = float(high[i-2])
            fvg_size = fvg_top - fvg_bottom; mid_price = (fvg_top + fvg_bottom) / 2
            current = float(df["Close"].iloc[-1])
            filled = current <= fvg_top and current >= fvg_bottom
            fvgs.append({"type": "bullish", "top": round(fvg_top, 2), "bottom": round(fvg_bottom, 2),
                         "size": round(fvg_size, 2), "midpoint": round(mid_price, 2), "filled": filled, "bar_index": i})
        elif high[i] < low[i-2]:
            fvg_top = float(low[i-2]); fvg_bottom = float(high[i])
            fvg_size = fvg_top - fvg_bottom; mid_price = (fvg_top + fvg_bottom) / 2
            current = float(df["Close"].iloc[-1])
            filled = current >= fvg_bottom and current <= fvg_top
            fvgs.append({"type": "bearish", "top": round(fvg_top, 2), "bottom": round(fvg_bottom, 2),
                         "size": round(fvg_size, 2), "midpoint": round(mid_price, 2), "filled": filled, "bar_index": i})
    return fvgs[-5:] if len(fvgs) > 5 else fvgs

def detect_wyckoff_phase(df):
    if df is None or len(df) < 30:
        return {"phase": "unknown", "confidence": 0, "description": "", "signal": "neutral", "metrics": {}}
    close = df["Close"].values; volume = df["Volume"].values
    high = df["High"].values; low = df["Low"].values; n = len(close)
    current_price = float(close[-1])
    ema20 = float(pd.Series(close).ewm(span=20).mean().iloc[-1])
    ema50 = float(pd.Series(close).ewm(span=50).mean().iloc[-1])
    price_change_20 = ((current_price / float(close[-20])) - 1) * 100 if n >= 20 else 0
    price_change_10 = ((current_price / float(close[-10])) - 1) * 100 if n >= 10 else 0
    price_range_20 = ((max(close[-20:]) - min(close[-20:])) / float(close[-20])) * 100 if n >= 20 else 0
    vol_avg_old = float(np.mean(volume[-40:-20])) if n >= 40 else float(np.mean(volume[:20]))
    vol_avg_new = float(np.mean(volume[-20:]))
    vol_change = ((vol_avg_new - vol_avg_old) / vol_avg_old * 100) if vol_avg_old > 0 else 0
    higher_lows = True; lower_highs = True
    if n >= 20:
        for i in range(-15, -5):
            if low[i] < low[i-5]: higher_lows = False
            if high[i] > high[i-5]: lower_highs = False
    phase = "unknown"; confidence = 50; description = ""; signal = "neutral"
    if price_range_20 < 10 and vol_change < -10 and higher_lows:
        phase = "accumulation"; confidence = 70 + (10 if vol_change < -20 else 0)
        description = "Tight range with declining volume."; signal = "bullish_soon"
    elif price_change_20 > 5 and current_price > ema20 and current_price > ema50:
        phase = "markup"; confidence = 75 + (10 if vol_change > 10 else 0)
        description = "Strong uptrend."; signal = "bullish"
    elif price_range_20 < 10 and current_price > ema50 and lower_highs:
        phase = "distribution"; confidence = 65 + (15 if vol_change > 15 else 0)
        description = "Distributing near highs."; signal = "bearish_soon"
    elif price_change_20 < -5 and current_price < ema20 and current_price < ema50:
        phase = "markdown"; confidence = 75; description = "Downtrend."; signal = "bearish"
    elif n >= 10:
        recent_low = float(min(low[-10:]))
        prev_low = float(min(low[-30:-10])) if n >= 30 else float(min(low[:20]))
        if recent_low < prev_low and price_change_10 > 3:
            phase = "spring"; confidence = 60; description = "Potential spring."; signal = "strong_bullish"
    return {"phase": phase, "confidence": min(confidence, 95), "description": description, "signal": signal,
            "metrics": {"price_change_20d": round(price_change_20, 2), "price_change_10d": round(price_change_10, 2),
                        "higher_lows": higher_lows, "lower_highs": lower_highs}}

def calc_accumulation_score(df, poc_price, va_low, va_high):
    if df is None or len(df) < 20:
        return {"score": 0, "level": "unknown", "factors": []}
    close = df["Close"].values; volume = df["Volume"].values
    current_price = float(close[-1]); factors = []; score = 0
    if poc_price and current_price < poc_price:
        dist = abs(current_price - poc_price) / poc_price * 100
        if dist <= 5: score += 25; factors.append({"name": "Below POC", "score": 25, "detail": f"{round(dist,1)}% below", "pass": True})
        elif dist <= 15: score += 15; factors.append({"name": "Below POC", "score": 15, "detail": f"{round(dist,1)}% below", "pass": True})
        else: factors.append({"name": "Below POC", "score": 0, "detail": "too far", "pass": False})
    else: factors.append({"name": "Below POC", "score": 0, "detail": "above POC", "pass": False})
    if va_low and current_price <= va_low * 1.02:
        score += 20; factors.append({"name": "Near VA Low", "score": 20, "detail": "near/below", "pass": True})
    else: factors.append({"name": "Near VA Low", "score": 0, "detail": "above", "pass": False})
    if len(volume) >= 20:
        v1 = float(np.mean(volume[-20:-10])); v2 = float(np.mean(volume[-10:]))
        if v1 > 0:
            vd = (v2 - v1) / v1 * 100
            if vd < -15: score += 15; factors.append({"name": "Vol Decreasing", "score": 15, "detail": f"down {round(vd)}%", "pass": True})
            else: factors.append({"name": "Vol Decreasing", "score": 0, "detail": f"{round(vd)}%", "pass": False})
    score = min(score, 100)
    level = "strong" if score >= 70 else "moderate" if score >= 40 else "weak" if score >= 20 else "none"
    return {"score": score, "level": level, "factors": factors}

def detect_candlestick_patterns(df):
    if df is None or len(df) < 3: return []
    patterns = []
    o, h, l, c = df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values
    i = len(df) - 1; i1 = i - 1; i2 = i - 2
    body = abs(c[i] - o[i]); range_total = h[i] - l[i]
    if range_total == 0: return []
    upper_shadow = h[i] - max(o[i], c[i]); lower_shadow = min(o[i], c[i]) - l[i]
    is_bullish = c[i] > o[i]; is_bearish = c[i] < o[i]
    body1 = abs(c[i1] - o[i1]); is_bullish1 = c[i1] > o[i1]; is_bearish1 = c[i1] < o[i1]
    body_pct = body / range_total
    if body_pct < 0.35 and lower_shadow >= body * 2 and upper_shadow < body * 0.5:
        patterns.append({"name": "Hammer", "type": "bullish", "strength": "strong", "description": "Bullish reversal"})
    if body_pct < 0.35 and upper_shadow >= body * 2 and lower_shadow < body * 0.5:
        patterns.append({"name": "Shooting Star", "type": "bearish", "strength": "strong", "description": "Bearish reversal"})
    if is_bullish and is_bearish1 and c[i] > o[i1] and o[i] < c[i1] and body > body1 * 0.5:
        patterns.append({"name": "Bullish Engulfing", "type": "bullish", "strength": "strong", "description": "Bullish reversal"})
    if is_bearish and is_bullish1 and c[i] < o[i1] and o[i] > c[i1] and body > body1 * 0.5:
        patterns.append({"name": "Bearish Engulfing", "type": "bearish", "strength": "strong", "description": "Bearish reversal"})
    if body_pct < 0.15 and lower_shadow > body * 2 and upper_shadow > body * 2:
        patterns.append({"name": "Doji", "type": "neutral", "strength": "moderate", "description": "Indecision"})
    if len(df) >= 3:
        if is_bearish1 and body_pct < 0.2 and is_bullish and c[i] > (o[i1] + c[i1]) / 2:
            patterns.append({"name": "Morning Star", "type": "bullish", "strength": "strong", "description": "Bullish reversal"})
        if is_bullish1 and body_pct < 0.2 and is_bearish and c[i] < (o[i1] + c[i1]) / 2:
            patterns.append({"name": "Evening Star", "type": "bearish", "strength": "strong", "description": "Bearish reversal"})
    return patterns

def get_pattern_score_bonus(patterns):
    if not patterns: return 0
    bonus = 0
    for p in patterns:
        if p["type"] == "bullish": bonus += 5 if p["strength"] == "strong" else 3
        elif p["type"] == "bearish": bonus -= 3 if p["strength"] == "strong" else 1
    return max(-10, min(15, bonus))


# ============================================
# INCREMENTAL BAR STORAGE
# ============================================

async def fetch_bars_from_api(client, symbol, limit=252):
    end = datetime.utcnow()
    start = end - timedelta(days=400)
    url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars"
    params = {
        "timeframe": "1Day",
        "start": start.strftime("%Y-%m-%dT00:00:00Z"),
        "end": end.strftime("%Y-%m-%dT23:59:59Z"),
        "limit": limit,
        "feed": "iex",
        "adjustment": "split",
    }
    try:
        r = await client.get(url, headers=ALPACA_HEADERS, params=params)
        if r.status_code != 200:
            return []
        data = r.json()
        return data.get("bars", [])
    except Exception:
        return []


async def get_or_fetch_bars(client, db, symbol):
    doc = await db.stock_bars.find_one({"ticker": symbol})
    if doc and doc.get("bars") and len(doc["bars"]) >= 20:
        new_bars_raw = await fetch_bars_from_api(client, symbol, limit=5)
        if new_bars_raw:
            existing_dates = {b["date"] for b in doc["bars"]}
            new_bars = []
            for b in new_bars_raw:
                bar_date = b["t"][:10]
                if bar_date not in existing_dates:
                    new_bars.append({
                        "date": bar_date,
                        "o": b["o"], "h": b["h"], "l": b["l"], "c": b["c"], "v": b["v"],
                    })
            if new_bars:
                all_bars = doc["bars"] + new_bars
                all_bars.sort(key=lambda x: x["date"])
                all_bars = all_bars[-MAX_STORED_BARS:]
                last_bar = all_bars[-1]["date"]
                await db.stock_bars.update_one(
                    {"ticker": symbol},
                    {"$set": {"bars": all_bars, "last_bar_date": last_bar, "updated_at": datetime.utcnow()}}
                )
                return _bars_to_df(all_bars)
            else:
                return _bars_to_df(doc["bars"])
        else:
            return _bars_to_df(doc["bars"])
    else:
        bars_raw = await fetch_bars_from_api(client, symbol, limit=252)
        if not bars_raw:
            return None
        bars = []
        for b in bars_raw:
            bars.append({
                "date": b["t"][:10],
                "o": b["o"], "h": b["h"], "l": b["l"], "c": b["c"], "v": b["v"],
            })
        bars.sort(key=lambda x: x["date"])
        bars = bars[-MAX_STORED_BARS:]
        last_bar = bars[-1]["date"] if bars else ""
        await db.stock_bars.update_one(
            {"ticker": symbol},
            {"$set": {"ticker": symbol, "bars": bars, "last_bar_date": last_bar, "updated_at": datetime.utcnow()}},
            upsert=True,
        )
        return _bars_to_df(bars)


def _bars_to_df(bars):
    if not bars or len(bars) < 5:
        return None
    df = pd.DataFrame(bars)
    df = df.rename(columns={"date": "datetime", "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    return df


async def get_or_fetch_bars_batch(client, db, symbols, max_concurrent=10):
    semaphore = asyncio.Semaphore(max_concurrent)
    results = {}
    async def _fetch_one(sym):
        async with semaphore:
            df = await get_or_fetch_bars(client, db, sym)
            results[sym] = df
    await asyncio.gather(*[_fetch_one(s) for s in symbols])
    return results


async def fetch_bars(client, symbol):
    """Backward-compatible wrapper. Used by stock_search.py."""
    db = get_db()
    return await get_or_fetch_bars(client, db, symbol)


# ============================================
# STOCK ANALYSIS
# ============================================

def analyze_stock(ticker, df, sector_code, sector_scores):
    if df is None or len(df) < 20:
        return None
    close = df["Close"]; volume = df["Volume"]; high = df["High"]; low = df["Low"]
    price = float(close.iloc[-1]); prev_close = float(close.iloc[-2])
    change_pct = round(((price - prev_close) / prev_close) * 100, 2)
    avg_vol = float(volume.rolling(20).mean().iloc[-1]); curr_vol = float(volume.iloc[-1])
    rel_vol = round(curr_vol / avg_vol, 2) if avg_vol > 0 else 1
    rsi = round(float(calc_rsi(close)), 2); macd = calc_macd(close)
    ema10 = round(float(calc_ema(close, 10)), 2)
    ema20 = round(float(calc_ema(close, 20)), 2)
    ema50 = round(float(calc_ema(close, 50)), 2)
    poc_result = calc_volume_profile(high, low, volume)
    high_52w = round(float(high.max()), 2); low_52w = round(float(low.min()), 2)
    pct_from_high = round(((price - high_52w) / high_52w) * 100, 2) if high_52w > 0 else 0
    pct_from_low = round(((price - low_52w) / low_52w) * 100, 2) if low_52w > 0 else 0
    range_position = round(((price - low_52w) / (high_52w - low_52w)) * 100, 1) if (high_52w - low_52w) > 0 else 50
    poc = poc_result[0]; va_high = poc_result[1]; va_low = poc_result[2]; vp_distribution = poc_result[3]
    patterns = detect_candlestick_patterns(df)
    fvgs = detect_fvg(df); wyckoff = detect_wyckoff_phase(df)
    accumulation = calc_accumulation_score(df, poc, va_low, va_high)
    ds = close.diff(); gs = ds.where(ds > 0, 0).rolling(14).mean()
    ls = (-ds.where(ds < 0, 0)).rolling(14).mean(); rs_s = 100 - (100 / (1 + gs / ls))
    e10s = close.ewm(span=10).mean(); e20s = close.ewm(span=20).mean(); e50s = close.ewm(span=50).mean()
    price_history = []
    start_idx = max(20, len(df) - 90)
    for idx in range(start_idx, len(df)):
        dr = float(rs_s.iloc[idx]) if not pd.isna(rs_s.iloc[idx]) else 50
        price_history.append({
            "date": df["datetime"].iloc[idx].strftime("%Y-%m-%d") if "datetime" in df.columns else f"d{idx}",
            "close": round(float(close.iloc[idx]), 2), "high": round(float(high.iloc[idx]), 2),
            "low": round(float(low.iloc[idx]), 2), "volume": int(volume.iloc[idx]),
            "rsi": round(dr, 1), "ema10": round(float(e10s.iloc[idx]), 2),
            "ema20": round(float(e20s.iloc[idx]), 2), "ema50": round(float(e50s.iloc[idx]), 2),
        })
    pattern_bonus = get_pattern_score_bonus(patterns)
    patterns_list = [{"name": p["name"], "type": p["type"], "strength": p["strength"], "description": p["description"]} for p in patterns]
    ind_data = {"price": price, "rsi": rsi, "macd_histogram": macd["histogram"],
        "ema10": ema10, "ema20": ema20, "ema50": ema50, "relative_volume": rel_vol,
        "poc_price": poc, "va_high": va_high, "change_pct": change_pct,
        "sector_strength": sector_scores.get(sector_code, 50), "pattern_bonus": pattern_bonus}
    setup_score = calc_setup_score(ind_data); setup_type = detect_setup_type(ind_data)
    return {
        "ticker": ticker, "name": ticker, "sector_code": sector_code,
        "price": round(price, 2), "change_pct": change_pct,
        "avg_volume": round(avg_vol, 0), "relative_volume": rel_vol,
        "rsi": rsi, "macd": macd, "ema10": ema10, "ema20": ema20, "ema50": ema50,
        "momentum_score": rsi, "volume_score": round(rel_vol * 30, 2),
        "poc_price": poc, "value_area_high": va_high, "value_area_low": va_low,
        "setup_score": setup_score, "setup_type": setup_type, "vp_distribution": vp_distribution,
        "candlestick_patterns": patterns_list, "fvg": fvgs, "wyckoff": wyckoff,
        "accumulation": accumulation, "price_history": price_history, "pattern_bonus": pattern_bonus,
        "high_52w": high_52w, "low_52w": low_52w, "pct_from_high": pct_from_high,
        "pct_from_low": pct_from_low, "range_position": range_position,
        "updated_at": datetime.utcnow(),
    }


# ============================================
# SECTORS
# ============================================

async def fetch_and_analyze_sectors(force=False):
    db = get_db()
    t_start = time.time()
    print("=" * 50)
    print("SECTORS REFRESH (Incremental)")
    print("=" * 50)

    async with httpx.AsyncClient(timeout=30) as client:
        all_syms = [
            "SPY", "QQQ", "IWM", "DIA", "FXE", "UUP",
            "TLT", "HYG", "LQD", "GLD", "USO", "RSP", "IWO", "VXX", "EEM", "IYT",
        ] + list(SECTOR_MAP.keys())
        bars_map = await get_or_fetch_bars_batch(client, db, all_syms, max_concurrent=8)

        spy_df = bars_map.get("SPY")
        spy_return = 0
        if spy_df is not None and len(spy_df) >= 20:
            spy_return = ((float(spy_df["Close"].iloc[-1]) / float(spy_df["Close"].iloc[-20])) - 1) * 100
            spy_close = spy_df["Close"]
            spy_ema20 = float(spy_close.ewm(span=20).mean().iloc[-1])
            spy_ema50 = float(spy_close.ewm(span=50).mean().iloc[-1])
            d = spy_close.diff(); g = d.where(d > 0, 0).rolling(14).mean()
            lo = (-d.where(d < 0, 0)).rolling(14).mean()
            spy_rsi_val = float((100 - (100 / (1 + g / lo))).iloc[-1])
            spy_price = float(spy_close.iloc[-1])
            spy_change = float(((spy_close.iloc[-1] / spy_close.iloc[-2]) - 1) * 100)
            await db.market_regime.update_one({"symbol": "SPY"},
                {"$set": {"symbol": "SPY", "price": round(spy_price, 2), "change_pct": round(spy_change, 2),
                          "ema20": round(spy_ema20, 2), "ema50": round(spy_ema50, 2), "rsi": round(spy_rsi_val, 1),
                          "return_20d": round(spy_return, 2), "updated_at": datetime.utcnow()}}, upsert=True)
            print(f"  SPY: ${spy_price:.2f} RSI={spy_rsi_val:.1f} ret20d={spy_return:.2f}%")

        for idx_sym in ["QQQ", "IWM", "DIA"]:
            idx_df = bars_map.get(idx_sym)
            if idx_df is not None and len(idx_df) >= 2:
                ip = float(idx_df["Close"].iloc[-1]); ipp = float(idx_df["Close"].iloc[-2])
                ic = round(((ip - ipp) / ipp) * 100, 2)
                ir20 = round(((float(idx_df["Close"].iloc[-1]) / float(idx_df["Close"].iloc[-20])) - 1) * 100, 2) if len(idx_df) >= 20 else 0
                await db.market_regime.update_one({"symbol": idx_sym},
                    {"$set": {"symbol": idx_sym, "price": round(ip, 2), "change_pct": ic, "return_20d": ir20, "updated_at": datetime.utcnow()}}, upsert=True)
                print(f"  {idx_sym}: ${ip:.2f} ({ic:+.2f}%)")

        for crypto in ["BTC/USD", "ETH/USD"]:
            try:
                cr = await client.get(f"{ALPACA_DATA_URL}/v1beta3/crypto/us/bars",
                    headers=ALPACA_HEADERS, params={"symbols": crypto, "timeframe": "1Day", "limit": 5})
                if cr.status_code == 200:
                    cbars = cr.json().get("bars", {}).get(crypto, [])
                    if cbars and len(cbars) >= 2:
                        cp = float(cbars[-1].get("c", 0)); cpp = float(cbars[-2].get("c", cp))
                        cc = round(((cp - cpp) / cpp) * 100, 2) if cpp > 0 else 0
                        await db.market_regime.update_one({"symbol": crypto},
                            {"$set": {"symbol": crypto, "price": round(cp, 2), "change_pct": cc, "updated_at": datetime.utcnow()}}, upsert=True)
                        print(f"  {crypto}: ${cp:.2f} ({cc:+.2f}%)")
            except Exception as e:
                print(f"  {crypto} error: {e}")

        for fx in ["FXE", "UUP"]:
            fx_df = bars_map.get(fx)
            if fx_df is not None and len(fx_df) >= 2:
                fp = float(fx_df["Close"].iloc[-1]); fpp = float(fx_df["Close"].iloc[-2])
                fc = round(((fp - fpp) / fpp) * 100, 2)
                await db.market_regime.update_one({"symbol": fx},
                    {"$set": {"symbol": fx, "price": round(fp, 2), "change_pct": fc, "updated_at": datetime.utcnow()}}, upsert=True)
                print(f"  {fx}: ${fp:.2f} ({fc:+.2f}%)")

        # Macro indicators (Bonds, Commodities, Breadth, Risk Appetite)
        macro_syms = ["TLT", "HYG", "LQD", "GLD", "USO", "RSP", "IWO", "VXX", "EEM", "IYT"]
        for sym in macro_syms:
            sym_df = bars_map.get(sym)
            if sym_df is not None and len(sym_df) >= 2:
                sp = float(sym_df["Close"].iloc[-1])
                spp = float(sym_df["Close"].iloc[-2])
                sc = round(((sp - spp) / spp) * 100, 2)
                extra = {"symbol": sym, "price": round(sp, 2), "change_pct": sc, "updated_at": datetime.utcnow()}
                # Add RSI and EMAs for richer analysis
                if len(sym_df) >= 20:
                    try:
                        extra["rsi"] = round(float(calc_rsi(sym_df["Close"])), 1)
                        extra["ema20"] = round(float(calc_ema(sym_df["Close"], 20)), 2)
                        extra["ema50"] = round(float(calc_ema(sym_df["Close"], 50)), 2) if len(sym_df) >= 50 else 0
                        extra["return_20d"] = round(((sp / float(sym_df["Close"].iloc[-20])) - 1) * 100, 2)
                    except:
                        pass
                await db.market_regime.update_one({"symbol": sym}, {"$set": extra}, upsert=True)
                print(f"  {sym}: ${sp:.2f} ({sc:+.2f}%)")
            else:
                print(f"  {sym}: no data")
        
        results = []
        for etf, name in SECTOR_MAP.items():
            try:
                df = bars_map.get(etf)
                if df is None or len(df) < 20: print(f"  SKIP {etf}"); continue
                close = df["Close"]; volume = df["Volume"]
                ret_20d = ((float(close.iloc[-1]) / float(close.iloc[-20])) - 1) * 100
                strength = round(float(ret_20d - spy_return), 2)
                rsi = round(float(calc_rsi(close)), 2)
                ema10 = float(calc_ema(close, 10)); ema20_val = float(calc_ema(close, 20)); ema50 = float(calc_ema(close, 50))
                price = float(close.iloc[-1])
                avg_vol = float(volume.rolling(20).mean().iloc[-1]); curr_vol = float(volume.iloc[-1])
                rel_vol = round(curr_vol / avg_vol, 2) if avg_vol > 0 else 1
                trend = 90 if price > ema10 > ema20_val > ema50 else (70 if price > ema20_val > ema50 else (50 if price > ema50 else 30))
                composite = round((strength * 2 + trend + rsi) / 4, 2)
                history = []
                d = close.diff(); g = d.where(d > 0, 0).rolling(14).mean()
                lo = (-d.where(d < 0, 0)).rolling(14).mean(); rs_s = 100 - (100 / (1 + g / lo))
                e10s = close.ewm(span=10).mean(); e20s = close.ewm(span=20).mean(); e50s = close.ewm(span=50).mean()
                for idx in range(max(20, len(df) - 90), len(df)):
                    dc = float(close.iloc[idx]); dr = float(rs_s.iloc[idx]) if not pd.isna(rs_s.iloc[idx]) else 50
                    zone = "oversold" if dr <= 30 else ("weak" if dr <= 40 else ("overbought" if dr >= 70 else ("strong" if dr >= 60 else "neutral")))
                    history.append({"date": df["datetime"].iloc[idx].strftime("%Y-%m-%d") if "datetime" in df.columns else f"d{idx}",
                        "close": round(dc, 2), "rsi": round(dr, 1), "ema10": round(float(e10s.iloc[idx]), 2),
                        "ema20": round(float(e20s.iloc[idx]), 2), "ema50": round(float(e50s.iloc[idx]), 2), "zone": zone})
                sector_doc = {"code": etf, "name": name, "etf_ticker": etf, "price": round(price, 2),
                    "return_20d": round(float(ret_20d), 2), "strength_score": strength, "trend_score": trend,
                    "volume_score": round(rel_vol * 30, 2), "rsi": rsi, "composite_score": composite,
                    "history": history, "updated_at": datetime.utcnow()}
                await db.sectors.update_one({"code": etf}, {"$set": sector_doc}, upsert=True)
                results.append(sector_doc)
                print(f"  OK {etf}: ${price:.2f} score={composite:.2f}")
            except Exception as e:
                print(f"  ERROR {etf}: {e}"); traceback.print_exc()

    elapsed = round(time.time() - t_start, 1)
    print(f"\nSECTORS DONE: {len(results)}/11 in {elapsed}s")
    return results


# ============================================
# STOCKS
# ============================================

async def fetch_and_analyze_stocks(force=False):
    db = get_db()
    t_start = time.time()
    print("=" * 50)
    print("STOCKS REFRESH (Incremental + Parallel)")
    print("=" * 50)

    sector_scores = {}
    async for s in db.sectors.find():
        sector_scores[s["code"]] = s.get("composite_score", 50)

    all_tickers = []
    ticker_to_sector = {}
    for sector_code, tickers in SECTOR_STOCKS.items():
        for ticker in tickers:
            all_tickers.append(ticker)
            ticker_to_sector[ticker] = sector_code

    total = len(all_tickers)
    print(f"  Total stocks: {total}")

    batch_size = 20
    results = []

    async with httpx.AsyncClient(timeout=30) as client:
        for batch_num in range(0, total, batch_size):
            batch = all_tickers[batch_num:batch_num + batch_size]
            batch_idx = batch_num // batch_size + 1
            total_batches = (total + batch_size - 1) // batch_size
            t_batch = time.time()
            print(f"\n  Batch {batch_idx}/{total_batches} ({len(batch)} stocks)")

            bars_map = await get_or_fetch_bars_batch(client, db, batch, max_concurrent=10)

            success = 0; skipped = 0
            for ticker in batch:
                df = bars_map.get(ticker)
                sector_code = ticker_to_sector.get(ticker, "UNKNOWN")
                asset_doc = analyze_stock(ticker, df, sector_code, sector_scores)
                if asset_doc:
                    await db.assets.update_one({"ticker": ticker}, {"$set": asset_doc}, upsert=True)
                    results.append(asset_doc)
                    success += 1
                else:
                    skipped += 1

            batch_time = round(time.time() - t_batch, 1)
            print(f"    -> {success} OK, {skipped} skipped in {batch_time}s")

    elapsed = round(time.time() - t_start, 1)
    print(f"\n{'=' * 50}")
    print(f"STOCKS DONE: {len(results)}/{total} in {elapsed}s")
    if results:
        print(f"  Average: {elapsed/len(results):.2f}s per stock")
    first_run_count = await db.stock_bars.count_documents({})
    print(f"  Bars in MongoDB: {first_run_count} stocks cached")
    print("=" * 50)
    return results
