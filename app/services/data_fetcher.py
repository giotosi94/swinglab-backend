import httpx
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta
from app.db.mongodb import get_db
from app.config import settings
import traceback
import time
from app.services.max_strategy import analyze_max_strategy

SECTOR_MAP = {
    "XLK": "Technology", "XLF": "Financials", "XLV": "Health Care",
    "XLI": "Industrials", "XLY": "Consumer Discretionary", "XLP": "Consumer Staples",
    "XLE": "Energy", "XLU": "Utilities", "XLB": "Materials",
    "XLRE": "Real Estate", "XLC": "Communication Services",
}

SECTOR_STOCKS = {
    "XLK": ["AAPL","MSFT","NVDA","AVGO","AMD","CRM","ADBE","INTC","CSCO","ORCL",
            "PLTR","NOW","SNOW","CRWD","PANW","MNDY","SHOP","XYZ","UBER","DDOG"],
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
            "DVN","FANG","WMB","KMI","TRGP","BKR","MRO","APA","AR"],
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
MAX_STORED_BARS = 1000
MAX_STALE_CALENDAR_DAYS = 7
LEGACY_TICKERS = ["CTRA", "SQ"]


def _data_freshness(df):
    if df is None or len(df) == 0 or "datetime" not in df.columns:
        return {"status": "NO_DATA", "eligible": False, "last_bar_date": None, "calendar_days_old": None}
    last_bar = pd.to_datetime(df["datetime"].iloc[-1]).to_pydatetime()
    if last_bar.tzinfo is not None:
        last_bar = last_bar.replace(tzinfo=None)
    days_old = (datetime.utcnow().date() - last_bar.date()).days
    stale = days_old > MAX_STALE_CALENDAR_DAYS
    return {
        "status": "STALE_OR_DELISTED" if stale else "FRESH",
        "eligible": not stale,
        "last_bar_date": last_bar.strftime("%Y-%m-%d"),
        "calendar_days_old": days_old,
    }


async def cleanup_legacy_max_strategy_data(db):
    return {
        "assets": (await db.assets.delete_many({"ticker": {"$in": LEGACY_TICKERS}})).deleted_count,
        "signals": (await db.max_strategy_signals.delete_many({"ticker": {"$in": LEGACY_TICKERS}})).deleted_count,
        "daily_bars": (await db.stock_bars.delete_many({"ticker": {"$in": LEGACY_TICKERS}})).deleted_count,
        "bars_4h": (await db.stock_bars_4h.delete_many({"ticker": {"$in": LEGACY_TICKERS}})).deleted_count,
    }


async def mark_stale_asset(db, ticker, sector_code, freshness):
    await db.assets.update_one(
        {"ticker": ticker},
        {"$set": {
            "ticker": ticker,
            "name": ticker,
            "sector_code": sector_code,
            "data_status": freshness["status"],
            "data_eligible": False,
            "last_bar_date": freshness["last_bar_date"],
            "calendar_days_old": freshness["calendar_days_old"],
            "max_strategy": {
                "status": "STALE_OR_DELISTED",
                "version": "max_structure_v1_5_2",
                "data_eligible": False,
                "strategy_eligible": False,
                "watch_ready": False,
                "trade_ready": False,
                "live_eligible": False,
                "live_entry_enabled": False,
                "rejection_reasons": ["STALE_OR_DELISTED"],
                "entry_plan": {"status": "BLOCKED", "order_action": "WAIT", "blocking_phase": True},
            },
            "updated_at": datetime.utcnow(),
        }},
        upsert=True,
    )
    await db.max_strategy_signals.delete_many({"ticker": ticker, "outcomes.bars_observed": 0})


async def fetch_long_history_symbol(client, symbol, target_bars=750):
    target_bars = max(300, min(int(target_bars), MAX_STORED_BARS))
    end = datetime.utcnow() - timedelta(minutes=20)
    start = end - timedelta(days=int(target_bars * 1.75) + 120)
    url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars"
    page_token = None
    raw_bars = []
    while True:
        params = {
            "timeframe": "1Day",
            "start": start.strftime("%Y-%m-%dT00:00:00Z"),
            "end": end.strftime("%Y-%m-%dT23:59:59Z"),
            "limit": min(10000, target_bars + 100),
            "feed": "iex",
            "adjustment": "split",
            "sort": "asc",
        }
        if page_token:
            params["page_token"] = page_token
        response = await client.get(url, headers=ALPACA_HEADERS, params=params)
        if response.status_code != 200:
            raise RuntimeError(f"Alpaca {response.status_code}: {response.text[:200]}")
        payload = response.json()
        raw_bars.extend(payload.get("bars", []))
        page_token = payload.get("next_page_token")
        if not page_token or len(raw_bars) >= target_bars:
            break
    normalized = {}
    for bar in raw_bars:
        date = str(bar.get("t", ""))[:10]
        if date:
            normalized[date] = {"date": date, "o": bar.get("o"), "h": bar.get("h"), "l": bar.get("l"), "c": bar.get("c"), "v": bar.get("v", 0)}
    return sorted(normalized.values(), key=lambda item: item["date"])[-target_bars:]


async def get_bars_coverage(target_bars=750):
    db = get_db()
    docs = await db.stock_bars.find({}, {"ticker": 1, "bars": 1}).to_list(400)
    counts = sorted(len(doc.get("bars", [])) for doc in docs if doc.get("ticker"))
    complete = sum(1 for count in counts if count >= target_bars)
    return {
        "target_bars": target_bars,
        "tickers_total": len(counts),
        "tickers_complete": complete,
        "coverage_pct": round(complete / len(counts) * 100, 1) if counts else 0,
        "bars_min": min(counts) if counts else 0,
        "bars_median": int(np.median(counts)) if counts else 0,
        "bars_max": max(counts) if counts else 0,
    }


async def backfill_long_history(target_bars=750, max_concurrent=4):
    target_bars = max(300, min(int(target_bars), MAX_STORED_BARS))
    db = get_db()
    job_id = f"stock_bars_{target_bars}"
    symbols = sorted(set([ticker for tickers in SECTOR_STOCKS.values() for ticker in tickers] + list(SECTOR_MAP.keys()) + ["SPY", "QQQ", "IWM", "DIA"]))
    await db.backfill_jobs.update_one({"_id": job_id}, {"$set": {"status": "running", "target_bars": target_bars, "total": len(symbols), "processed": 0, "completed": 0, "skipped": 0, "failed": 0, "started_at": datetime.utcnow(), "updated_at": datetime.utcnow(), "errors": []}}, upsert=True)
    semaphore = asyncio.Semaphore(max(1, min(int(max_concurrent), 8)))
    async with httpx.AsyncClient(timeout=45) as client:
        async def process(symbol):
            async with semaphore:
                current = await db.stock_bars.find_one({"ticker": symbol}, {"bars": 1})
                current_bars = current.get("bars", []) if current else []
                if len(current_bars) >= target_bars:
                    return "skipped", symbol, len(current_bars), None
                try:
                    fetched = await fetch_long_history_symbol(client, symbol, target_bars)
                    merged = {bar["date"]: bar for bar in current_bars if bar.get("date")}
                    merged.update({bar["date"]: bar for bar in fetched if bar.get("date")})
                    bars = sorted(merged.values(), key=lambda item: item["date"])[-MAX_STORED_BARS:]
                    await db.stock_bars.update_one({"ticker": symbol}, {"$set": {"ticker": symbol, "bars": bars, "last_bar_date": bars[-1]["date"] if bars else "", "history_target": target_bars, "history_backfilled_at": datetime.utcnow(), "updated_at": datetime.utcnow()}}, upsert=True)
                    return "completed", symbol, len(bars), None
                except Exception as error:
                    return "failed", symbol, len(current_bars), str(error)
        tasks = [asyncio.create_task(process(symbol)) for symbol in symbols]
        counters = {"completed": 0, "skipped": 0, "failed": 0}
        errors = []
        for future in asyncio.as_completed(tasks):
            status, symbol, count, error = await future
            counters[status] += 1
            if error:
                errors.append({"ticker": symbol, "error": error})
            await db.backfill_jobs.update_one({"_id": job_id}, {"$set": {**counters, "processed": sum(counters.values()), "last_ticker": symbol, "last_bar_count": count, "errors": errors[-30:], "updated_at": datetime.utcnow()}})
    coverage = await get_bars_coverage(target_bars)
    status = "completed" if counters["failed"] == 0 else "completed_with_errors"
    await db.backfill_jobs.update_one({"_id": job_id}, {"$set": {"status": status, "coverage": coverage, "finished_at": datetime.utcnow()}})
    return {"job_id": job_id, "status": status, "coverage": coverage, **counters}


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


def _detect_poc_shift(ticker, price, poc, prev_position=None):
    """
    🆕 POC SHIFT (metodo Rea) sui daily — PERSISTENTE (prev letto dal DB).
    Rileva quando il POC passa da SOPRA il prezzo (resistenza) a SOTTO (supporto).
    Lo shift sopra→sotto è bullish: il muro è diventato pavimento.
    """
    result = {
        "poc_position": "unknown",
        "shifted_bull": False,
        "poc_distance_pct": None,
    }
    if not poc or not price or poc <= 0 or price <= 0:
        return result

    dist_pct = (price - poc) / poc * 100
    result["poc_distance_pct"] = round(dist_pct, 2)
    current_pos = "below" if dist_pct > 0 else "above"
    result["poc_position"] = current_pos

    # Shift bull: prima POC sopra (above), ora sotto (below)
    if prev_position == "above" and current_pos == "below":
        result["shifted_bull"] = True

    return result


def calc_weekly_trend(df):
    """MTF Light: resample daily bars -> weekly per il trend di fondo. Zero API extra."""
    default = {"weekly_trend": "UNKNOWN", "aligned": True, "score": 50,
               "weekly_rsi": 50, "weekly_ema20_slope": "flat", "bars": 0}
    if df is None or len(df) < 60 or "datetime" not in df.columns:
        return default
    try:
        w = df.set_index("datetime").resample("W-FRI").agg({
            "Open": "first", "High": "max", "Low": "min",
            "Close": "last", "Volume": "sum",
        }).dropna()
    except Exception:
        return default
    if len(w) < 12:
        return {**default, "bars": len(w)}

    wclose = w["Close"]
    wprice = float(wclose.iloc[-1])
    wema10 = float(wclose.ewm(span=10).mean().iloc[-1])
    wema20_series = wclose.ewm(span=20).mean()
    wema20 = float(wema20_series.iloc[-1])
    wema50 = float(wclose.ewm(span=min(50, len(w))).mean().iloc[-1])
    wrsi = round(float(calc_rsi(wclose)), 1) if len(w) >= 15 else 50

    slope = "flat"
    if len(wema20_series) >= 4:
        prev = float(wema20_series.iloc[-4])
        if wema20 > prev * 1.005:
            slope = "rising"
        elif wema20 < prev * 0.995:
            slope = "falling"

    if wprice > wema10 > wema20 > wema50 and wema50 > 0:
        trend, score = "BULL", 90
    elif wprice > wema20 > wema50 and wema50 > 0:
        trend, score = "BULL", 75
    elif wprice > wema50 and wema50 > 0:
        trend, score = "NEUTRAL", 55
    elif wprice > wema20:
        trend, score = "NEUTRAL", 45
    else:
        trend, score = "BEAR", 25

    aligned = trend in ("BULL", "NEUTRAL") and slope != "falling"

    return {
        "weekly_trend": trend,
        "weekly_rsi": wrsi,
        "weekly_ema20_slope": slope,
        "weekly_price": round(wprice, 2),
        "weekly_ema20": round(wema20, 2),
        "weekly_ema50": round(wema50, 2),
        "aligned": aligned,
        "score": score,
        "bars": len(w),
    }


# ============================================
# INCREMENTAL BAR STORAGE
# ============================================

async def fetch_bars_from_api(client, symbol, limit=252):
    """Fetch bars: Alpaca IEX first (fresh, sort=desc), Twelve Data fallback."""
    # 🔧 v4.4 — sort=desc per ottenere le barre PIU' RECENTI (non le piu' vecchie)
    end = datetime.utcnow() - timedelta(minutes=20)
    start = end - timedelta(days=400)
    url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars"
    params = {
        "timeframe": "1Day",
        "start": start.strftime("%Y-%m-%dT00:00:00Z"),
        "end": end.strftime("%Y-%m-%dT23:59:59Z"),
        "limit": limit,
        "feed": "iex",
        "adjustment": "split",
        "sort": "desc",
    }
    try:
        r = await client.get(url, headers=ALPACA_HEADERS, params=params)
        if r.status_code == 200:
            data = r.json()
            bars = data.get("bars", [])
            if bars:
                bars.reverse()  # desc -> cronologico (il resto del codice assume ascendente)
                return bars
    except Exception as e:
        print(f"  Alpaca error {symbol}: {e}")

    # Fallback: Twelve Data (solo se Alpaca fallisce)
    try:
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": symbol,
            "interval": "1day",
            "outputsize": str(limit),
            "apikey": settings.TWELVEDATA_API_KEY,
        }
        r = await client.get(url, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if "values" in data:
                bars = []
                for v in reversed(data["values"]):
                    bars.append({
                        "t": v["datetime"] + "T00:00:00Z",
                        "o": float(v["open"]),
                        "h": float(v["high"]),
                        "l": float(v["low"]),
                        "c": float(v["close"]),
                        "v": int(v["volume"]),
                    })
                return bars
    except Exception:
        pass

    return []


async def _fetch_bars_alpaca(client, symbol, limit=252):
    """Fetch bars directly from Alpaca IEX (historical bulk, sort=desc)."""
    end = datetime.utcnow() - timedelta(minutes=20)
    start = end - timedelta(days=400)
    url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars"
    params = {
        "timeframe": "1Day",
        "start": start.strftime("%Y-%m-%dT00:00:00Z"),
        "end": end.strftime("%Y-%m-%dT23:59:59Z"),
        "limit": limit,
        "feed": "iex",
        "adjustment": "split",
        "sort": "desc",
    }
    try:
        r = await client.get(url, headers=ALPACA_HEADERS, params=params)
        if r.status_code != 200:
            return []
        data = r.json()
        bars = data.get("bars", [])
        bars.reverse()  # desc -> cronologico
        return bars
    except Exception:
        return []


async def get_or_fetch_bars(client, db, symbol):
    doc = await db.stock_bars.find_one({"ticker": symbol})

    if doc and doc.get("bars") and len(doc["bars"]) >= 20:
        # Recency dell'ultima barra salvata
        last_bar_date = doc["bars"][-1].get("date", "")
        try:
            days_old = (datetime.utcnow() - datetime.strptime(last_bar_date, "%Y-%m-%d")).days
        except Exception:
            days_old = 999

        # Freschezza dell'ultimo update
        last_update = doc.get("updated_at")
        hours_since_update = 999
        if last_update:
            hours_since_update = (datetime.utcnow() - last_update).total_seconds() / 3600

        # 🔧 v4.4 — Cache valida SOLO se aggiornata di recente E l'ultima barra è recente.
        # (max 4 giorni per coprire weekend/festivi). Questo evita il "congelamento"
        # in cui updated_at è fresco ma le barre sono vecchie di settimane.
        if hours_since_update < 4 and days_old <= 4:
            return _bars_to_df(doc["bars"])

        # Altrimenti fetch fresh e merge (fetch_bars_from_api ora ritorna le barre recenti)
        new_bars_raw = await fetch_bars_from_api(client, symbol, limit=90)
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

        # Nessuna barra nuova: aggiorna solo il timestamp per non martellare l'API
        await db.stock_bars.update_one(
            {"ticker": symbol},
            {"$set": {"updated_at": datetime.utcnow()}}
        )
        return _bars_to_df(doc["bars"])

    else:
        # Step 1: Bulk historical from Alpaca IEX (sort=desc -> recenti)
        alpaca_bars_raw = await _fetch_bars_alpaca(client, symbol, limit=252)
        bars = []
        for b in (alpaca_bars_raw or []):
            bars.append({
                "date": b["t"][:10],
                "o": b["o"], "h": b["h"], "l": b["l"], "c": b["c"], "v": b["v"],
            })

        # Step 2: Fill recent gap con fetch_bars_from_api (Alpaca desc / Twelve Data)
        recent_raw = await fetch_bars_from_api(client, symbol, limit=45)
        if recent_raw:
            existing_dates = {b["date"] for b in bars}
            for b in recent_raw:
                bar_date = b["t"][:10]
                if bar_date not in existing_dates:
                    bars.append({
                        "date": bar_date,
                        "o": b["o"], "h": b["h"], "l": b["l"], "c": b["c"], "v": b["v"],
                    })

        if not bars:
            return None

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


async def fetch_4h_bars_from_api(client, symbol, limit=240):
    end = datetime.utcnow() - timedelta(minutes=20)
    start = end - timedelta(days=180)
    url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars"
    params = {
        "timeframe": "4Hour",
        "start": start.strftime("%Y-%m-%dT00:00:00Z"),
        "end": end.strftime("%Y-%m-%dT23:59:59Z"),
        "limit": min(1000, limit),
        "feed": "iex",
        "adjustment": "split",
        "sort": "desc",
    }
    try:
        response = await client.get(url, headers=ALPACA_HEADERS, params=params)
        if response.status_code != 200:
            return []
        bars = response.json().get("bars", [])
        bars.reverse()
        return [{
            "datetime": bar.get("t"),
            "o": bar.get("o"), "h": bar.get("h"), "l": bar.get("l"),
            "c": bar.get("c"), "v": bar.get("v", 0),
        } for bar in bars][-limit:]
    except Exception:
        return []


async def get_or_fetch_4h_bars(client, db, symbol, limit=240):
    doc = await db.stock_bars_4h.find_one({"ticker": symbol})
    if doc and doc.get("bars"):
        updated = doc.get("updated_at")
        if updated and (datetime.utcnow() - updated).total_seconds() < 4 * 3600:
            return _bars_4h_to_df(doc["bars"])
    bars = await fetch_4h_bars_from_api(client, symbol, limit)
    if bars:
        await db.stock_bars_4h.update_one(
            {"ticker": symbol},
            {"$set": {"ticker": symbol, "timeframe": "4H", "bars": bars, "updated_at": datetime.utcnow()}},
            upsert=True,
        )
        return _bars_4h_to_df(bars)
    return _bars_4h_to_df(doc.get("bars", [])) if doc else None


def _bars_4h_to_df(bars):
    if not bars or len(bars) < 20:
        return None
    frame = pd.DataFrame(bars).rename(columns={"datetime": "datetime", "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True).dt.tz_convert(None)
    return frame.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).sort_values("datetime").reset_index(drop=True)


async def get_or_fetch_4h_bars_batch(client, db, symbols, max_concurrent=6):
    semaphore = asyncio.Semaphore(max_concurrent)
    results = {}
    async def fetch_one(symbol):
        async with semaphore:
            results[symbol] = await get_or_fetch_4h_bars(client, db, symbol)
    await asyncio.gather(*[fetch_one(symbol) for symbol in symbols])
    return results


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

def analyze_stock(ticker, df, sector_code, sector_scores, prev_poc_position=None, df_4h=None):
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
    high_52w = round(float(high.tail(252).max()), 2); low_52w = round(float(low.tail(252).min()), 2)
    pct_from_high = round(((price - high_52w) / high_52w) * 100, 2) if high_52w > 0 else 0
    pct_from_low = round(((price - low_52w) / low_52w) * 100, 2) if low_52w > 0 else 0
    range_position = round(((price - low_52w) / (high_52w - low_52w)) * 100, 1) if (high_52w - low_52w) > 0 else 50
    poc = poc_result[0]; va_high = poc_result[1]; va_low = poc_result[2]; vp_distribution = poc_result[3]

    # 🆕 POC SHIFT (metodo Rea) — prev_poc_position letto dal DB (persistente)
    poc_shift = _detect_poc_shift(ticker, price, poc, prev_position=prev_poc_position)
    max_strategy = analyze_max_strategy(df, bars_4h=df_4h)

    patterns = detect_candlestick_patterns(df)
    fvgs = detect_fvg(df); wyckoff = detect_wyckoff_phase(df)
    accumulation = calc_accumulation_score(df, poc, va_low, va_high)
    mtf = calc_weekly_trend(df)
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
        "accumulation": accumulation, "mtf": mtf, "price_history": price_history, "pattern_bonus": pattern_bonus,
        "poc_shift": poc_shift,
        "max_strategy": max_strategy,
        "high_52w": high_52w, "low_52w": low_52w, "pct_from_high": pct_from_high,
        "pct_from_low": pct_from_low, "range_position": range_position,
        "data_status": "FRESH", "data_eligible": True,
        "last_bar_date": df["datetime"].iloc[-1].strftime("%Y-%m-%d"),
        "calendar_days_old": (datetime.utcnow().date() - df["datetime"].iloc[-1].date()).days,
        "updated_at": datetime.utcnow(),
    }


async def save_max_strategy_validation_snapshot(db, asset_doc, df):
    max_strategy = (asset_doc or {}).get("max_strategy") or {}
    plan = max_strategy.get("entry_plan") or {}
    status = plan.get("status")
    if status not in ("ARMED", "TRIGGERED", "CONFIRMED_4H", "CONFIRMED_DAILY", "WAIT_RETEST"):
        return None
    ticker = asset_doc.get("ticker")
    if not ticker or df is None or len(df) == 0:
        return None
    signal_date = df["datetime"].iloc[-1].strftime("%Y-%m-%d")
    trigger_price = plan.get("trigger_price")
    setup_key = f"{ticker}:{signal_date}:{status}:{trigger_price}"
    snapshot = {
        "setup_key": setup_key,
        "ticker": ticker,
        "signal_date": signal_date,
        "strategy_version": max_strategy.get("version"),
        "status": status,
        "order_action": plan.get("order_action"),
        "execution_mode": plan.get("execution_mode"),
        "signal_price": asset_doc.get("price"),
        "trigger_price": trigger_price,
        "maximum_entry_price": plan.get("maximum_entry_price"),
        "invalidation_price": plan.get("invalidation_price"),
        "weekly_plan_qualified": plan.get("weekly_plan_qualified"),
        "blocking_phase": plan.get("blocking_phase"),
        "market_phase": (max_strategy.get("market_phase") or {}).get("phase"),
        "strategy_type": max_strategy.get("strategy_type"),
        "max_score": max_strategy.get("max_score"),
        "weekly_context": max_strategy.get("weekly_context"),
        "daily_confirmation": max_strategy.get("daily_confirmation"),
        "execution_4h": max_strategy.get("execution_4h"),
        "structural_base": max_strategy.get("structural_base"),
        "active_base": max_strategy.get("active_base"),
        "outcomes": {
            "bars_observed": 0,
            "mfe_pct": 0.0,
            "mae_pct": 0.0,
            "trigger_reached": False,
            "maximum_entry_exceeded": False,
            "invalidation_reached": False,
            "return_5d_pct": None,
            "return_10d_pct": None,
            "return_20d_pct": None,
        },
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    await db.max_strategy_signals.update_one(
        {"setup_key": setup_key},
        {"$setOnInsert": snapshot, "$set": {"last_seen_at": datetime.utcnow(), "latest_status": status}},
        upsert=True,
    )
    return setup_key


async def update_max_strategy_validation_outcomes(db, ticker, df):
    if df is None or len(df) == 0:
        return 0
    signals = await db.max_strategy_signals.find({"ticker": ticker}).to_list(200)
    updated = 0
    for signal in signals:
        signal_date = signal.get("signal_date")
        signal_price = signal.get("signal_price")
        if not signal_date or not signal_price or signal_price <= 0:
            continue
        future = df[df["datetime"] > pd.to_datetime(signal_date)].copy()
        if len(future) == 0:
            continue
        bars_observed = len(future)
        highest = float(future["High"].max())
        lowest = float(future["Low"].min())
        mfe_pct = (highest - signal_price) / signal_price * 100
        mae_pct = (lowest - signal_price) / signal_price * 100
        trigger = signal.get("trigger_price")
        maximum_entry = signal.get("maximum_entry_price")
        invalidation = signal.get("invalidation_price")
        outcomes = {
            "bars_observed": bars_observed,
            "mfe_pct": round(mfe_pct, 2),
            "mae_pct": round(mae_pct, 2),
            "trigger_reached": bool(trigger and highest >= trigger),
            "maximum_entry_exceeded": bool(maximum_entry and highest > maximum_entry),
            "invalidation_reached": bool(invalidation and lowest <= invalidation),
            "return_5d_pct": round((float(future["Close"].iloc[4]) - signal_price) / signal_price * 100, 2) if bars_observed >= 5 else None,
            "return_10d_pct": round((float(future["Close"].iloc[9]) - signal_price) / signal_price * 100, 2) if bars_observed >= 10 else None,
            "return_20d_pct": round((float(future["Close"].iloc[19]) - signal_price) / signal_price * 100, 2) if bars_observed >= 20 else None,
        }
        await db.max_strategy_signals.update_one(
            {"_id": signal["_id"]},
            {"$set": {"outcomes": outcomes, "updated_at": datetime.utcnow()}},
        )
        updated += 1
    return updated


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

                # 🆕 SECTOR ROTATION (metodo Rea: Ann3M vs Ann6M + Compressione 20d)
                n = len(close)
                # Rendimenti annualizzati 3M (~63d) e 6M (~126d)
                if n >= 63:
                    r3 = (float(close.iloc[-1]) / float(close.iloc[-63])) - 1
                    ann_3m = round(((1 + r3) ** (252/63) - 1) * 100, 2)
                else:
                    ann_3m = 0
                if n >= 126:
                    r6 = (float(close.iloc[-1]) / float(close.iloc[-126])) - 1
                    ann_6m = round(((1 + r6) ** (252/126) - 1) * 100, 2)
                else:
                    ann_6m = ann_3m
                # Accelerazione momentum: >0 = soldi che rientrano (rotazione IN)
                momentum_accel = round(ann_3m - ann_6m, 2)
                # Compressione 20d: banda stretta = molla carica (pronto a esplodere)
                last20 = close.iloc[-20:]
                mean20 = float(last20.mean())
                compression_20d = round((float(last20.max()) - float(last20.min())) / mean20, 3) if mean20 > 0 else 0.5

                # 200SMA breadth-proxy dell'ETF (sopra/sotto la sua 200)
                if n >= 200:
                    sma200 = float(close.iloc[-200:].mean())
                    above_200 = float(close.iloc[-1]) > sma200
                else:
                    above_200 = None

                # ROTATION SCORE (0-100): momentum accel + compressione + forza relativa
                rot = 50
                rot += min(25, max(-25, momentum_accel * 0.5))   # accelerazione pesa
                rot += 10 if compression_20d < 0.05 else (5 if compression_20d < 0.08 else 0)  # molla carica
                rot += min(10, max(-10, strength))                # forza vs SPY
                rotation_score = round(max(0, min(100, rot)), 1)

                # Classificazione quadrante Rea
                if momentum_accel > 5 and compression_20d < 0.07:
                    rotation_signal = "EXPLOSIVE"      # 🎯 rotazione IN + compresso
                elif momentum_accel > 5:
                    rotation_signal = "ROTATING_IN"    # 🚀 soldi rientrano
                elif momentum_accel < -8:
                    rotation_signal = "ROTATING_OUT"   # 📉 soldi escono
                else:
                    rotation_signal = "NEUTRAL"
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
                    "history": history,
                    # 🆕 Sector Rotation (metodo Rea)
                    "ann_3m": ann_3m, "ann_6m": ann_6m,
                    "momentum_accel": momentum_accel, "compression_20d": compression_20d,
                    "rotation_score": rotation_score, "rotation_signal": rotation_signal,
                    "above_200sma": above_200,
                    "updated_at": datetime.utcnow()}
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
    legacy_cleanup = await cleanup_legacy_max_strategy_data(db)
    if any(legacy_cleanup.values()):
        print(f"  Legacy cleanup: {legacy_cleanup}")

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
            bars_4h_map = await get_or_fetch_4h_bars_batch(client, db, batch, max_concurrent=6)

            success = 0; skipped = 0; stale = 0
            for ticker in batch:
                df = bars_map.get(ticker)
                sector_code = ticker_to_sector.get(ticker, "UNKNOWN")
                freshness = _data_freshness(df)
                if not freshness["eligible"]:
                    await mark_stale_asset(db, ticker, sector_code, freshness)
                    stale += 1
                    skipped += 1
                    print(f"      STALE {ticker}: last={freshness['last_bar_date']} age={freshness['calendar_days_old']}d")
                    continue
                # 🆕 Leggi la posizione POC precedente dal DB (per lo shift persistente)
                prev_doc = await db.assets.find_one({"ticker": ticker}, {"poc_shift": 1})
                prev_pos = (prev_doc or {}).get("poc_shift", {}).get("poc_position")
                asset_doc = analyze_stock(ticker, df, sector_code, sector_scores, prev_poc_position=prev_pos, df_4h=bars_4h_map.get(ticker))
                if asset_doc:
                    await db.assets.update_one({"ticker": ticker}, {"$set": asset_doc}, upsert=True)
                    await save_max_strategy_validation_snapshot(db, asset_doc, df)
                    await update_max_strategy_validation_outcomes(db, ticker, df)
                    results.append(asset_doc)
                    success += 1
                else:
                    skipped += 1

            batch_time = round(time.time() - t_batch, 1)
            print(f"    -> {success} OK, {skipped} skipped ({stale} stale) in {batch_time}s")

    elapsed = round(time.time() - t_start, 1)
    print(f"\n{'=' * 50}")
    print(f"STOCKS DONE: {len(results)}/{total} in {elapsed}s")
    if results:
        print(f"  Average: {elapsed/len(results):.2f}s per stock")
    first_run_count = await db.stock_bars.count_documents({})
    print(f"  Bars in MongoDB: {first_run_count} stocks cached")
    print("=" * 50)
    return results
