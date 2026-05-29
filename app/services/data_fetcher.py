import httpx
import pandas as pd
import numpy as np
from datetime import datetime
from app.db.mongodb import get_db
import traceback

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
}

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

# ============================================
# YAHOO FINANCE DIRECT API (bypassa yfinance)
# ============================================

_crumb = None
_cookie_jar = None

async def _init_yahoo_session(client):
    """Ottieni cookie e crumb da Yahoo Finance"""
    global _crumb, _cookie_jar
    try:
        r1 = await client.get("https://fc.yahoo.com", follow_redirects=True)
        r2 = await client.get(
            "https://query2.finance.yahoo.com/v1/test/getcrumb",
            follow_redirects=True
        )
        if r2.status_code == 200 and r2.text:
            _crumb = r2.text
            _cookie_jar = dict(client.cookies)
            print(f"Yahoo session OK - crumb: {_crumb[:10]}...")
            return True
        else:
            print(f"Crumb failed: status={r2.status_code}, text={r2.text[:100]}")
    except Exception as e:
        print(f"Yahoo session error: {e}")
    return False

async def fetch_chart(ticker, period="3mo", interval="1d"):
    """Scarica OHLCV direttamente da Yahoo Finance chart API"""
    global _crumb, _cookie_jar

    async with httpx.AsyncClient(
        headers=HEADERS,
        cookies=_cookie_jar,
        follow_redirects=True,
        timeout=30
    ) as client:
        if not _crumb:
            ok = await _init_yahoo_session(client)
            if not ok:
                print(f"  SKIP {ticker}: no Yahoo session")
                return None

        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {
            "range": period,
            "interval": interval,
            "crumb": _crumb,
            "includePrePost": "false",
            "events": "",
        }

        try:
            r = await client.get(url, params=params)
            if r.status_code == 401:
                print(f"  {ticker}: 401 - refreshing session...")
                _crumb = None
                _cookie_jar = None
                ok = await _init_yahoo_session(client)
                if ok:
                    r = await client.get(url, params=params)
                else:
                    return None

            if r.status_code != 200:
                print(f"  {ticker}: HTTP {r.status_code}")
                return None

            data = r.json()
            result = data.get("chart", {}).get("result")
            if not result:
                print(f"  {ticker}: no chart result")
                return None

            quote = result[0]
            timestamps = quote.get("timestamp", [])
            ohlcv = quote.get("indicators", {}).get("quote", [{}])[0]

            if not timestamps or not ohlcv.get("close"):
                print(f"  {ticker}: empty data")
                return None

            df = pd.DataFrame({
                "Open": ohlcv.get("open", []),
                "High": ohlcv.get("high", []),
                "Low": ohlcv.get("low", []),
                "Close": ohlcv.get("close", []),
                "Volume": ohlcv.get("volume", []),
            }, index=pd.to_datetime(timestamps, unit="s"))

            df = df.dropna()
            return df
        except Exception as e:
            print(f"  {ticker} fetch error: {e}")
            return None

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

def calc_volume_profile(highs, lows, volumes, bins=50):
    try:
        price_min = float(lows.min())
        price_max = float(highs.max())
        if price_max <= price_min:
            return None, None, None

        bin_edges = np.linspace(price_min, price_max, bins + 1)
        volume_per_level = np.zeros(bins)

        for idx in range(len(highs)):
            row_low = float(lows.iloc[idx])
            row_high = float(highs.iloc[idx])
            row_vol = float(volumes.iloc[idx])
            spread = max(1, int((row_high - row_low) / ((price_max - price_min) / bins)))
            for i in range(bins):
                if row_low <= bin_edges[i + 1] and row_high >= bin_edgesvolume_per_level[i] += row_vol / spread

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
        return poc_price, va_high, va_low
    except Exception as e:
        print(f"    VP error: {e}")
        return None, None, None

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

    return min(score, 100)

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
    print("=" * 50)
    print("STARTING SECTOR REFRESH")
    print("=" * 50)

    spy_df = await fetch_chart("SPY")
    spy_return = 0
    if spy_df is not None and len(spy_df) >= 20:
        spy_return = ((float(spy_df["Close"].iloc[-1]) / float(spy_df["Close"].iloc[-20])) - 1) * 100
        print(f"SPY 20d return: {spy_return:.2f}%")
    else:
        print("WARNING: SPY data not available")

    results = []
    for etf, name in SECTOR_MAP.items():
        try:
            print(f"\n  Fetching {etf} ({name})...")
            df = await fetch_chart(etf)
            if df is None or len(df) < 20:
                print(f"  SKIP {etf}: insufficient data")
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
                "code": etf,
                "name": name,
                "etf_ticker": etf,
                "price": round(price, 2),
                "return_20d": round(float(ret_20d), 2),
                "strength_score": strength,
                "trend_score": trend,
                "volume_score": round(rel_vol * 30, 2),
                "rsi": rsi,
                "composite_score": composite,
                "updated_at": datetime.utcnow(),
            }

            await db.sectors.update_one(
                {"code": etf}, {"$set": sector_doc}, upsert=True
            )
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
    print("=" * 50)
    print("STARTING STOCKS REFRESH")
    print("=" * 50)

    sector_scores = {}
    async for s in db.sectors.find():
        sector_scores[s["code"]] = s.get("composite_score", 50)
    print(f"Sector scores: {sector_scores}")

    results = []
    for sector_code, tickers in SECTOR_STOCKS.items():
        print(f"\n--- {sector_code} ---")
        for ticker in tickers:
            try:
                df = await fetch_chart(ticker)
                if df is None or len(df) < 20:
                    print(f"    SKIP {ticker}: no data")
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

                poc, va_high, va_low = calc_volume_profile(high, low, volume)

                ind_data = {
                    "price": price,
                    "rsi": rsi,
                    "macd_histogram": macd["histogram"],
                    "ema10": ema10,
                    "ema20": ema20,
                    "ema50": ema50,
                    "relative_volume": rel_vol,
                    "poc_price": poc,
                    "va_high": va_high,
                    "change_pct": change_pct,
                    "sector_strength": sector_scores.get(sector_code, 50),
                }

                setup_score = calc_setup_score(ind_data)
                setup_type = detect_setup_type(ind_data)

                asset_doc = {
                    "ticker": ticker,
                    "name": ticker,
                    "sector_code": sector_code,
                    "price": round(price, 2),
                    "change_pct": change_pct,
                    "avg_volume": round(avg_vol, 0),
                    "relative_volume": rel_vol,
                    "rsi": rsi,
                    "macd": macd,
                    "ema10": ema10,
                    "ema20": ema20,
                    "ema50": ema50,
                    "momentum_score": rsi,
                    "volume_score": round(rel_vol * 30, 2),
                    "poc_price": poc,
                    "value_area_high": va_high,
                    "value_area_low": va_low,
                    "setup_score": setup_score,
                    "setup_type": setup_type,
                    "updated_at": datetime.utcnow(),
                }

                await db.assets.update_one(
                    {"ticker": ticker}, {"$set": asset_doc}, upsert=True
                )
                results.append(asset_doc)
                print(f"    OK {ticker}: ${price:.2f} score={setup_score} [{setup_type}]")
            except Exception as e:
                print(f"    ERROR {ticker}: {e}")

    print(f"\nSTOCKS DONE: {len(results)}/110")
    return results
