import yfinance as yf
import numpy as np
import requests
from datetime import datetime
from app.db.mongodb import get_db
import traceback

# Custom session per evitare blocchi Yahoo
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
})

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

def calc_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1] if not rsi.empty else 50
    return 50 if np.isnan(val) else val

def calc_ema(prices, period):
    ema = prices.ewm(span=period, adjust=False).mean()
    val = ema.iloc[-1] if not ema.empty else 0
    return 0 if np.isnan(val) else val

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
        if price_max == price_min:
            return None, None, None

        bin_edges = np.linspace(price_min, price_max, bins + 1)
        volume_per_level = np.zeros(bins)

        for idx in range(len(highs)):
            row_low = float(lows.iloc[idx])
            row_high = float(highs.iloc[idx])
            row_vol = float(volumes.iloc[idx])
            spread_bins = max(1, int((row_high - row_low) / ((price_max - price_min) / bins)))
            for i in range(bins):
                if row_low <= bin_edges[i + 1] and row_high >= bin_edges[i]:
                    volume_per_level[i] += row_vol / spread_bins

        poc_idx = int(np.argmax(volume_per_level))
        poc_price = round((bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2, 2)

        total_vol = volume_per_level.sum()
        target_vol = total_vol * 0.70
        sorted_indices = np.argsort(volume_per_level)[::-1]
        cumulative = 0
        va_indices = []
        for idx in sorted_indices:
            cumulative += volume_per_level[idx]
            va_indices.append(idx)
            if cumulative >= target_vol:
                break

        va_low = round(float(bin_edges[min(va_indices)]), 2)
        va_high = round(float(bin_edges[max(va_indices) + 1]), 2)
        return poc_price, va_high, va_low
    except Exception as e:
        print(f"    Volume profile error: {e}")
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
