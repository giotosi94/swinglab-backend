import httpx
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime
from app.db.mongodb import get_db
from app.config import settings
from app.services.data_fetcher import (
    calc_rsi, calc_ema, calc_macd, calc_volume_profile,
    calc_setup_score, detect_setup_type,
    detect_candlestick_patterns, get_pattern_score_bonus
)

TD_BASE = "https://api.twelvedata.com"


async def search_and_analyze_stock(ticker):
    """Fetch and analyze a single stock on-demand"""
    if not settings.TWELVEDATA_API_KEY:
        return None

    async with httpx.AsyncClient(timeout=60) as client:
        url = f"{TD_BASE}/time_series"
        params = {
            "symbol": ticker,
            "interval": "1day",
            "outputsize": 70,
            "apikey": settings.TWELVEDATA_API_KEY,
        }
        try:
            r = await client.get(url, params=params)
            if r.status_code != 200:
                return None
            data = r.json()
            if "code" in data and data["code"] != 200:
                return None
            values = data.get("values", [])
            if not values or len(values) < 20:
                return None

            df = pd.DataFrame(values)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime").reset_index(drop=True)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.rename(columns={
                "open": "Open", "high": "High",
                "low": "Low", "close": "Close",
                "volume": "Volume"
            })
            df = df.dropna()

            if len(df) < 20:
                return None

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

            high_52w = round(float(high.max()), 2)
            low_52w = round(float(low.min()), 2)
            pct_from_high = round(((price - high_52w) / high_52w) * 100, 2) if high_52w > 0 else 0
            pct_from_low = round(((price - low_52w) / low_52w) * 100, 2) if low_52w > 0 else 0
            range_position = round(((price - low_52w) / (high_52w - low_52w)) * 100, 1) if (high_52w - low_52w) > 0 else 50

            patterns = detect_candlestick_patterns(df)
            pattern_bonus = get_pattern_score_bonus(patterns)
            patterns_list = [{"name": p["name"], "type": p["type"], "strength": p["strength"], "description": p["description"]} for p in patterns]

            ind_data = {
                "price": price, "rsi": rsi, "macd_histogram": macd["histogram"],
                "ema10": ema10, "ema20": ema20, "ema50": ema50,
                "relative_volume": rel_vol, "poc_price": poc, "va_high": va_high,
                "change_pct": change_pct, "sector_strength": 50,
                "pattern_bonus": pattern_bonus,
            }

            setup_score = calc_setup_score(ind_data)
            setup_type = detect_setup_type(ind_data)

            # Save to DB
            db = get_db()
            asset_doc = {
                "ticker": ticker, "name": ticker, "sector_code": "SEARCH",
                "price": round(price, 2), "change_pct": change_pct,
                "avg_volume": round(avg_vol, 0), "relative_volume": rel_vol,
                "rsi": rsi, "macd": macd,
                "ema10": ema10, "ema20": ema20, "ema50": ema50,
                "momentum_score": rsi, "volume_score": round(rel_vol * 30, 2),
                "poc_price": poc, "value_area_high": va_high, "value_area_low": va_low,
                "setup_score": setup_score, "setup_type": setup_type,
                "vp_distribution": vp_distribution,
                "candlestick_patterns": patterns_list,
                "pattern_bonus": pattern_bonus,
                "high_52w": high_52w, "low_52w": low_52w,
                "pct_from_high": pct_from_high, "pct_from_low": pct_from_low,
                "range_position": range_position,
                "updated_at": datetime.utcnow(),
                "source": "search",
            }

            await db.assets.update_one(
                {"ticker": ticker}, {"$set": asset_doc}, upsert=True
            )

            return asset_doc

        except Exception as e:
            print(f"Search error {ticker}: {e}")
            return None
