import httpx
import pandas as pd
import numpy as np
from datetime import datetime
from app.db.mongodb import get_db
from app.config import settings
from app.services.data_fetcher import (
    calc_rsi, calc_ema, calc_macd, calc_volume_profile,
    calc_setup_score, detect_setup_type,
    detect_candlestick_patterns, get_pattern_score_bonus,
    fetch_bars, ALPACA_HEADERS
)


async def search_and_analyze_stock(ticker):
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            df = await fetch_bars(client, ticker)
            if df is None or len(df) < 20:
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

            # Price history for chart
            rs_s = pd.Series(dtype=float)
            ds = close.diff()
            gs = ds.where(ds > 0, 0).rolling(14).mean()
            ls = (-ds.where(ds < 0, 0)).rolling(14).mean()
            rs_s = 100 - (100 / (1 + gs / ls))
            e10s = close.ewm(span=10).mean()
            e20s = close.ewm(span=20).mean()
            e50s = close.ewm(span=50).mean()
            price_history = []
            start_idx = max(20, len(df) - 90)
            for idx in range(start_idx, len(df)):
                dr = float(rs_s.iloc[idx]) if not pd.isna(rs_s.iloc[idx]) else 50
                price_history.append({
                    "date": df["datetime"].iloc[idx].strftime("%Y-%m-%d") if "datetime" in df.columns else "d{}".format(idx),
                    "close": round(float(close.iloc[idx]), 2),
                    "high": round(float(high.iloc[idx]), 2),
                    "low": round(float(low.iloc[idx]), 2),
                    "volume": int(volume.iloc[idx]),
                    "rsi": round(dr, 1),
                    "ema10": round(float(e10s.iloc[idx]), 2),
                    "ema20": round(float(e20s.iloc[idx]), 2),
                    "ema50": round(float(e50s.iloc[idx]), 2),
                })

            ind_data = {
                "price": price, "rsi": rsi, "macd_histogram": macd["histogram"],
                "ema10": ema10, "ema20": ema20, "ema50": ema50,
                "relative_volume": rel_vol, "poc_price": poc, "va_high": va_high,
                "change_pct": change_pct, "sector_strength": 50,
                "pattern_bonus": pattern_bonus,
            }

            setup_score = calc_setup_score(ind_data)
            setup_type = detect_setup_type(ind_data)

            db = get_db()
            existing_asset = await db.assets.find_one({"ticker": ticker})
            asset_doc = {
                "ticker": ticker, "name": ticker,
                "sector_code": existing_asset.get("sector_code", "SEARCH") if existing_asset else "SEARCH",
                "price": round(price, 2), "change_pct": change_pct,
                "avg_volume": round(avg_vol, 0), "relative_volume": rel_vol,
                "rsi": rsi, "macd": macd,
                "ema10": ema10, "ema20": ema20, "ema50": ema50,
                "poc_price": poc, "value_area_high": va_high, "value_area_low": va_low,
                "setup_score": setup_score, "setup_type": setup_type,
                "vp_distribution": vp_distribution,
                "candlestick_patterns": patterns_list,
                "price_history": price_history,
                "pattern_bonus": pattern_bonus,
                "high_52w": high_52w, "low_52w": low_52w,
                "pct_from_high": pct_from_high, "pct_from_low": pct_from_low,
                "range_position": range_position,
                "updated_at": datetime.utcnow(),
                "source": "search",
            }

            # Preserve LLM analysis if exists
            existing = await db.assets.find_one({"ticker": ticker}, {"llm_analysis": 1, "llm_analysis_at": 1})
            if existing and existing.get("llm_analysis"):
                asset_doc["llm_analysis"] = existing["llm_analysis"]
                asset_doc["llm_analysis_at"] = existing.get("llm_analysis_at", "")

            # Opzione A: salva SOLO se il ticker è già nell'universo reale.
            # I ticker cercati "usa e getta" NON inquinano più db.assets.
            if existing_asset:
                asset_doc.pop("sector_code", None)  # non sovrascrivere il settore reale
                await db.assets.update_one(
                    {"ticker": ticker}, {"$set": asset_doc}, upsert=False
                )
            return asset_doc

        except Exception as e:
            print("Search error {}: {}".format(ticker, e))
            return None
