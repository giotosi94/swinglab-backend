import math
import numpy as np
import pandas as pd


def _safe_float(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _atr_series(df, period=14):
    prev_close = df["Close"].shift(1)
    true_range = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.rolling(period, min_periods=period).mean()


def _volume_profile(df, lookback, bins=48):
    window = df.tail(min(lookback, len(df))).copy()
    if len(window) < 10:
        return None
    low = _safe_float(window["Low"].min())
    high = _safe_float(window["High"].max())
    if high <= low:
        return None
    edges = np.linspace(low, high, bins + 1)
    volumes = np.zeros(bins, dtype=float)
    for row in window.itertuples(index=False):
        bar_low = _safe_float(getattr(row, "Low"))
        bar_high = _safe_float(getattr(row, "High"))
        bar_volume = max(0.0, _safe_float(getattr(row, "Volume")))
        if bar_high < bar_low or bar_volume <= 0:
            continue
        start = max(0, min(bins - 1, int(np.searchsorted(edges, bar_low, side="right") - 1)))
        end = max(0, min(bins - 1, int(np.searchsorted(edges, bar_high, side="left"))))
        touched = max(1, end - start + 1)
        distributed = bar_volume / touched
        volumes[start:end + 1] += distributed
    if volumes.sum() <= 0:
        return None
    centers = (edges[:-1] + edges[1:]) / 2
    poc_index = int(np.argmax(volumes))
    poc = float(centers[poc_index])
    ranked = np.argsort(volumes)[::-1]
    selected = []
    cumulative = 0.0
    target = float(volumes.sum()) * 0.70
    for index in ranked:
        selected.append(int(index))
        cumulative += float(volumes[index])
        if cumulative >= target:
            break
    val = float(centers[min(selected)])
    vah = float(centers[max(selected)])
    concentration = float(volumes[poc_index] / volumes.sum() * 100)
    return {
        "lookback": int(lookback),
        "poc": round(poc, 2),
        "vah": round(vah, 2),
        "val": round(val, 2),
        "concentration_pct": round(concentration, 2),
        "window_start": window["datetime"].iloc[0].strftime("%Y-%m-%d"),
        "window_end": window["datetime"].iloc[-1].strftime("%Y-%m-%d"),
    }


def _profile_migration(df, lookback, shift=5):
    current = _volume_profile(df, lookback)
    previous = _volume_profile(df.iloc[:-shift], lookback) if len(df) > lookback + shift else None
    if not current:
        return None
    migration_pct = 0.0
    direction = "STABLE"
    if previous and previous.get("poc", 0) > 0:
        migration_pct = (current["poc"] - previous["poc"]) / previous["poc"] * 100
        if migration_pct >= 0.75:
            direction = "UP"
        elif migration_pct <= -0.75:
            direction = "DOWN"
    current["migration_5d_pct"] = round(migration_pct, 2)
    current["migration"] = direction
    return current


def _swing_points(values, left=3, right=3, mode="low"):
    points = []
    for index in range(left, len(values) - right):
        window = values[index - left:index + right + 1]
        value = values[index]
        if mode == "low" and value == np.min(window):
            points.append(index)
        elif mode == "high" and value == np.max(window):
            points.append(index)
    return points


def _master_poc(df, atr):
    if len(df) < 120:
        return None
    working = df.tail(min(504, len(df))).reset_index(drop=True)
    lows = working["Low"].to_numpy(dtype=float)
    highs = working["High"].to_numpy(dtype=float)
    swing_lows = _swing_points(lows, 4, 4, "low")
    swing_highs = _swing_points(highs, 4, 4, "high")
    candidates = []
    for low_index in swing_lows:
        future_highs = [index for index in swing_highs if index >= low_index + 10]
        if not future_highs:
            continue
        high_index = max(future_highs, key=lambda index: highs[index])
        move_pct = (highs[high_index] - lows[low_index]) / lows[low_index] * 100 if lows[low_index] > 0 else 0
        if move_pct >= 12 and high_index - low_index >= 10:
            candidates.append((high_index, move_pct, low_index, high_index))
    if not candidates:
        return None
    _, move_pct, start, end = max(candidates, key=lambda item: (item[0], item[1]))
    segment = working.iloc[start:end + 1].copy()
    profile = _volume_profile(segment, len(segment), bins=64)
    if not profile:
        return None
    poc = profile["poc"]
    zone = max(atr * 0.25, poc * 0.003)
    after = working.iloc[end + 1:]
    touches = 0
    for row in after.itertuples(index=False):
        if _safe_float(row.Low) <= poc + zone and _safe_float(row.High) >= poc - zone:
            touches += 1
    status = "VIRGIN" if touches == 0 else "TESTED_ONCE" if touches == 1 else "TESTED_MULTIPLE"
    price = _safe_float(working["Close"].iloc[-1])
    if price < poc - zone:
        status = "BROKEN"
    elif touches > 0 and price > poc + zone:
        status = "RECLAIMED"
    profile.update({
        "type": "IMPULSE_LOW_TO_HIGH",
        "impulse_start": working["datetime"].iloc[start].strftime("%Y-%m-%d"),
        "impulse_end": working["datetime"].iloc[end].strftime("%Y-%m-%d"),
        "impulse_move_pct": round(move_pct, 2),
        "zone_low": round(poc - zone, 2),
        "zone_high": round(poc + zone, 2),
        "touches_after_formation": touches,
        "status": status,
    })
    return profile


def _compression(df, atr_series, poc60):
    price = _safe_float(df["Close"].iloc[-1])
    atr20 = _safe_float(atr_series.tail(20).mean())
    atr60 = _safe_float(atr_series.tail(60).mean())
    range10 = (_safe_float(df["High"].tail(10).max()) - _safe_float(df["Low"].tail(10).min())) / price * 100 if price > 0 else 0
    range60 = (_safe_float(df["High"].tail(60).max()) - _safe_float(df["Low"].tail(60).min())) / price * 100 if price > 0 else 0
    rolling_mean = df["Close"].rolling(20).mean()
    rolling_std = df["Close"].rolling(20).std(ddof=0)
    bandwidth = _safe_float((rolling_std.iloc[-1] * 4 / rolling_mean.iloc[-1]) * 100) if _safe_float(rolling_mean.iloc[-1]) > 0 else 0
    history_bandwidth = (rolling_std * 4 / rolling_mean.replace(0, np.nan) * 100).dropna().tail(120)
    percentile = float((history_bandwidth <= bandwidth).mean() * 100) if len(history_bandwidth) else 50.0
    volume10 = _safe_float(df["Volume"].tail(10).mean())
    volume_prev10 = _safe_float(df["Volume"].iloc[-20:-10].mean())
    volume_ratio = volume10 / volume_prev10 if volume_prev10 > 0 else 1.0
    distance_atr = abs(price - poc60) / atr20 if poc60 and atr20 > 0 else 99.0
    factors = {
        "atr_contracting": atr20 > 0 and atr60 > 0 and atr20 <= atr60 * 0.85,
        "range_contracting": range60 > 0 and range10 <= range60 * 0.45,
        "bandwidth_low": percentile <= 30,
        "close_near_poc60": distance_atr <= 1.0,
        "volume_contracting": volume_ratio <= 0.90,
    }
    score = sum(20 for passed in factors.values() if passed)
    state = "COMPRESSED" if score >= 60 else "BUILDING" if score >= 40 else "NONE"
    return {
        "state": state,
        "score": score,
        "atr20_pct": round(atr20 / price * 100, 2) if price > 0 else 0,
        "atr20_vs_atr60": round(atr20 / atr60, 3) if atr60 > 0 else 0,
        "range10_pct": round(range10, 2),
        "range60_pct": round(range60, 2),
        "bollinger_bandwidth_pct": round(bandwidth, 2),
        "bandwidth_percentile": round(percentile, 1),
        "volume10_vs_previous10": round(volume_ratio, 3),
        "distance_from_poc60_atr": round(distance_atr, 2),
        "factors": factors,
    }


def _bottom_state(df, profiles, compression, atr):
    price = _safe_float(df["Close"].iloc[-1])
    high252 = _safe_float(df["High"].tail(252).max())
    low252 = _safe_float(df["Low"].tail(252).min())
    drawdown = (price - high252) / high252 * 100 if high252 > 0 else 0
    range_position = (price - low252) / (high252 - low252) * 100 if high252 > low252 else 50
    recent_low = _safe_float(df["Low"].tail(20).min())
    prior_low = _safe_float(df["Low"].iloc[-60:-20].min()) if len(df) >= 60 else recent_low
    lower_low_pct = (recent_low - prior_low) / prior_low * 100 if prior_low > 0 else 0
    volume_down = df.loc[df["Close"] < df["Open"], "Volume"].tail(10).mean()
    volume_up = df.loc[df["Close"] >= df["Open"], "Volume"].tail(10).mean()
    accumulation_ratio = _safe_float(volume_up / volume_down, 1.0) if _safe_float(volume_down) > 0 else 1.0
    poc20 = profiles.get("poc20") or {}
    poc60 = profiles.get("poc60") or {}
    factors = {
        "deep_location": drawdown <= -30 or range_position <= 30,
        "no_material_new_low": lower_low_pct >= -2.0,
        "poc20_not_falling": poc20.get("migration") in ("UP", "STABLE"),
        "poc60_not_falling": poc60.get("migration") in ("UP", "STABLE"),
        "compression_present": compression.get("score", 0) >= 40,
        "up_volume_dominant": accumulation_ratio >= 1.05,
    }
    confirmation_count = sum(1 for key, value in factors.items() if key != "deep_location" and value)
    if factors["deep_location"] and confirmation_count >= 4:
        state = "ACCUMULATING"
    elif factors["deep_location"] and confirmation_count >= 2:
        state = "BOTTOM_BUILDING"
    elif drawdown <= -20:
        state = "MARKDOWN"
    else:
        state = "NEUTRAL"
    depth = "CAPITULATION" if drawdown <= -70 else "DEEP" if drawdown <= -50 else "DEPRESSED" if drawdown <= -30 else "NORMAL"
    invalidation = recent_low - max(atr * 0.25, price * 0.003)
    return {
        "state": state,
        "depth": depth,
        "drawdown_52w_pct": round(drawdown, 2),
        "range_position_52w": round(range_position, 1),
        "distance_from_52w_low_pct": round((price - low252) / low252 * 100, 2) if low252 > 0 else 0,
        "recent_low": round(recent_low, 2),
        "prior_low": round(prior_low, 2),
        "recent_low_change_pct": round(lower_low_pct, 2),
        "up_down_volume_ratio": round(accumulation_ratio, 2),
        "invalidation_price": round(max(0, invalidation), 2),
        "factors": factors,
    }


def analyze_max_strategy(df):
    if df is None or len(df) < 140:
        return {"status": "INSUFFICIENT_DATA", "bars": 0 if df is None else len(df)}
    data = df.copy().sort_values("datetime").reset_index(drop=True)
    price = _safe_float(data["Close"].iloc[-1])
    atr_series = _atr_series(data, 14)
    atr = _safe_float(atr_series.iloc[-1], price * 0.02)
    profiles = {
        "poc20": _profile_migration(data, 20),
        "poc60": _profile_migration(data, 60),
        "poc120": _profile_migration(data, 120),
    }
    for profile in profiles.values():
        if profile and profile.get("poc", 0) > 0:
            signed = (price - profile["poc"]) / profile["poc"] * 100
            profile["signed_distance_pct"] = round(signed, 2)
            profile["price_position"] = "ABOVE" if signed > 0.25 else "BELOW" if signed < -0.25 else "AT_POC"
    poc60 = (profiles.get("poc60") or {}).get("poc")
    compression = _compression(data, atr_series, poc60)
    bottom = _bottom_state(data, profiles, compression, atr)
    master = _master_poc(data, atr)
    triggers = []
    if profiles.get("poc60") and profiles["poc60"].get("price_position") == "ABOVE" and profiles["poc60"].get("migration") in ("UP", "STABLE"):
        triggers.append("POC60_RECLAIM")
    if profiles.get("poc20") and profiles["poc20"].get("migration") == "UP":
        triggers.append("POC20_MIGRATION_UP")
    if master and master.get("status") in ("VIRGIN", "TESTED_ONCE") and master.get("zone_low", 0) <= price <= master.get("zone_high", 0):
        triggers.append("MASTER_POC_CONTACT")
    if compression.get("state") == "COMPRESSED":
        triggers.append("COMPRESSION_READY")
    score = 0
    score += 25 if bottom.get("depth") == "CAPITULATION" else 18 if bottom.get("depth") == "DEEP" else 10 if bottom.get("depth") == "DEPRESSED" else 0
    score += 25 if bottom.get("state") == "ACCUMULATING" else 15 if bottom.get("state") == "BOTTOM_BUILDING" else 0
    score += compression.get("score", 0) * 0.20
    score += 10 if "POC60_RECLAIM" in triggers else 0
    score += 10 if "POC20_MIGRATION_UP" in triggers else 0
    score += 10 if master and master.get("status") == "VIRGIN" else 5 if master and master.get("status") == "TESTED_ONCE" else 0
    watch_ready = bottom.get("state") in ("BOTTOM_BUILDING", "ACCUMULATING") and score >= 45
    return {
        "status": "OK",
        "version": "max_structure_v1",
        "bars_analyzed": len(data),
        "price": round(price, 2),
        "atr14": round(atr, 4),
        "atr14_pct": round(atr / price * 100, 2) if price > 0 else 0,
        "profiles": profiles,
        "master_poc": master,
        "compression": compression,
        "bottom": bottom,
        "triggers": triggers,
        "max_score": round(min(100, score), 1),
        "watch_ready": watch_ready,
        "live_entry_enabled": False,
    }
