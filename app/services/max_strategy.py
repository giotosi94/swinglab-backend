import math
import numpy as np
import pandas as pd


def _safe_float(value, default=0.0):
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _atr_series(df, period=14):
    previous = df["Close"].shift(1)
    true_range = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - previous).abs(),
        (df["Low"] - previous).abs(),
    ], axis=1).max(axis=1)
    return true_range.rolling(period, min_periods=period).mean()


def _volume_profile(df, bins=48):
    if df is None or len(df) < 8:
        return None
    window = df.copy().reset_index(drop=True)
    low = _safe_float(window["Low"].min())
    high = _safe_float(window["High"].max())
    if high <= low:
        return None
    edges = np.linspace(low, high, bins + 1)
    volumes = np.zeros(bins, dtype=float)
    for row in window.itertuples(index=False):
        bar_low = _safe_float(row.Low)
        bar_high = _safe_float(row.High)
        bar_volume = max(0.0, _safe_float(row.Volume))
        if bar_high < bar_low or bar_volume <= 0:
            continue
        start = max(0, min(bins - 1, int(np.searchsorted(edges, bar_low, side="right") - 1)))
        end = max(0, min(bins - 1, int(np.searchsorted(edges, bar_high, side="left"))))
        volumes[start:end + 1] += bar_volume / max(1, end - start + 1)
    total = float(volumes.sum())
    if total <= 0:
        return None
    centers = (edges[:-1] + edges[1:]) / 2
    poc_index = int(np.argmax(volumes))
    ranked = np.argsort(volumes)[::-1]
    selected = []
    cumulative = 0.0
    for index in ranked:
        selected.append(int(index))
        cumulative += float(volumes[index])
        if cumulative >= total * 0.70:
            break
    return {
        "poc": round(float(centers[poc_index]), 2),
        "vah": round(float(centers[max(selected)]), 2),
        "val": round(float(centers[min(selected)]), 2),
        "concentration_pct": round(float(volumes[poc_index] / total * 100), 2),
        "bars": len(window),
        "window_start": window["datetime"].iloc[0].strftime("%Y-%m-%d"),
        "window_end": window["datetime"].iloc[-1].strftime("%Y-%m-%d"),
    }


def _rolling_profile(df, lookback, shift=5):
    current = _volume_profile(df.tail(lookback), bins=48)
    previous = _volume_profile(df.iloc[:-shift].tail(lookback), bins=48) if len(df) >= lookback + shift else None
    if not current:
        return None
    migration = 0.0
    if previous and previous.get("poc", 0) > 0:
        migration = (current["poc"] - previous["poc"]) / previous["poc"] * 100
    current["lookback"] = lookback
    current["migration_5d_pct"] = round(migration, 2)
    current["migration"] = "UP" if migration >= 0.75 else "DOWN" if migration <= -0.75 else "STABLE"
    return current


def _swing_points(values, left=3, right=3, mode="low"):
    points = []
    for index in range(left, len(values) - right):
        section = values[index - left:index + right + 1]
        if mode == "low" and values[index] == np.min(section):
            points.append(index)
        if mode == "high" and values[index] == np.max(section):
            points.append(index)
    return points


def _data_quality(df):
    anomalies = []
    clean = df.copy().sort_values("datetime").drop_duplicates("datetime", keep="last").reset_index(drop=True)
    invalid_ohlc = clean[
        (clean["Low"] <= 0)
        | (clean["High"] < clean["Low"])
        | (clean["Open"] < clean["Low"])
        | (clean["Open"] > clean["High"])
        | (clean["Close"] < clean["Low"])
        | (clean["Close"] > clean["High"])
        | (clean["Volume"] < 0)
    ]
    if len(invalid_ohlc):
        anomalies.append({"type": "INVALID_OHLC", "count": len(invalid_ohlc)})
    returns = clean["Close"].pct_change()
    jumps = returns[returns.abs() >= 0.35]
    split_suspects = []
    for index, change in jumps.items():
        if index <= 0:
            continue
        ratio = clean["Close"].iloc[index] / clean["Close"].iloc[index - 1]
        common_split = min(abs(ratio - level) for level in (0.1, 0.2, 0.25, 0.3333, 0.5, 2.0, 3.0, 4.0, 5.0, 10.0))
        if common_split <= 0.08 or abs(change) >= 0.60:
            split_suspects.append({
                "date": clean["datetime"].iloc[index].strftime("%Y-%m-%d"),
                "change_pct": round(change * 100, 2),
                "ratio": round(ratio, 4),
            })
    if split_suspects:
        anomalies.append({"type": "CORPORATE_ACTION_OR_SCALE_BREAK", "events": split_suspects[-5:]})
    stale = int(clean["datetime"].duplicated().sum())
    if stale:
        anomalies.append({"type": "DUPLICATE_DATES", "count": stale})
    status = "FAILED" if invalid_ohlc.shape[0] > 0 or split_suspects else "OK"
    return {
        "status": status,
        "live_eligible": status == "OK",
        "bars": len(clean),
        "anomalies": anomalies,
    }


def _weekly_frame(df):
    return df.set_index("datetime").resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna().reset_index()


def _structural_profiles(df, atr):
    weekly = _weekly_frame(df.tail(min(1000, len(df))))
    if len(weekly) < 30:
        return []
    lows = weekly["Low"].to_numpy(dtype=float)
    highs = weekly["High"].to_numpy(dtype=float)
    swing_lows = _swing_points(lows, 2, 2, "low")
    swing_highs = _swing_points(highs, 2, 2, "high")
    candidates = []
    for start in swing_lows:
        ends = [index for index in swing_highs if index >= start + 4]
        if not ends:
            continue
        end = max(ends, key=lambda index: highs[index])
        move = (highs[end] - lows[start]) / lows[start] * 100 if lows[start] > 0 else 0
        if move >= 15 and end - start >= 4:
            candidates.append((end, move, start, end))
    candidates = sorted(candidates, key=lambda item: (item[0], item[1]), reverse=True)
    profiles = []
    used = set()
    for _, move, start, end in candidates:
        key = (start, end)
        if key in used:
            continue
        used.add(key)
        start_date = weekly["datetime"].iloc[start]
        end_date = weekly["datetime"].iloc[end]
        segment = df[(df["datetime"] >= start_date) & (df["datetime"] <= end_date)]
        profile = _volume_profile(segment, bins=64)
        if not profile:
            continue
        poc = profile["poc"]
        zone = max(atr * 0.25, poc * 0.003)
        after = df[df["datetime"] > end_date]
        touches = []
        inside_previous = False
        for row in after.itertuples(index=False):
            inside = _safe_float(row.Low) <= poc + zone and _safe_float(row.High) >= poc - zone
            if inside and not inside_previous:
                touches.append(row.datetime.strftime("%Y-%m-%d"))
            inside_previous = inside
        close = _safe_float(df["Close"].iloc[-1])
        distance_atr = abs(close - poc) / atr if atr > 0 else 99.0
        if close < poc - zone:
            status = "BROKEN"
        elif touches and close > poc + zone:
            status = "RECLAIMED"
        elif len(touches) == 0:
            status = "VIRGIN"
        elif len(touches) == 1:
            status = "FIRST_TEST"
        else:
            status = "MULTI_TESTED"
        profile.update({
            "id": f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}",
            "type": "WEEKLY_IMPULSE",
            "impulse_start": start_date.strftime("%Y-%m-%d"),
            "impulse_end": end_date.strftime("%Y-%m-%d"),
            "impulse_move_pct": round(move, 2),
            "zone_low": round(poc - zone, 2),
            "zone_high": round(poc + zone, 2),
            "touch_count": len(touches),
            "first_test_date": touches[0] if touches else None,
            "last_test_date": touches[-1] if touches else None,
            "status": status,
            "distance_atr": round(distance_atr, 2),
            "actionable": distance_atr <= 1.0 and status in ("VIRGIN", "FIRST_TEST", "RECLAIMED"),
        })
        profiles.append(profile)
        if len(profiles) == 3:
            break
    return profiles


def _detect_active_base(df, atr):
    if len(df) < 60 or atr <= 0:
        return None
    best = None
    for length in (20, 30, 40, 60, 90):
        if len(df) < length + 20:
            continue
        window = df.tail(length).reset_index(drop=True)
        lows = window["Low"].to_numpy(dtype=float)
        highs = window["High"].to_numpy(dtype=float)
        swing_lows = _swing_points(lows, 2, 2, "low")
        swing_highs = _swing_points(highs, 2, 2, "high")
        if len(swing_lows) < 2 or len(swing_highs) < 2:
            continue
        high_prices = np.array([highs[index] for index in swing_highs[-5:]], dtype=float)
        tolerance = max(atr * 0.5, float(np.median(high_prices)) * 0.0075)
        clusters = []
        for value in sorted(high_prices):
            matched = False
            for cluster in clusters:
                if abs(value - np.median(cluster)) <= tolerance:
                    cluster.append(float(value))
                    matched = True
                    break
            if not matched:
                clusters.append([float(value)])
        tested = [cluster for cluster in clusters if len(cluster) >= 2]
        if not tested:
            continue
        neck_cluster = max(tested, key=lambda cluster: (len(cluster), np.median(cluster)))
        neck = float(np.median(neck_cluster))
        base_low = float(np.min(lows))
        base_high = float(np.max(highs))
        width_pct = (base_high - base_low) / base_low * 100 if base_low > 0 else 999
        profile = _volume_profile(window, bins=48)
        if not profile:
            continue
        low_sequence = [lows[index] for index in swing_lows[-4:]]
        higher_lows = len(low_sequence) >= 2 and low_sequence[-1] >= low_sequence[0] * 0.98
        current = float(window["Close"].iloc[-1])
        previous = float(window["Close"].iloc[-2])
        neck_low = neck - tolerance * 0.5
        neck_high = neck + tolerance * 0.5
        breakout_index = None
        for index in range(max(1, len(window) - 15), len(window)):
            if window["Close"].iloc[index] > neck_high + atr * 0.20 and window["Close"].iloc[index - 1] <= neck_high:
                breakout_index = index
                break
        breakout = breakout_index is not None
        retest = False
        if breakout and breakout_index < len(window) - 1:
            post = window.iloc[breakout_index + 1:]
            retest = bool(((post["Low"] <= neck_high + atr * 0.25) & (post["Close"] >= neck_low - atr * 0.25)).any())
        if current < base_low - atr * 0.25:
            state = "BASE_INVALIDATED"
        elif breakout and retest:
            state = "NECK_RETEST"
        elif breakout:
            state = "NECK_BREAKOUT"
        elif abs(current - neck) <= atr * 0.6:
            state = "NECK_TEST"
        else:
            state = "ACCUMULATION_BASE"
        quality = 0
        quality += 25 if higher_lows else 0
        quality += 20 if len(neck_cluster) >= 3 else 10
        quality += 20 if profile["poc"] >= base_low + (base_high - base_low) * 0.35 else 0
        volume_recent = _safe_float(window["Volume"].tail(10).mean())
        volume_old = _safe_float(window["Volume"].iloc[:10].mean())
        volume_contracting = volume_old > 0 and volume_recent <= volume_old * 0.90
        quality += 20 if volume_contracting else 0
        quality += 15 if width_pct <= 25 else 5 if width_pct <= 40 else 0
        result = {
            "state": state,
            "start": window["datetime"].iloc[0].strftime("%Y-%m-%d"),
            "days": length,
            "base_low": round(base_low, 2),
            "base_high": round(base_high, 2),
            "base_width_pct": round(width_pct, 2),
            "base_poc": profile["poc"],
            "base_vah": profile["vah"],
            "base_val": profile["val"],
            "neck_center": round(neck, 2),
            "neck_low": round(neck_low, 2),
            "neck_high": round(neck_high, 2),
            "neck_tests": len(neck_cluster),
            "higher_lows": higher_lows,
            "volume_contracting": volume_contracting,
            "quality_score": min(100, quality),
            "breakout": breakout,
            "retest": retest,
            "invalidation_price": round(base_low - atr * 0.25, 2),
        }
        if best is None or result["quality_score"] > best["quality_score"]:
            best = result
    return best


def _compression(df, atr_series, poc60):
    price = _safe_float(df["Close"].iloc[-1])
    atr20 = _safe_float(atr_series.tail(20).mean())
    atr60 = _safe_float(atr_series.tail(60).mean())
    range10 = (_safe_float(df["High"].tail(10).max()) - _safe_float(df["Low"].tail(10).min())) / price * 100 if price > 0 else 0
    range60 = (_safe_float(df["High"].tail(60).max()) - _safe_float(df["Low"].tail(60).min())) / price * 100 if price > 0 else 0
    mean20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std(ddof=0)
    bandwidth = _safe_float(std20.iloc[-1] * 4 / mean20.iloc[-1] * 100) if _safe_float(mean20.iloc[-1]) > 0 else 0
    history = (std20 * 4 / mean20.replace(0, np.nan) * 100).dropna().tail(120)
    percentile = float((history <= bandwidth).mean() * 100) if len(history) else 50.0
    volume10 = _safe_float(df["Volume"].tail(10).mean())
    previous10 = _safe_float(df["Volume"].iloc[-20:-10].mean())
    volume_ratio = volume10 / previous10 if previous10 > 0 else 1.0
    distance_atr = abs(price - poc60) / atr20 if poc60 and atr20 > 0 else 99.0
    factors = {
        "atr_contracting": atr20 > 0 and atr60 > 0 and atr20 <= atr60 * 0.85,
        "range_contracting": range60 > 0 and range10 <= range60 * 0.45,
        "bandwidth_low": percentile <= 30,
        "close_near_poc60": distance_atr <= 1.0,
        "volume_contracting": volume_ratio <= 0.90,
    }
    score = sum(20 for value in factors.values() if value)
    return {
        "state": "COMPRESSED" if score >= 60 else "BUILDING" if score >= 40 else "NONE",
        "score": score,
        "atr20_pct": round(atr20 / price * 100, 2) if price > 0 else 0,
        "atr20_vs_atr60": round(atr20 / atr60, 3) if atr60 > 0 else 0,
        "range10_pct": round(range10, 2),
        "range60_pct": round(range60, 2),
        "bandwidth_percentile": round(percentile, 1),
        "volume10_vs_previous10": round(volume_ratio, 3),
        "distance_from_poc60_atr": round(distance_atr, 2),
        "factors": factors,
    }


def _bottom_state(df, profiles, compression, atr):
    price = _safe_float(df["Close"].iloc[-1])
    annual = df.tail(252)
    high252 = _safe_float(annual["High"].max())
    low252 = _safe_float(annual["Low"].min())
    drawdown = (price - high252) / high252 * 100 if high252 > 0 else 0
    range_position = (price - low252) / (high252 - low252) * 100 if high252 > low252 else 50
    recent_low = _safe_float(df["Low"].tail(20).min())
    prior_low = _safe_float(df["Low"].iloc[-60:-20].min()) if len(df) >= 60 else recent_low
    low_change = (recent_low - prior_low) / prior_low * 100 if prior_low > 0 else 0
    down_volume = _safe_float(df.loc[df["Close"] < df["Open"], "Volume"].tail(10).mean())
    up_volume = _safe_float(df.loc[df["Close"] >= df["Open"], "Volume"].tail(10).mean())
    ratio = up_volume / down_volume if down_volume > 0 else 1.0
    poc20 = profiles.get("poc20") or {}
    poc60 = profiles.get("poc60") or {}
    location_valid = drawdown <= -25 or range_position <= 20
    factors = {
        "max_location_valid": location_valid,
        "no_material_new_low": low_change >= -2.0,
        "poc20_not_falling": poc20.get("migration") in ("UP", "STABLE"),
        "poc60_not_falling": poc60.get("migration") in ("UP", "STABLE"),
        "compression_present": compression.get("score", 0) >= 40,
        "up_volume_dominant": ratio >= 1.05,
    }
    confirmations = sum(1 for key, value in factors.items() if key != "max_location_valid" and value)
    state = "ACCUMULATING" if location_valid and confirmations >= 4 else "BOTTOM_BUILDING" if location_valid and confirmations >= 2 else "MARKDOWN" if drawdown <= -20 else "NEUTRAL"
    depth = "CAPITULATION" if drawdown <= -70 else "DEEP" if drawdown <= -50 else "DEPRESSED" if drawdown <= -30 else "NORMAL"
    return {
        "state": state,
        "depth": depth,
        "drawdown_52w_pct": round(drawdown, 2),
        "range_position_52w": round(range_position, 1),
        "distance_from_52w_low_pct": round((price - low252) / low252 * 100, 2) if low252 > 0 else 0,
        "recent_low": round(recent_low, 2),
        "invalidation_price": round(max(0, recent_low - max(atr * 0.25, price * 0.003)), 2),
        "up_down_volume_ratio": round(ratio, 2),
        "factors": factors,
    }


def analyze_max_strategy(df):
    if df is None or len(df) < 140:
        return {"status": "INSUFFICIENT_DATA", "bars": 0 if df is None else len(df), "live_eligible": False}
    data = df.copy().sort_values("datetime").reset_index(drop=True)
    quality = _data_quality(data)
    price = _safe_float(data["Close"].iloc[-1])
    atr_series = _atr_series(data, 14)
    atr = _safe_float(atr_series.iloc[-1], price * 0.02)
    profiles = {
        "poc20": _rolling_profile(data, 20),
        "poc60": _rolling_profile(data, 60),
        "poc120": _rolling_profile(data, 120),
    }
    for profile in profiles.values():
        if profile and profile.get("poc", 0) > 0:
            signed = (price - profile["poc"]) / profile["poc"] * 100
            profile["signed_distance_pct"] = round(signed, 2)
            profile["distance_atr"] = round(abs(price - profile["poc"]) / atr, 2) if atr > 0 else 99
            profile["price_position"] = "ABOVE" if signed > 0.25 else "BELOW" if signed < -0.25 else "AT_POC"
    compression = _compression(data, atr_series, (profiles.get("poc60") or {}).get("poc"))
    bottom = _bottom_state(data, profiles, compression, atr)
    structural = _structural_profiles(data, atr)
    active_base = _detect_active_base(data, atr)
    actionable_profiles = [profile for profile in structural if profile.get("actionable")]
    structural_event = None
    if actionable_profiles:
        selected = min(actionable_profiles, key=lambda profile: profile.get("distance_atr", 99))
        structural_event = "MASTER_POC_RECLAIM" if selected["status"] == "RECLAIMED" else "MASTER_POC_FIRST_TEST" if selected["status"] == "FIRST_TEST" else "MASTER_POC_CONTACT"
    triggers = []
    if structural_event:
        triggers.append(structural_event)
    if active_base and active_base.get("state") in ("NECK_BREAKOUT", "NECK_RETEST"):
        triggers.append(active_base["state"])
    elif active_base and active_base.get("state") == "NECK_TEST":
        triggers.append("ACCUMULATION_NECK_TEST")
    if compression.get("state") == "COMPRESSED":
        triggers.append("COMPRESSION_READY")
    if profiles.get("poc20") and profiles["poc20"].get("migration") == "UP":
        triggers.append("POC20_MIGRATION_UP")
    rejection_reasons = []
    if quality["status"] != "OK":
        rejection_reasons.append("DATA_QUALITY_FAILED")
    if price < 2:
        rejection_reasons.append("PRICE_BELOW_2")
    if not bottom["factors"]["max_location_valid"] and not actionable_profiles:
        rejection_reasons.append("MAX_LOCATION_NOT_VALID")
    if not any(trigger in triggers for trigger in ("MASTER_POC_RECLAIM", "MASTER_POC_FIRST_TEST", "MASTER_POC_CONTACT", "NECK_BREAKOUT", "NECK_RETEST")):
        rejection_reasons.append("NO_STRUCTURAL_TRIGGER")
    score = 0
    score += 25 if bottom["depth"] == "CAPITULATION" else 18 if bottom["depth"] == "DEEP" else 10 if bottom["depth"] == "DEPRESSED" else 0
    score += 25 if bottom["state"] == "ACCUMULATING" else 15 if bottom["state"] == "BOTTOM_BUILDING" else 0
    score += compression["score"] * 0.15
    score += 20 if structural_event else 0
    score += 15 if active_base and active_base.get("state") == "NECK_RETEST" else 12 if active_base and active_base.get("state") == "NECK_BREAKOUT" else 5 if active_base and active_base.get("state") == "NECK_TEST" else 0
    score += 5 if active_base and active_base.get("quality_score", 0) >= 60 else 0
    live_eligible = not rejection_reasons
    return {
        "status": "OK",
        "version": "max_structure_v1_1",
        "bars_analyzed": len(data),
        "price": round(price, 2),
        "atr14": round(atr, 4),
        "atr14_pct": round(atr / price * 100, 2) if price > 0 else 0,
        "data_quality": quality,
        "profiles": profiles,
        "structural_profiles": structural,
        "master_poc": structural[0] if structural else None,
        "active_base": active_base,
        "compression": compression,
        "bottom": bottom,
        "triggers": triggers,
        "rejection_reasons": rejection_reasons,
        "max_score": round(min(100, score), 1),
        "watch_ready": live_eligible and score >= 45,
        "live_eligible": live_eligible,
        "live_entry_enabled": False,
    }
