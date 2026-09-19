import math
import numpy as np
import pandas as pd


def _native(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _native(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_native(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_native(item) for item in value)
    return value


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
        base_low = float(np.min(lows))
        base_high = float(np.max(highs))
        width_pct = (base_high - base_low) / base_low * 100 if base_low > 0 else 999
        profile = _volume_profile(window, bins=48)
        if not profile:
            continue
        base_midpoint = base_low + (base_high - base_low) * 0.50
        qualified = [
            cluster for cluster in tested
            if float(np.median(cluster)) > profile["poc"]
            and float(np.median(cluster)) >= base_midpoint
        ]
        if not qualified:
            continue
        neck_cluster = max(qualified, key=lambda cluster: (float(np.median(cluster)), len(cluster)))
        neck = float(np.median(neck_cluster))
        low_sequence = [lows[index] for index in swing_lows[-4:]]
        higher_lows = bool(len(low_sequence) >= 2 and low_sequence[-1] >= low_sequence[0] * 0.98)
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
        breakout_date = window["datetime"].iloc[breakout_index].strftime("%Y-%m-%d") if breakout else None
        breakout_age_days = len(window) - 1 - breakout_index if breakout else None
        retest = False
        retest_index = None
        if breakout and breakout_index < len(window) - 1:
            post = window.iloc[breakout_index + 1:]
            mask = (post["Low"] <= neck_high + atr * 0.25) & (post["Close"] >= neck_low - atr * 0.25)
            if bool(mask.any()):
                retest = True
                retest_index = int(mask[mask].index[0])
        retest_date = window["datetime"].iloc[retest_index].strftime("%Y-%m-%d") if retest_index is not None else None
        retest_age_days = len(window) - 1 - retest_index if retest_index is not None else None
        breakout_fresh = breakout and breakout_age_days is not None and breakout_age_days <= 12
        retest_fresh = retest and retest_age_days is not None and retest_age_days <= 8
        if current < base_low - atr * 0.25:
            state = "BASE_INVALIDATED"
        elif breakout and retest and retest_fresh:
            state = "NECK_RETEST"
        elif breakout and breakout_fresh:
            state = "NECK_BREAKOUT"
        elif breakout:
            state = "POST_BREAKOUT"
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
            "breakout_date": breakout_date,
            "breakout_age_days": breakout_age_days,
            "breakout_fresh": breakout_fresh,
            "retest": retest,
            "retest_date": retest_date,
            "retest_age_days": retest_age_days,
            "retest_fresh": retest_fresh,
            "invalidation_price": round(base_low - atr * 0.25, 2),
        }
        if best is None or result["quality_score"] > best["quality_score"]:
            best = result
    return best


def _detect_weekly_rounding_base_v131(df, daily_atr):
    weekly = _weekly_frame(df.tail(min(1000, len(df))))
    if len(weekly) < 30:
        return None
    weekly_atr_series = _atr_series(weekly, 14)
    weekly_atr = _safe_float(weekly_atr_series.iloc[-1], daily_atr * 2.2)
    current_price = _safe_float(weekly["Close"].iloc[-1])
    best = None
    for length in (20, 26, 32, 40, 52, 65):
        if len(weekly) < length + 8:
            continue
        window = weekly.tail(length).reset_index(drop=True)
        preceding = weekly.iloc[max(0, len(weekly) - length - 26):len(weekly) - length].reset_index(drop=True)
        lows = window["Low"].to_numpy(dtype=float)
        highs = window["High"].to_numpy(dtype=float)
        closes = window["Close"].to_numpy(dtype=float)
        swing_lows = _swing_points(lows, 2, 2, "low")
        swing_highs = _swing_points(highs, 2, 2, "high")
        if len(swing_lows) < 2 or len(swing_highs) < 2:
            continue
        base_low = float(np.min(lows))
        bottom_index = int(np.argmin(lows))
        if bottom_index < 3 or bottom_index > len(window) - 4:
            continue
        prior_peak = float(preceding["High"].max()) if len(preceding) else float(np.max(highs[:bottom_index + 1]))
        markdown_pct = (base_low - prior_peak) / prior_peak * 100 if prior_peak > 0 else 0
        preceding_downtrend = False
        if len(preceding) >= 8:
            first_half = float(preceding["Close"].iloc[:max(3, len(preceding)//2)].mean())
            second_half = float(preceding["Close"].iloc[-max(3, len(preceding)//2):].mean())
            preceding_downtrend = second_half <= first_half * 0.92
        if markdown_pct > -15 and not preceding_downtrend:
            continue
        left_low_indices = [index for index in swing_lows if index <= bottom_index]
        right_low_indices = [index for index in swing_lows if index > bottom_index]
        if not right_low_indices:
            continue
        right_lows = [float(lows[index]) for index in right_low_indices[-4:]]
        right_side_recovery = right_lows[-1] >= base_low * 1.02
        double_bottom = any(abs(value - base_low) <= weekly_atr * 0.75 for value in right_lows)
        higher_second_low = any(value > base_low and value <= base_low + weekly_atr * 1.5 for value in right_lows)
        shape_valid = right_side_recovery and (double_bottom or higher_second_low)
        if not shape_valid:
            continue
        profile = _volume_profile(window, bins=56)
        if not profile:
            continue
        pre_bottom_highs = [index for index in swing_highs if index < bottom_index]
        post_bottom_highs = [index for index in swing_highs if index > bottom_index]
        structural_high_indices = pre_bottom_highs[-3:] + post_bottom_highs[-6:]
        if len(structural_high_indices) < 2:
            continue
        high_prices = np.array([highs[index] for index in structural_high_indices], dtype=float)
        tolerance = max(weekly_atr * 0.40, float(np.median(high_prices)) * 0.012)
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
        qualified = [cluster for cluster in tested if float(np.median(cluster)) >= profile["poc"] - weekly_atr * 0.25]
        if not qualified:
            continue
        neck_cluster = max(qualified, key=lambda cluster: (len(cluster), float(np.median(cluster))))
        neck = float(np.median(neck_cluster))
        neck_low = neck - tolerance * 0.5
        neck_high = neck + tolerance * 0.5
        poc_neck_distance_atr = abs(neck - profile["poc"]) / weekly_atr if weekly_atr > 0 else 99.0
        poc_neck_confluence = "STRONG" if poc_neck_distance_atr <= 0.50 else "NORMAL" if poc_neck_distance_atr <= 1.0 else "SEPARATE"
        breakout_indices = [
            index for index in range(max(1, bottom_index + 1), len(window))
            if closes[index] > neck_high + weekly_atr * 0.10 and closes[index - 1] <= neck_high
        ]
        breakout_index = breakout_indices[0] if breakout_indices else None
        breakout = breakout_index is not None
        breakout_age_weeks = len(window) - 1 - breakout_index if breakout else None
        breakout_recent = breakout and breakout_age_weeks <= 16
        post_breakout_high = float(np.max(highs[breakout_index:])) if breakout else 0.0
        post_breakout_high_index = int(breakout_index + np.argmax(highs[breakout_index:])) if breakout else None
        correction_origin = None
        inefficiency_type = None
        if breakout:
            search_start = max(0, bottom_index - 20)
            for index in range(search_start + 2, bottom_index + 1):
                if window["High"].iloc[index] < window["Low"].iloc[index - 2]:
                    level = float(window["Low"].iloc[index - 2])
                    if level > neck:
                        correction_origin = level if correction_origin is None else min(correction_origin, level)
                        inefficiency_type = "BEARISH_FVG_ORIGIN"
            if correction_origin is None:
                overhead = [float(highs[index]) for index in pre_bottom_highs if highs[index] > neck]
                correction_origin = min(overhead) if overhead else None
                inefficiency_type = "PRIOR_SWING_HIGH" if correction_origin else None
        target_reached = bool(correction_origin and post_breakout_high >= correction_origin - weekly_atr * 0.20)
        target_rejected = bool(target_reached and current_price <= correction_origin - weekly_atr * 0.50)
        return_after_target = bool(target_reached and post_breakout_high_index is not None and len(window) - 1 > post_breakout_high_index)
        near_neck = abs(current_price - neck) <= weekly_atr * 0.75
        retest_in_progress = bool(breakout and return_after_target and near_neck and current_price >= neck_low - weekly_atr * 0.30)
        recent = window.tail(5).reset_index(drop=True)
        recent_low_index = int(np.argmin(recent["Low"].to_numpy(dtype=float)))
        reaction_low_confirmed = recent_low_index <= 2 and recent_low_index < len(recent) - 1
        post_low_high = float(recent["High"].iloc[recent_low_index + 1:].max()) if recent_low_index < len(recent) - 1 else 0.0
        reaction_break = reaction_low_confirmed and float(recent["Close"].iloc[-1]) > post_low_high * 0.995
        selling_volume = _safe_float(recent.loc[recent["Close"] < recent["Open"], "Volume"].mean())
        prior_selling_volume = _safe_float(window.iloc[-10:-5].loc[window.iloc[-10:-5]["Close"] < window.iloc[-10:-5]["Open"], "Volume"].mean())
        selling_volume_contracting = prior_selling_volume <= 0 or selling_volume <= prior_selling_volume
        retest_confirmed = bool(retest_in_progress and reaction_low_confirmed and reaction_break and selling_volume_contracting and current_price >= neck_low)
        second_leg_ready = bool(retest_confirmed and target_rejected)
        failed_breakout = bool(breakout_recent and current_price < neck_low - weekly_atr * 0.35)
        quality = 0
        quality += 20 if markdown_pct <= -20 else 12
        quality += 20 if shape_valid else 0
        quality += 15 if len(neck_cluster) >= 3 else 10
        quality += 20 if poc_neck_confluence == "STRONG" else 12 if poc_neck_confluence == "NORMAL" else 0
        quality += 15 if right_side_recovery else 0
        quality += 10 if breakout else 0
        if not breakout and quality < 65:
            continue
        if failed_breakout:
            state = "FAILED_ROUNDING_BREAKOUT"
        elif second_leg_ready:
            state = "SECOND_LEG_REENTRY"
        elif retest_confirmed:
            state = "POC_NECK_RETEST_CONFIRMED"
        elif retest_in_progress:
            state = "POC_NECK_RETEST_IN_PROGRESS"
        elif target_rejected:
            state = "CORRECTION_ORIGIN_REJECTION"
        elif target_reached:
            state = "CORRECTION_ORIGIN_REACHED"
        elif breakout_recent:
            state = "ROUNDING_NECK_BREAKOUT"
        elif breakout:
            state = "HISTORICAL_ROUNDING_BREAKOUT"
        else:
            state = "ROUNDING_ACCUMULATION"
        next_required_event = None
        if state == "POC_NECK_RETEST_IN_PROGRESS":
            next_required_event = "REACTION_SWING"
        elif state == "ROUNDING_ACCUMULATION":
            next_required_event = "NECK_BREAKOUT"
        elif state == "CORRECTION_ORIGIN_REJECTION":
            next_required_event = "POC_NECK_RETEST"
        result = {
            "type": "ROUNDING_ACCUMULATION",
            "state": state,
            "start": window["datetime"].iloc[0].strftime("%Y-%m-%d"),
            "weeks": length,
            "base_low": round(base_low, 2),
            "base_poc": profile["poc"],
            "base_vah": profile["vah"],
            "base_val": profile["val"],
            "neck_center": round(neck, 2),
            "neck_low": round(neck_low, 2),
            "neck_high": round(neck_high, 2),
            "neck_tests": len(neck_cluster),
            "poc_neck_distance_atr": round(poc_neck_distance_atr, 2),
            "poc_neck_confluence": poc_neck_confluence,
            "markdown_pct": round(markdown_pct, 2),
            "shape_valid": shape_valid,
            "right_side_recovery": right_side_recovery,
            "quality_score": min(100, quality),
            "breakout": breakout,
            "breakout_date": window["datetime"].iloc[breakout_index].strftime("%Y-%m-%d") if breakout else None,
            "breakout_age_weeks": breakout_age_weeks,
            "breakout_recent": breakout_recent,
            "post_breakout_high": round(post_breakout_high, 2) if breakout else None,
            "correction_origin": round(correction_origin, 2) if correction_origin else None,
            "correction_origin_type": inefficiency_type,
            "target_reached": target_reached,
            "target_rejected": target_rejected,
            "retest_in_progress": retest_in_progress,
            "reaction_low_confirmed": reaction_low_confirmed,
            "reaction_break": reaction_break,
            "selling_volume_contracting": selling_volume_contracting,
            "retest_confirmed": retest_confirmed,
            "second_leg_ready": second_leg_ready,
            "failed_breakout": failed_breakout,
            "next_required_event": next_required_event,
            "invalidation_price": round(neck_low - weekly_atr * 0.35, 2),
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


def _market_phase_gates(df, profiles, bottom, structural_profiles, active_base, structural_base, atr):
    weekly = _weekly_frame(df.tail(min(1000, len(df))))
    price = _safe_float(df["Close"].iloc[-1])
    annual = df.tail(252)
    high252 = _safe_float(annual["High"].max())
    low252 = _safe_float(annual["Low"].min())
    range_position = (price - low252) / (high252 - low252) * 100 if high252 > low252 else 50.0
    weekly_close = weekly["Close"] if len(weekly) else pd.Series(dtype=float)
    weekly_rsi = 50.0
    if len(weekly_close) >= 15:
        delta = weekly_close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        weekly_rsi = _safe_float((100 - (100 / (1 + rs))).iloc[-1], 50.0)
    recent_low = _safe_float(weekly["Low"].tail(4).min()) if len(weekly) >= 4 else _safe_float(df["Low"].tail(20).min())
    prior_low = _safe_float(weekly["Low"].iloc[-12:-4].min()) if len(weekly) >= 12 else recent_low
    new_weekly_low = recent_low < prior_low - max(atr * 0.25, prior_low * 0.005)
    weekly_ema10 = _safe_float(weekly_close.ewm(span=10).mean().iloc[-1], price) if len(weekly_close) else price
    weekly_ema20_series = weekly_close.ewm(span=20).mean() if len(weekly_close) else pd.Series([price])
    weekly_ema20 = _safe_float(weekly_ema20_series.iloc[-1], price)
    weekly_ema20_prev = _safe_float(weekly_ema20_series.iloc[-4], weekly_ema20) if len(weekly_ema20_series) >= 4 else weekly_ema20
    weekly_ema20_falling = weekly_ema20 < weekly_ema20_prev * 0.995
    poc20 = profiles.get("poc20") or {}
    poc60 = profiles.get("poc60") or {}
    poc20_supportive = poc20.get("migration") in ("UP", "STABLE")
    poc60_supportive = poc60.get("migration") in ("UP", "STABLE")
    local_high = _safe_float(df["High"].iloc[-20:-5].max()) if len(df) >= 25 else _safe_float(df["High"].tail(20).max())
    local_break = price > local_high + atr * 0.10
    higher_low = _safe_float(df["Low"].tail(10).min()) >= _safe_float(df["Low"].iloc[-30:-10].min()) * 0.98 if len(df) >= 30 else False
    weekly_rsi_turn = weekly_rsi >= 35 and len(weekly_close) >= 3 and weekly_close.iloc[-1] >= weekly_close.iloc[-2]
    base_trigger = bool(active_base and active_base.get("state") in ("NECK_BREAKOUT", "NECK_RETEST"))
    structural_trigger = bool(structural_base and structural_base.get("state") in (
        "ROUNDING_NECK_BREAKOUT", "POC_NECK_RETEST_CONFIRMED", "SECOND_LEG_REENTRY"
    ))
    master_reaction = any(
        profile.get("distance_atr", 99) <= 0.75 and profile.get("status") in ("FIRST_TEST", "RECLAIMED")
        for profile in structural_profiles
    )
    momentum_signals = {
        "no_new_weekly_low": not new_weekly_low,
        "poc20_supportive": poc20_supportive,
        "poc60_supportive": poc60_supportive,
        "higher_low": bool(higher_low),
        "local_high_break": bool(local_break),
        "weekly_rsi_turn": bool(weekly_rsi_turn),
        "base_trigger": base_trigger,
        "structural_trigger": structural_trigger,
        "master_poc_reaction": master_reaction,
    }
    momentum_count = sum(1 for value in momentum_signals.values() if value)
    momentum_change = momentum_count >= 4 and (local_break or base_trigger or structural_trigger or master_reaction)
    falling_knife = bool(
        bottom.get("drawdown_52w_pct", 0) <= -25
        and new_weekly_low
        and weekly_ema20_falling
        and not momentum_change
    )
    nearest_master_below = None
    below = [profile for profile in structural_profiles if profile.get("poc", 0) < price]
    if below:
        nearest_master_below = max(below, key=lambda profile: profile.get("poc", 0))
    waiting_master_poc = bool(falling_knife and nearest_master_below)
    mature_markup = bool(
        range_position >= 80
        and price > weekly_ema20
        and (
            (active_base and price > active_base.get("neck_center", price) + atr * 2.0)
            or (structural_base and price > structural_base.get("neck_center", price) + atr * 2.0)
            or price > (poc60.get("poc") or price) + atr * 3.0
        )
    )
    range_bottom = bool(bottom.get("range_position_52w", 50) <= 20 and bottom.get("drawdown_52w_pct", 0) > -25)
    deep_drawdown = bool(bottom.get("drawdown_52w_pct", 0) <= -25)
    if mature_markup:
        phase = "MATURE_MARKUP_BLOCK"
    elif falling_knife:
        phase = "FALLING_KNIFE_WAIT"
    elif deep_drawdown and momentum_change:
        phase = "DEEP_REVERSAL"
    elif deep_drawdown:
        phase = "DEEP_DRAWDOWN_WAIT"
    elif range_bottom and momentum_change:
        phase = "RANGE_BOTTOM_REVERSAL"
    elif range_bottom:
        phase = "RANGE_BOTTOM_WAIT"
    elif master_reaction and momentum_change:
        phase = "MASTER_POC_REVERSAL"
    else:
        phase = "NEUTRAL"
    return {
        "phase": phase,
        "deep_drawdown": deep_drawdown,
        "range_bottom": range_bottom,
        "falling_knife_block": falling_knife,
        "mature_markup_block": mature_markup,
        "momentum_change": momentum_change,
        "momentum_score": momentum_count,
        "momentum_signals": momentum_signals,
        "weekly_rsi": round(weekly_rsi, 1),
        "weekly_ema20": round(weekly_ema20, 2),
        "weekly_ema20_falling": weekly_ema20_falling,
        "new_weekly_low": new_weekly_low,
        "range_position_52w": round(range_position, 1),
        "waiting_master_poc": waiting_master_poc,
        "next_master_poc": round(nearest_master_below.get("poc"), 2) if nearest_master_below else None,
        "next_master_poc_status": nearest_master_below.get("status") if nearest_master_below else None,
    }

def analyze_max_strategy(df):
    if df is None or len(df) < 140:
        return {"status": "INSUFFICIENT_DATA", "bars": 0 if df is None else len(df), "data_eligible": False, "strategy_eligible": False, "trade_ready": False}
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
    structural_base = _detect_weekly_rounding_base_v131(data, atr)
    phase = _market_phase_gates(data, profiles, bottom, structural, active_base, structural_base, atr)
    qualified_profiles = [
        profile for profile in structural
        if profile.get("distance_atr", 99) <= 0.75
        and profile.get("status") in ("VIRGIN", "FIRST_TEST", "RECLAIMED")
        and profile.get("impulse_move_pct", 0) >= 20
        and profile.get("bars", 0) >= 20
    ]
    active_structural_profile = min(qualified_profiles, key=lambda profile: profile.get("distance_atr", 99)) if qualified_profiles else None
    structural_event = None
    if active_structural_profile:
        status = active_structural_profile.get("status")
        structural_event = "MASTER_POC_RECLAIM" if status == "RECLAIMED" else "MASTER_POC_FIRST_TEST" if status == "FIRST_TEST" else "MASTER_POC_CONTACT"
    if phase["phase"] == "DEEP_REVERSAL":
        strategy_type = "MAX_DEEP_REVERSAL"
    elif phase["phase"] == "RANGE_BOTTOM_REVERSAL":
        strategy_type = "RANGE_BOTTOM_REVERSAL"
    elif phase["phase"] == "MASTER_POC_REVERSAL" or active_structural_profile and phase["momentum_change"]:
        strategy_type = "MASTER_POC_RETURN"
    else:
        strategy_type = None
    triggers = []
    if structural_event:
        triggers.append(structural_event)
    if active_base and active_base.get("state") in ("NECK_BREAKOUT", "NECK_RETEST"):
        triggers.append(active_base["state"])
    elif active_base and active_base.get("state") == "NECK_TEST":
        triggers.append("ACCUMULATION_NECK_TEST")
    if structural_base:
        structural_state = structural_base.get("state")
        if structural_state in (
            "ROUNDING_NECK_BREAKOUT", "POC_NECK_RETEST_IN_PROGRESS",
            "POC_NECK_RETEST_CONFIRMED", "SECOND_LEG_REENTRY",
            "CORRECTION_ORIGIN_REACHED", "CORRECTION_ORIGIN_REJECTION",
            "FAILED_ROUNDING_BREAKOUT",
        ):
            triggers.append(structural_state)
    if compression.get("state") == "COMPRESSED":
        triggers.append("COMPRESSION_READY")
    if profiles.get("poc20") and profiles["poc20"].get("migration") == "UP":
        triggers.append("POC20_MIGRATION_UP")
    data_rejections = []
    if quality["status"] != "OK":
        data_rejections.append("DATA_QUALITY_FAILED")
    if price < 2:
        data_rejections.append("PRICE_BELOW_2")
    strategy_rejections = []
    if phase["falling_knife_block"]:
        strategy_rejections.append("FALLING_KNIFE_BLOCK")
    if phase["mature_markup_block"]:
        strategy_rejections.append("MATURE_MARKUP_BLOCK")
    if not phase["momentum_change"]:
        strategy_rejections.append("MOMENTUM_CHANGE_REQUIRED")
    if not strategy_type:
        strategy_rejections.append("NO_QUALIFIED_REVERSAL")
    structural_trigger = any(trigger in triggers for trigger in (
        "MASTER_POC_RECLAIM", "MASTER_POC_FIRST_TEST", "NECK_BREAKOUT", "NECK_RETEST",
        "ROUNDING_NECK_BREAKOUT", "POC_NECK_RETEST_CONFIRMED", "SECOND_LEG_REENTRY",
    ))
    if not structural_trigger:
        strategy_rejections.append("NO_EXECUTABLE_STRUCTURAL_TRIGGER")
    score = 0
    score += 25 if bottom["depth"] == "CAPITULATION" else 18 if bottom["depth"] == "DEEP" else 10 if bottom["depth"] == "DEPRESSED" else 0
    score += 25 if bottom["state"] == "ACCUMULATING" else 15 if bottom["state"] == "BOTTOM_BUILDING" else 0
    score += compression["score"] * 0.15
    score += 20 if structural_event in ("MASTER_POC_FIRST_TEST", "MASTER_POC_RECLAIM") else 0
    score += 15 if active_base and active_base.get("state") == "NECK_RETEST" else 12 if active_base and active_base.get("state") == "NECK_BREAKOUT" else 5 if active_base and active_base.get("state") == "NECK_TEST" else 0
    score += 5 if active_base and active_base.get("quality_score", 0) >= 60 else 0
    score += 15 if structural_base and structural_base.get("state") == "SECOND_LEG_REENTRY" else 12 if structural_base and structural_base.get("state") == "POC_NECK_RETEST_CONFIRMED" else 8 if structural_base and structural_base.get("state") == "ROUNDING_NECK_BREAKOUT" else 0
    score += min(10, phase["momentum_score"] * 2)
    data_eligible = not data_rejections
    strategy_eligible = data_eligible and not strategy_rejections
    watch_ready = data_eligible and phase["phase"] in ("DEEP_REVERSAL", "DEEP_DRAWDOWN_WAIT", "RANGE_BOTTOM_REVERSAL", "MASTER_POC_REVERSAL") and score >= 35 and not phase["mature_markup_block"]
    tranche_1_trigger = structural_event in ("MASTER_POC_FIRST_TEST", "MASTER_POC_RECLAIM") and phase["momentum_change"]
    tranche_2_trigger = any(trigger in triggers for trigger in ("NECK_BREAKOUT", "ROUNDING_NECK_BREAKOUT")) and phase["momentum_change"]
    tranche_3_trigger = any(trigger in triggers for trigger in ("NECK_RETEST", "POC_NECK_RETEST_CONFIRMED", "SECOND_LEG_REENTRY")) and phase["momentum_change"]
    tranche_1_ready = strategy_eligible and tranche_1_trigger
    tranche_2_ready = strategy_eligible and tranche_2_trigger
    tranche_3_ready = strategy_eligible and tranche_3_trigger
    planned_tranche_pct = 40 if tranche_3_ready else 35 if tranche_2_ready else 25 if tranche_1_ready else 0
    entry_trigger = "RETEST" if tranche_3_ready else "BREAKOUT" if tranche_2_ready else "MASTER_POC_REACTION" if tranche_1_ready else None
    trade_ready = bool(tranche_1_ready or tranche_2_ready or tranche_3_ready)
    waiting_state = None
    if phase["falling_knife_block"]:
        waiting_state = "WAITING_MASTER_POC" if phase["waiting_master_poc"] else "FALLING_KNIFE_WAIT"
    elif phase["deep_drawdown"] and not phase["momentum_change"]:
        waiting_state = "DEEP_REVERSAL_WAITING_TRIGGER"
    elif structural_base and structural_base.get("state") == "POC_NECK_RETEST_IN_PROGRESS":
        waiting_state = "WAITING_REACTION_SWING"
    rejection_reasons = list(dict.fromkeys(data_rejections + strategy_rejections))
    return _native({
        "status": "OK",
        "version": "max_structure_v1_4",
        "bars_analyzed": len(data),
        "price": round(price, 2),
        "atr14": round(atr, 4),
        "atr14_pct": round(atr / price * 100, 2) if price > 0 else 0,
        "data_quality": quality,
        "profiles": profiles,
        "structural_profiles": structural,
        "master_poc": structural[0] if structural else None,
        "active_structural_profile": active_structural_profile,
        "active_structural_event": structural_event,
        "strategy_type": strategy_type,
        "market_phase": phase,
        "waiting_state": waiting_state,
        "active_base": active_base,
        "structural_base": structural_base,
        "compression": compression,
        "bottom": bottom,
        "triggers": triggers,
        "data_rejection_reasons": data_rejections,
        "strategy_rejection_reasons": strategy_rejections,
        "rejection_reasons": rejection_reasons,
        "max_score": round(min(100, score), 1),
        "data_eligible": data_eligible,
        "strategy_eligible": strategy_eligible,
        "watch_ready": watch_ready,
        "tranche_1_ready": tranche_1_ready,
        "tranche_2_ready": tranche_2_ready,
        "tranche_3_ready": tranche_3_ready,
        "planned_tranche_pct": planned_tranche_pct,
        "entry_trigger": entry_trigger,
        "trade_ready": trade_ready,
        "live_eligible": strategy_eligible,
        "live_entry_enabled": False,
    })
