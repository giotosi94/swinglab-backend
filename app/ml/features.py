"""
SwingLab ML — Feature Extraction v2.0
Feature engineering: rimossa sector_rank (morta), aggiunte 3 interazioni.
"""

# ---- Feature encoding maps ----
SETUP_ENCODE = {
    "breakout": 0, "pullback_to_poc": 1, "ema_bounce": 2,
    "oversold_reversal": 3, "overbought_warning": 4, "neutral": 5,
}

WYCKOFF_ENCODE = {
    "accumulation": 0, "markup": 1, "spring": 2,
    "distribution": 3, "markdown": 4,
}

REGIME_ENCODE = {"BULL": 0, "NEUTRAL": 1, "BEAR": 2, "CRASH": 3}

# v2.0 — sector_rank rimossa (sempre morta). +3 interazioni.
FEATURE_NAMES = [
    "rsi", "macd_histogram", "ema_alignment", "relative_volume",
    "poc_distance_pct", "setup_type_encoded",
    "wyckoff_encoded", "accumulation_score", "range_position",
    "change_pct", "regime_encoded", "confluence_score",
    "has_bullish_patterns", "pct_from_high",
    # 🆕 v2.0 interaction features
    "rsi_volume_interaction",
    "trend_strength",
    "value_proximity",
]


def get_feature_names():
    """Return ordered list of feature names."""
    return FEATURE_NAMES.copy()


def _calc_ema_alignment(asset):
    """Calculate EMA alignment: 2=full, 1=partial, 0=none."""
    price = asset.get("price", 0)
    ema10 = asset.get("ema10", 0)
    ema20 = asset.get("ema20", 0)
    ema50 = asset.get("ema50", 0)
    if price and ema10 and ema20 and ema50:
        if price > ema10 > ema20 > ema50:
            return 2
        elif price > ema20 > ema50:
            return 1
    return 0


def _calc_poc_distance(asset):
    """Calculate % distance from POC price."""
    price = asset.get("price", 0)
    poc = asset.get("poc_price", 0)
    if price and poc:
        return round(abs(price - poc) / price * 100, 2)
    return 50.0


def _has_bullish(asset):
    """Check if asset has bullish candlestick patterns."""
    patterns = asset.get("candlestick_patterns", [])
    return 1 if any(p.get("type") == "bullish" for p in patterns) else 0


def extract_features_from_asset(asset, market_context=None):
    """
    Extract ML features from an asset document.
    v2.0: 14 base + 3 interaction = 17 features.
    """
    mc = market_context or {}
    wyckoff = asset.get("wyckoff", {})
    accum = asset.get("accumulation", {})
    macd = asset.get("macd", {})

    # Regime
    regime = mc.get("regime", asset.get("market_regime", "NEUTRAL"))

    # ---- Base values ----
    rsi = round(asset.get("rsi", 50), 2)
    macd_hist = round(macd.get("histogram", 0) if isinstance(macd, dict) else 0, 4)
    ema_align = _calc_ema_alignment(asset)
    rel_vol = round(asset.get("relative_volume", 1), 2)
    poc_dist = _calc_poc_distance(asset)
    accum_score = round(accum.get("score", 0) if isinstance(accum, dict) else 0, 1)
    confluence = round(asset.get("setup_score", 0), 1)

    # ---- 🆕 v2.0 Interaction features ----
    # 1. RSI sweet-spot × volume: premia RSI vicino a 50 CON volume alto
    rsi_sweet = max(0.0, 1.0 - abs(rsi - 50) / 50)  # 0..1, picco a rsi=50
    rsi_volume_interaction = round(rsi_sweet * rel_vol, 3)

    # 2. Trend strength: allineamento EMA × direzione MACD (-2..+2)
    macd_dir = 1 if macd_hist > 0 else (-1 if macd_hist < 0 else 0)
    trend_strength = round(ema_align * macd_dir, 2)

    # 3. Value proximity: vicinanza a POC × accumulazione istituzionale (0..1)
    poc_closeness = max(0.0, (5.0 - poc_dist) / 5.0)  # 1 se sul POC, 0 se >5% lontano
    value_proximity = round(poc_closeness * (accum_score / 100.0), 3)

    features = {
        "rsi": rsi,
        "macd_histogram": macd_hist,
        "ema_alignment": ema_align,
        "relative_volume": rel_vol,
        "poc_distance_pct": poc_dist,
        "setup_type_encoded": SETUP_ENCODE.get(asset.get("setup_type", "neutral"), 5),
        "wyckoff_encoded": WYCKOFF_ENCODE.get(wyckoff.get("phase", ""), 5),
        "accumulation_score": accum_score,
        "range_position": round(asset.get("range_position", 50), 1),
        "change_pct": round(asset.get("change_pct", 0), 2),
        "regime_encoded": REGIME_ENCODE.get(regime, 1),
        "confluence_score": confluence,
        "has_bullish_patterns": _has_bullish(asset),
        "pct_from_high": round(asset.get("pct_from_high", -50), 2),
        # interazioni
        "rsi_volume_interaction": rsi_volume_interaction,
        "trend_strength": trend_strength,
        "value_proximity": value_proximity,
    }
    return features


def extract_features_from_trade(trade, asset_at_entry=None, market_context=None):
    """Extract features from a trade document (from trade_history)."""
    asset = asset_at_entry or {}
    mc = market_context or {}

    pseudo = {
        "rsi": trade.get("rsi_at_entry", asset.get("rsi", 50)),
        "macd": trade.get("macd_at_entry", asset.get("macd", {})),
        "price": trade.get("entry_price", asset.get("price", 0)),
        "ema10": trade.get("ema10_at_entry", asset.get("ema10", 0)),
        "ema20": trade.get("ema20_at_entry", asset.get("ema20", 0)),
        "ema50": trade.get("ema50_at_entry", asset.get("ema50", 0)),
        "relative_volume": trade.get("relative_volume", asset.get("relative_volume", 1)),
        "poc_price": trade.get("poc_price", asset.get("poc_price", 0)),
        "setup_type": trade.get("setup_type", asset.get("setup_type", "neutral")),
        "sector_code": trade.get("sector", asset.get("sector_code", "")),
        "wyckoff": trade.get("wyckoff", asset.get("wyckoff", {})),
        "accumulation": trade.get("accumulation", asset.get("accumulation", {})),
        "range_position": trade.get("range_position", asset.get("range_position", 50)),
        "change_pct": trade.get("change_pct_at_entry", asset.get("change_pct", 0)),
        "market_regime": trade.get("market_regime", mc.get("regime", "NEUTRAL")),
        "setup_score": trade.get("confluence", trade.get("setup_score", asset.get("setup_score", 0))),
        "candlestick_patterns": trade.get("patterns", asset.get("candlestick_patterns", [])),
        "pct_from_high": trade.get("pct_from_high", asset.get("pct_from_high", -50)),
    }

    return extract_features_from_asset(pseudo, mc)


def features_to_array(features):
    """Convert feature dict to ordered list matching get_feature_names()."""
    return [features.get(name, 0) for name in FEATURE_NAMES]
