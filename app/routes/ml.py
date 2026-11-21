"""
SwingLab ML — API Routes

v3.0 — Model versioning endpoints + cron-ready weekly retrain
"""

from fastapi import APIRouter, Query
from app.db.mongodb import get_db
from app.ml.model import ml_model
from app.ml.trend_model import trend_predictor

router = APIRouter(prefix="/api/ml", tags=["ml"])


@router.get("/status")
async def ml_status():
    return await ml_model.get_status()


@router.post("/train")
async def ml_train():
    result = await ml_model.train(use_synthetic_if_needed=True)
    return result


@router.post("/predict/{ticker}")
async def ml_predict_ticker(ticker: str):
    db = get_db()
    asset = await db.assets.find_one({"ticker": ticker.upper()})
    if not asset:
        return {"error": f"Ticker {ticker} not found"}
    ps = await db.auto_trader.find_one({"_id": "alpaca_state"})
    market_context = ps.get("market", {}) if ps else {}
    result = await ml_model.predict(asset, market_context)
    result["ticker"] = ticker.upper()
    return result


@router.get("/predict/all")
async def ml_predict_all():
    db = get_db()
    assets = await db.assets.find({}).to_list(length=250)
    if not assets:
        return {"error": "No assets found"}
    ps = await db.auto_trader.find_one({"_id": "alpaca_state"})
    market_context = ps.get("market", {}) if ps else {}
    predictions = await ml_model.predict_batch(assets, market_context)
    sorted_preds = sorted(
        [{"ticker": k, **v} for k, v in predictions.items() if v.get("ml_score") is not None],
        key=lambda x: x["ml_score"], reverse=True,
    )
    return {
        "total": len(sorted_preds),
        "top_20": sorted_preds[:20],
        "model_status": await ml_model.get_status(),
    }


@router.get("/trend/status")
async def trend_status():
    return await trend_predictor.get_status()


@router.post("/trend/train")
async def trend_train():
    result = await trend_predictor.train()
    return result


@router.post("/trend/predict/{ticker}")
async def trend_predict_ticker(ticker: str):
    result = await trend_predictor.predict_stock(ticker)
    return result


@router.get("/trend/all")
async def trend_predict_all():
    results = await trend_predictor.predict_all()
    return {
        "total": len(results),
        "predictions": results,
        "model_status": await trend_predictor.get_status(),
    }


# ============================================
# v3.0 — MODEL VERSIONING (frontend + ops)
# ============================================

@router.get("/versions")
async def list_xgboost_versions(limit: int = Query(10, ge=1, le=30)):
    """Lista le versioni del modello XGBoost (WIN/LOSS)."""
    versions = await ml_model.list_versions(limit=limit)
    production = await ml_model.get_production_info()
    return {"family": "xgboost", "production": production, "versions": versions}


@router.get("/trend/versions")
async def list_trend_versions(limit: int = Query(10, ge=1, le=30)):
    """Lista le versioni del modello Trend."""
    versions = await trend_predictor.list_versions(limit=limit)
    production = await trend_predictor.get_production_info()
    return {"family": "trend", "production": production, "versions": versions}


@router.get("/production")
async def get_xgboost_production():
    return await ml_model.get_production_info()


@router.get("/trend/production")
async def get_trend_production():
    return await trend_predictor.get_production_info()


@router.post("/promote/{version_id}")
async def promote_xgboost(version_id: str):
    """Promuove una versione XGBoost a production."""
    return await ml_model.promote(version_id)


@router.post("/trend/promote/{version_id}")
async def promote_trend(version_id: str):
    """Promuove una versione Trend a production."""
    return await trend_predictor.promote(version_id)


@router.post("/rollback")
async def rollback_xgboost():
    """Rollback XGBoost alla previous_version."""
    return await ml_model.rollback()


@router.post("/trend/rollback")
async def rollback_trend():
    """Rollback Trend alla previous_version."""
    return await trend_predictor.rollback()


# ============================================
# v1.1 — ML DEBUG ENDPOINTS (permanent)
# ============================================

@router.get("/debug/ticker/{ticker}")
async def ml_debug_ticker(ticker: str):
    from datetime import datetime
    db = get_db()

    ticker = ticker.upper()
    asset = await db.assets.find_one({"ticker": ticker})

    if not asset:
        return {"error": f"Ticker {ticker} not found in DB"}

    ps = await db.auto_trader.find_one({"_id": "alpaca_state"})
    market_context = ps.get("market", {}) if ps else {}

    prediction = await ml_model.predict(asset, market_context)

    features_used = {
        "rsi": asset.get("rsi", 0),
        "macd_histogram": asset.get("macd", {}).get("histogram", 0),
        "ema10": asset.get("ema10", 0),
        "ema20": asset.get("ema20", 0),
        "ema50": asset.get("ema50", 0),
        "price": asset.get("price", 0),
        "relative_volume": asset.get("relative_volume", 1),
        "setup_score": asset.get("setup_score", 0),
        "poc_price": asset.get("poc_price", 0),
        "value_area_low": asset.get("value_area_low", 0),
        "value_area_high": asset.get("value_area_high", 0),
        "range_position": asset.get("range_position", 50),
        "pct_from_high": asset.get("pct_from_high", -50),
        "change_pct": asset.get("change_pct", 0),
        "accumulation_score": asset.get("accumulation", {}).get("score", 0),
    }

    warnings = []
    for feat_name, feat_val in features_used.items():
        if feat_val is None:
            warnings.append(f"⚠️ {feat_name} is None (missing)")
        elif isinstance(feat_val, (int, float)):
            if feat_val == 0 and feat_name not in ("change_pct", "macd_histogram"):
                warnings.append(f"⚠️ {feat_name} = 0 (sospetto)")

    model_status = await ml_model.get_status()

    return {
        "ticker": ticker,
        "prediction": prediction,
        "features_used": features_used,
        "warnings": warnings,
        "model_info": {
            "accuracy": model_status.get("accuracy", 0),
            "n_samples_trained": model_status.get("n_samples", 0),
            "n_real_positions": model_status.get("n_real_positions", 0),
            "top_features": model_status.get("top_features", {}),
            "current_version": model_status.get("current_version"),
        },
        "diagnosis": _diagnose_prediction(prediction, features_used, warnings),
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/debug/batch/analysis")
async def ml_batch_analysis():
    from datetime import datetime
    import statistics

    db = get_db()

    assets = await db.assets.find({}).to_list(length=250)
    if not assets:
        return {"error": "No assets found"}

    ps = await db.auto_trader.find_one({"_id": "alpaca_state"})
    market_context = ps.get("market", {}) if ps else {}

    predictions = await ml_model.predict_batch(assets, market_context)

    scores = [
        p.get("ml_score", 0)
        for ticker, p in predictions.items()
        if p.get("ml_score") is not None
    ]

    if not scores:
        return {"error": "No valid predictions", "predictions_total": len(predictions)}

    mean_score = statistics.mean(scores)
    median_score = statistics.median(scores)
    stdev_score = statistics.stdev(scores) if len(scores) > 1 else 0
    min_score = min(scores)
    max_score = max(scores)

    buckets = {
        "0-20": 0, "20-40": 0, "40-60": 0, "60-80": 0, "80-100": 0
    }
    for s in scores:
        if s < 20:
            buckets["0-20"] += 1
        elif s < 40:
            buckets["20-40"] += 1
        elif s < 60:
            buckets["40-60"] += 1
        elif s < 80:
            buckets["60-80"] += 1
        else:
            buckets["80-100"] += 1

    is_flat = stdev_score < 5.0
    dominant_bucket = max(buckets, key=buckets.get)
    dominant_pct = round((buckets[dominant_bucket] / len(scores)) * 100, 1)

    from collections import Counter
    rounded_scores = [round(s, 1) for s in scores]
    score_counts = Counter(rounded_scores)
    most_common_score, most_common_count = score_counts.most_common(1)[0]
    duplicate_pct = round((most_common_count / len(scores)) * 100, 1)

    health = "healthy"
    issues = []
    if is_flat:
        health = "critical"
        issues.append(f"🚨 Model output is FLAT (stdev {stdev_score:.1f})")
    if duplicate_pct > 30:
        health = "warning"
        issues.append(f"⚠️ {duplicate_pct}% of tickers have score = {most_common_score}%")
    if dominant_pct > 80:
        health = "warning"
        issues.append(f"⚠️ {dominant_pct}% of tickers in bucket {dominant_bucket}")
    if mean_score > 85 or mean_score < 15:
        health = "warning"
        issues.append(f"⚠️ Mean score {mean_score:.1f}% is extreme")

    if not issues:
        issues.append("✅ Distribution looks normal")

    return {
        "total_predictions": len(scores),
        "statistics": {
            "mean": round(mean_score, 2),
            "median": round(median_score, 2),
            "stdev": round(stdev_score, 2),
            "min": round(min_score, 2),
            "max": round(max_score, 2),
        },
        "distribution": buckets,
        "dominant_bucket": dominant_bucket,
        "dominant_pct": dominant_pct,
        "most_common_score": {
            "value": most_common_score,
            "count": most_common_count,
            "pct": duplicate_pct,
        },
        "health": health,
        "issues": issues,
        "checked_at": datetime.utcnow().isoformat(),
    }


def _diagnose_prediction(prediction, features, warnings):
    diagnosis = []

    ml_score = prediction.get("ml_score", 0)
    confidence = prediction.get("confidence", 0)

    if ml_score > 95:
        diagnosis.append("🚨 Score TROPPO alto (>95%) — probabile overfitting")
    elif ml_score < 5:
        diagnosis.append("🚨 Score TROPPO basso (<5%) — probabile overfitting")

    if confidence > 90 and ml_score > 85:
        diagnosis.append("⚠️ Confidence + Score entrambi altissimi — modello sovraconfidente")

    if len(warnings) > 3:
        diagnosis.append(f"🚨 {len(warnings)} features problematiche — dati incompleti")

    if abs(ml_score - 92.3) < 1.0:
        diagnosis.append("🚨 Score = 92.3% — pattern noto di modello flat")

    if not diagnosis:
        diagnosis.append("✅ Predizione sembra sana")

    return diagnosis


@router.get("/debug/health")
async def ml_health_check():
    from datetime import datetime
    db = get_db()

    health_report = {
        "checked_at": datetime.utcnow().isoformat(),
        "status": "unknown",
        "checks": {},
    }

    status = await ml_model.get_status()
    health_report["checks"]["model_loaded"] = {
        "status": "ok" if status.get("is_trained") else "critical",
        "details": status,
    }

    try:
        assets = await db.assets.find({}).to_list(50)
        ps = await db.auto_trader.find_one({"_id": "alpaca_state"})
        market_context = ps.get("market", {}) if ps else {}
        preds = await ml_model.predict_batch(assets[:20], market_context)

        scores = [p.get("ml_score", 0) for p in preds.values() if p.get("ml_score") is not None]
        if scores:
            import statistics
            stdev = statistics.stdev(scores) if len(scores) > 1 else 0
            health_report["checks"]["predictions_variance"] = {
                "status": "ok" if stdev > 5 else "critical",
                "stdev": round(stdev, 2),
                "message": "Good variance" if stdev > 5 else f"FLAT predictions (stdev {stdev:.2f} < 5)",
            }
    except Exception as e:
        health_report["checks"]["predictions_variance"] = {
            "status": "error",
            "error": str(e),
        }

    n_real = status.get("n_real_positions", 0)
    health_report["checks"]["training_data"] = {
        "status": "ok" if n_real >= 20 else "warning",
        "n_real_positions": n_real,
        "message": (
            f"Good ({n_real} real positions)" if n_real >= 20
            else f"Insufficient ({n_real} real positions, need 20+)"
        ),
    }

    accuracy = status.get("accuracy", 0)
    health_report["checks"]["model_accuracy"] = {
        "status": "ok" if accuracy > 55 else ("warning" if accuracy > 45 else "critical"),
        "accuracy": accuracy,
        "message": (
            "Good" if accuracy > 55 else
            "Marginal (below random+5%)" if accuracy > 45 else
            "Poor (below random)"
        ),
    }

    critical_count = sum(1 for c in health_report["checks"].values() if c.get("status") == "critical")
    warning_count = sum(1 for c in health_report["checks"].values() if c.get("status") == "warning")

    if critical_count > 0:
        health_report["status"] = "critical"
    elif warning_count > 0:
        health_report["status"] = "warning"
    else:
        health_report["status"] = "healthy"

    suggestions = []
    if critical_count > 0:
        suggestions.append("🚨 URGENT: Retrain model with more real data")
    if health_report["checks"].get("training_data", {}).get("n_real_positions", 0) < 20:
        suggestions.append("⏳ Wait for more closed trades before retraining")
    if health_report["checks"].get("model_accuracy", {}).get("accuracy", 0) < 45:
        suggestions.append("🔧 Consider changing feature engineering")

    health_report["suggestions"] = suggestions if suggestions else ["✅ System is healthy"]

    return health_report


@router.post("/collect-backtest-data")
async def collect_backtest_data(
    days: int = 250,
    min_confluence: float = 55,
    stop_loss_pct: float = 6.0,
    take_profit_pct: float = 12.0,
):
    """v2.0 Genera training data da backtest storico."""
    from app.ml.backtest_collector import collect_backtest_training_data
    result = await collect_backtest_training_data(
        days=days,
        min_confluence=min_confluence,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
    )
    return result


@router.post("/train-hybrid")
async def train_hybrid(real_weight: int = 3):
    """v2.0 Training ibrido: real (dedup) + backtest data."""
    result = await ml_model.train_hybrid(real_weight=real_weight)
    return result


@router.post("/weekly-retrain")
async def weekly_retrain(
    days: int = 250,
    min_confluence: float = 55,
    real_weight: int = 3,
    also_train_trend: bool = True,
):
    """
    Cron settimanale (domenica):
    1. Collect backtest data
    2. Train hybrid XGBoost (con versioning automatico)
    3. Opzionalmente train Trend model
    4. Notifica Telegram
    """
    from datetime import datetime
    from app.ml.backtest_collector import collect_backtest_training_data

    report = {"started_at": datetime.utcnow().isoformat()}

    # ===== Step 1: Collect =====
    try:
        collect_result = await collect_backtest_training_data(
            days=days, min_confluence=min_confluence
        )
        report["collect"] = {
            "status": "ok",
            "total_samples": collect_result.get("total_samples", 0),
            "message": collect_result.get("message", ""),
        }
    except Exception as e:
        report["collect"] = {"status": "error", "error": str(e)}
        report["status"] = "failed_at_collect"
        return report

    # ===== Step 2: Train hybrid XGBoost =====
    try:
        train_result = await ml_model.train_hybrid(real_weight=real_weight)
        report["train_xgboost"] = {
            "status": "ok",
            "accuracy": train_result.get("accuracy", 0),
            "n_samples": train_result.get("n_samples", 0),
            "n_real_positions": train_result.get("n_real_positions", 0),
            "version": train_result.get("version"),
            "top_features": train_result.get("top_features", {}),
        }
    except Exception as e:
        report["train_xgboost"] = {"status": "error", "error": str(e)}
        report["status"] = "failed_at_xgboost"
        return report

    # ===== Step 3: Train Trend (opzionale) =====
    if also_train_trend:
        try:
            trend_result = await trend_predictor.train()
            report["train_trend"] = {
                "status": "ok",
                "accuracy": trend_result.get("accuracy", 0),
                "n_samples": trend_result.get("n_samples", 0),
                "version": trend_result.get("version"),
            }
        except Exception as e:
            report["train_trend"] = {"status": "error", "error": str(e)}
            # non blocchiamo tutto se fallisce solo il trend

    report["status"] = "success"

    # ===== Step 4: Telegram report =====
    try:
        from app.services.telegram_bot import send_telegram
        xgb = report.get("train_xgboost", {})
        tr = report.get("train_trend", {})
        msg = (
            f"🧠 <b>ML Weekly Retrain</b>\n\n"
            f"✅ XGBoost Accuracy: {xgb.get('accuracy', '?')}%\n"
            f"📦 Version: {xgb.get('version', '?')}\n"
            f"📊 Samples: {xgb.get('n_samples', '?')} "
            f"({xgb.get('n_real_positions', '?')} reali)\n"
        )
        if tr.get("status") == "ok":
            msg += (
                f"\n📈 Trend Accuracy: {tr.get('accuracy', '?')}%\n"
                f"📦 Trend Version: {tr.get('version', '?')}\n"
            )
        msg += "\nModelli aggiornati e promossi a production."
        await send_telegram(msg)
    except Exception as e:
        report["telegram"] = f"error: {e}"

    report["finished_at"] = datetime.utcnow().isoformat()
    return report
