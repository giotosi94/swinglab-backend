"""
SwingLab ML — API Routes
"""

from fastapi import APIRouter
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
# 🆕 v1.1 — ML DEBUG ENDPOINTS (permanent)
# ============================================

@router.get("/debug/ticker/{ticker}")
async def ml_debug_ticker(ticker: str):
    """
    🔧 Debug dettagliato per un singolo ticker.
    
    Mostra:
    - Le feature esatte usate dal modello
    - Score dettagliato con breakdown
    - Feature importance per questa predizione
    - Comparison con altri ticker recenti
    - Warning se qualcosa sembra sbagliato
    """
    from datetime import datetime
    db = get_db()
    
    ticker = ticker.upper()
    asset = await db.assets.find_one({"ticker": ticker})
    
    if not asset:
        return {"error": f"Ticker {ticker} not found in DB"}
    
    # Recupera market context
    ps = await db.auto_trader.find_one({"_id": "alpaca_state"})
    market_context = ps.get("market", {}) if ps else {}
    
    # Predizione
    prediction = await ml_model.predict(asset, market_context)
    
    # Estrai feature dal codice del modello
    # (Ricostruisce le feature che il modello vede)
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
    
    # Check NaN o valori sospetti
    warnings = []
    for feat_name, feat_val in features_used.items():
        if feat_val is None:
            warnings.append(f"⚠️ {feat_name} is None (missing)")
        elif isinstance(feat_val, (int, float)):
            if feat_val == 0 and feat_name not in ("change_pct", "macd_histogram"):
                warnings.append(f"⚠️ {feat_name} = 0 (sospetto)")
    
    # Model status
    model_status = await ml_model.get_status()
    
    return {
        "ticker": ticker,
        "prediction": prediction,
        "features_used": features_used,
        "warnings": warnings,
        "model_info": {
            "accuracy": model_status.get("accuracy", 0),
            "n_samples_trained": model_status.get("n_samples", 0),
            "n_real_trades": model_status.get("n_real_trades", 0),
            "top_features": model_status.get("top_features", {}),
        },
        "diagnosis": _diagnose_prediction(prediction, features_used, warnings),
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/debug/batch/analysis")
async def ml_batch_analysis():
    """
    🔧 Analisi statistica di TUTTE le predizioni ML.
    
    Rileva problemi tipo:
    - Predizioni "flat" (tutti uguali)
    - Varianza troppo bassa
    - Ticker con feature mancanti
    - Distribuzione anomala degli score
    """
    from datetime import datetime
    import statistics
    
    db = get_db()
    
    assets = await db.assets.find({}).to_list(length=250)
    if not assets:
        return {"error": "No assets found"}
    
    ps = await db.auto_trader.find_one({"_id": "alpaca_state"})
    market_context = ps.get("market", {}) if ps else {}
    
    predictions = await ml_model.predict_batch(assets, market_context)
    
    # Estrai scores
    scores = [
        p.get("ml_score", 0) 
        for ticker, p in predictions.items() 
        if p.get("ml_score") is not None
    ]
    
    if not scores:
        return {"error": "No valid predictions", "predictions_total": len(predictions)}
    
    # Statistiche
    mean_score = statistics.mean(scores)
    median_score = statistics.median(scores)
    stdev_score = statistics.stdev(scores) if len(scores) > 1 else 0
    min_score = min(scores)
    max_score = max(scores)
    
    # Distribution buckets
    buckets = {
        "0-20": 0, "20-40": 0, "40-60": 0, "60-80": 0, "80-100": 0
    }
    for s in scores:
        if s < 20: buckets["0-20"] += 1
        elif s < 40: buckets["20-40"] += 1
        elif s < 60: buckets["40-60"] += 1
        elif s < 80: buckets["60-80"] += 1
        else: buckets["80-100"] += 1
    
    # Detect "flat" pattern
    is_flat = stdev_score < 5.0  # troppo bassa varianza
    dominant_bucket = max(buckets, key=buckets.get)
    dominant_pct = round((buckets[dominant_bucket] / len(scores)) * 100, 1)
    
    # Detect ticker uguali (esempio: molti a 92.3%)
    from collections import Counter
    rounded_scores = [round(s, 1) for s in scores]
    score_counts = Counter(rounded_scores)
    most_common_score, most_common_count = score_counts.most_common(1)[0]
    duplicate_pct = round((most_common_count / len(scores)) * 100, 1)
    
    # Health assessment
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
    """Diagnostica intelligente della predizione."""
    diagnosis = []
    
    ml_score = prediction.get("ml_score", 0)
    ml_pred = prediction.get("prediction", "unknown")
    confidence = prediction.get("confidence", 0)
    
    # Check score troppo alto/basso
    if ml_score > 95:
        diagnosis.append("🚨 Score TROPPO alto (>95%) — probabile overfitting")
    elif ml_score < 5:
        diagnosis.append("🚨 Score TROPPO basso (<5%) — probabile overfitting")
    
    # Check confidence sospetta
    if confidence > 90 and ml_score > 85:
        diagnosis.append("⚠️ Confidence + Score entrambi altissimi — modello sovraconfidente")
    
    # Check warnings features
    if len(warnings) > 3:
        diagnosis.append(f"🚨 {len(warnings)} features problematiche — dati incompleti")
    
    # Score = 92.3% detection
    if abs(ml_score - 92.3) < 1.0:
        diagnosis.append("🚨 Score = 92.3% — pattern noto di modello flat")
    
    if not diagnosis:
        diagnosis.append("✅ Predizione sembra sana")
    
    return diagnosis


@router.get("/debug/health")
async def ml_health_check():
    """
    🔧 Health check completo del sistema ML.
    
    Verifica:
    - Modello caricato
    - Predizioni funzionanti
    - Distribuzione score
    - Feature quality
    - Training data quality
    """
    from datetime import datetime
    db = get_db()
    
    health_report = {
        "checked_at": datetime.utcnow().isoformat(),
        "status": "unknown",
        "checks": {},
    }
    
    # Check 1: Model status
    status = await ml_model.get_status()
    health_report["checks"]["model_loaded"] = {
        "status": "ok" if status.get("is_trained") else "critical",
        "details": status,
    }
    
    # Check 2: Batch analysis
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
    
    # Check 3: Training data quality
    n_real = status.get("n_real_trades", 0)
    health_report["checks"]["training_data"] = {
        "status": "ok" if n_real >= 20 else "warning",
        "n_real_trades": n_real,
        "message": (
            f"Good ({n_real} real trades)" if n_real >= 20
            else f"Insufficient ({n_real} real trades, need 20+)"
        ),
    }
    
    # Check 4: Accuracy
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
    
    # Overall status
    critical_count = sum(1 for c in health_report["checks"].values() if c.get("status") == "critical")
    warning_count = sum(1 for c in health_report["checks"].values() if c.get("status") == "warning")
    
    if critical_count > 0:
        health_report["status"] = "critical"
    elif warning_count > 0:
        health_report["status"] = "warning"
    else:
        health_report["status"] = "healthy"
    
    # Suggestions
    suggestions = []
    if critical_count > 0:
        suggestions.append("🚨 URGENT: Retrain model with more real data")
    if health_report["checks"].get("training_data", {}).get("n_real_trades", 0) < 20:
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
    from app.ml.model import ml_model
    result = await ml_model.train_hybrid(real_weight=real_weight)
    return result
