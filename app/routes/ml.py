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
