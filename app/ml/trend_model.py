"""
SwingLab ML — Trend Prediction Model
Predicts 5-day price trend (UP/FLAT/DOWN) for each stock.
Uses historical price data from stock_bars in MongoDB.

v3.0 — Lightweight model versioning (same pattern as SwingLabModel)
- Immutable versioned documents (trend_vN)
- Production pointer (trend_production)
- Automatic promote on train + rollback support
- Fully backward compatible with old trend_v1
"""

import numpy as np
import pandas as pd
import pickle
import base64
from datetime import datetime
from typing import Optional

from app.db.mongodb import get_db

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ============================================
# FEATURE EXTRACTION
# ============================================

def _calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def extract_trend_features(df, idx):
    """
    Extract 20 features from a DataFrame at a given index.
    df must have columns: Open, High, Low, Close, Volume
    idx must be >= 20 (need lookback)
    Returns list of 20 floats, or None if not enough data.
    """
    if idx < 20 or idx >= len(df):
        return None

    try:
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        open_ = df["Open"]
        volume = df["Volume"]

        price = float(close.iloc[idx])
        if price <= 0:
            return None

        # RSI
        rsi_series = _calc_rsi(close)
        rsi = float(rsi_series.iloc[idx]) if not pd.isna(rsi_series.iloc[idx]) else 50
        rsi_5d_ago = float(rsi_series.iloc[idx - 5]) if not pd.isna(rsi_series.iloc[idx - 5]) else 50
        rsi_change = rsi - rsi_5d_ago

        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd_hist = float((ema12 - ema26 - (ema12 - ema26).ewm(span=9).mean()).iloc[idx])

        # EMA distances
        ema10 = float(close.ewm(span=10).mean().iloc[idx])
        ema20 = float(close.ewm(span=20).mean().iloc[idx])
        ema50 = float(close.ewm(span=50).mean().iloc[idx]) if idx >= 50 else ema20
        ema10_dist = ((price - ema10) / price) * 100
        ema20_dist = ((price - ema20) / price) * 100
        ema50_dist = ((price - ema50) / price) * 100

        # EMA alignment
        if price > ema10 > ema20 > ema50:
            ema_align = 2
        elif price > ema20 > ema50:
            ema_align = 1
        else:
            ema_align = 0

        # Volume
        vol_20d = float(volume.iloc[idx - 19:idx + 1].mean())
        vol_5d = float(volume.iloc[idx - 4:idx + 1].mean())
        curr_vol = float(volume.iloc[idx])
        vol_ratio = curr_vol / vol_20d if vol_20d > 0 else 1
        vol_trend = vol_5d / vol_20d if vol_20d > 0 else 1

        # Price changes
        change_1d = ((price / float(close.iloc[idx - 1])) - 1) * 100
        change_5d = ((price / float(close.iloc[idx - 5])) - 1) * 100
        change_10d = ((price / float(close.iloc[idx - 10])) - 1) * 100
        change_20d = ((price / float(close.iloc[idx - 20])) - 1) * 100

        # Range
        high_5d = float(high.iloc[idx - 4:idx + 1].max())
        low_5d = float(low.iloc[idx - 4:idx + 1].min())
        range_5d = ((high_5d - low_5d) / price) * 100

        # Position in range
        high_20d = float(high.iloc[idx - 19:idx + 1].max())
        low_20d = float(low.iloc[idx - 19:idx + 1].min())
        pct_from_20d_high = ((price - high_20d) / high_20d) * 100
        pct_from_20d_low = ((price - low_20d) / low_20d) * 100 if low_20d > 0 else 0

        # Candle body
        o = float(open_.iloc[idx])
        h = float(high.iloc[idx])
        l = float(low.iloc[idx])
        c = float(close.iloc[idx])
        total_range = h - l if h > l else 0.01
        body_pct = ((c - o) / total_range) * 100
        upper_shadow = ((h - max(o, c)) / total_range) * 100
        lower_shadow = ((min(o, c) - l) / total_range) * 100

        # Consecutive up days
        consec = 0
        for j in range(idx, max(idx - 10, 0), -1):
            if float(close.iloc[j]) > float(close.iloc[j - 1]):
                consec += 1
            else:
                break

        return [
            round(rsi, 2), round(rsi_change, 2), round(macd_hist, 4),
            round(ema10_dist, 2), round(ema20_dist, 2), round(ema50_dist, 2), ema_align,
            round(vol_ratio, 2), round(vol_trend, 2),
            round(change_1d, 2), round(change_5d, 2), round(change_10d, 2), round(change_20d, 2),
            round(range_5d, 2), round(pct_from_20d_high, 2), round(pct_from_20d_low, 2),
            round(body_pct, 2), round(upper_shadow, 2), round(lower_shadow, 2),
            consec,
        ]
    except Exception:
        return None


TREND_FEATURE_NAMES = [
    "rsi", "rsi_change_5d", "macd_histogram",
    "ema10_dist", "ema20_dist", "ema50_dist", "ema_alignment",
    "volume_ratio", "volume_trend",
    "change_1d", "change_5d", "change_10d", "change_20d",
    "range_5d", "pct_from_20d_high", "pct_from_20d_low",
    "body_pct", "upper_shadow", "lower_shadow",
    "consecutive_up_days",
]

TREND_LABELS = {0: "DOWN", 1: "FLAT", 2: "UP"}

# ==========================================================
# FINESTRE DI CARICAMENTO — controllo della memoria
# ==========================================================
PREDICT_BARS = 120
TRAIN_BARS = 200


class TrendPredictor:
    """Predicts 5-day price trend for each stock with lightweight versioning."""

    MODEL_FAMILY = "trend"
    PRODUCTION_ID = "trend_production"
    LEGACY_ID = "trend_v1"
    MAX_VERSIONS = 10

    def __init__(self):
        self.model = None
        self.is_trained = False
        self.metadata = {}
        self.current_version_id: Optional[str] = None

    def _bars_to_df(self, bars):
        if not bars or len(bars) < 30:
            return None
        df = pd.DataFrame(bars)
        df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
        df = df.sort_values("date").reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Version helpers (same pattern as SwingLabModel)
    # ------------------------------------------------------------------

    async def _next_version_number(self) -> int:
        db = get_db()
        cursor = db.ml_models.find(
            {"_id": {"$regex": f"^{self.MODEL_FAMILY}_v\\d+$"}},
            {"_id": 1},
        )
        max_v = 0
        async for doc in cursor:
            try:
                num = int(doc["_id"].split("_v")[-1])
                if num > max_v:
                    max_v = num
            except (ValueError, IndexError):
                continue
        return max_v + 1

    async def _get_production_pointer(self) -> dict:
        db = get_db()
        doc = await db.ml_models.find_one({"_id": self.PRODUCTION_ID})
        return doc or {}

    async def _set_production(self, version_id: str, previous: Optional[str] = None):
        db = get_db()
        now = datetime.utcnow().isoformat()
        pointer = await self._get_production_pointer()
        old_active = pointer.get("active_version")

        await db.ml_models.update_one(
            {"_id": self.PRODUCTION_ID},
            {"$set": {
                "active_version": version_id,
                "previous_version": previous if previous is not None else old_active,
                "promoted_at": now,
                "promoted_by": "train",
                "updated_at": now,
            }},
            upsert=True,
        )

        if old_active and old_active != version_id:
            await db.ml_models.update_one(
                {"_id": old_active},
                {"$set": {"status": "archived"}},
            )
        await db.ml_models.update_one(
            {"_id": version_id},
            {"$set": {"status": "production"}},
        )

    async def _prune_old_versions(self):
        db = get_db()
        cursor = db.ml_models.find(
            {"_id": {"$regex": f"^{self.MODEL_FAMILY}_v\\d+$"}},
            {"_id": 1, "created_at": 1},
        ).sort("created_at", -1)
        docs = await cursor.to_list(length=50)
        if len(docs) <= self.MAX_VERSIONS:
            return
        to_delete = [d["_id"] for d in docs[self.MAX_VERSIONS:]]
        pointer = await self._get_production_pointer()
        protected = {
            pointer.get("active_version"),
            pointer.get("previous_version"),
            self.LEGACY_ID,
        }
        to_delete = [vid for vid in to_delete if vid not in protected]
        if to_delete:
            await db.ml_models.delete_many({"_id": {"$in": to_delete}})
            print(f"  Pruned {len(to_delete)} old trend model versions")

    # ------------------------------------------------------------------
    # Core save / load (backward compatible)
    # ------------------------------------------------------------------

    async def save_to_db(self, promote: bool = True) -> dict:
        if not self.model:
            return {"saved": False, "error": "no model"}

        try:
            db = get_db()
            version_num = await self._next_version_number()
            version_id = f"{self.MODEL_FAMILY}_v{version_num}"

            model_bytes = pickle.dumps(self.model)
            model_b64 = base64.b64encode(model_bytes).decode("utf-8")
            now = datetime.utcnow().isoformat()

            doc = {
                "model_family": self.MODEL_FAMILY,
                "version": version_num,
                "status": "staging",
                "model_data": model_b64,
                "metadata": self.metadata,
                "created_at": now,
                "created_by": self.metadata.get("model_type", "train"),
            }
            await db.ml_models.update_one(
                {"_id": version_id},
                {"$set": doc},
                upsert=True,
            )

            # Keep legacy document updated for compatibility
            await db.ml_models.update_one(
                {"_id": self.LEGACY_ID},
                {"$set": {
                    "model_data": model_b64,
                    "metadata": self.metadata,
                    "updated_at": now,
                    "mirrored_from": version_id,
                }},
                upsert=True,
            )

            result = {
                "saved": True,
                "version_id": version_id,
                "version": version_num,
            }

            if promote:
                await self._set_production(version_id)
                self.current_version_id = version_id
                result["promoted"] = True
                result["status"] = "production"
                print(f"  Trend model saved as {version_id} and promoted to production")
            else:
                print(f"  Trend model saved as {version_id} (staging)")

            await self._prune_old_versions()
            return result

        except Exception as e:
            print(f"  Save trend model error: {e}")
            return {"saved": False, "error": str(e)}

    async def load_from_db(self) -> bool:
        try:
            db = get_db()

            pointer = await self._get_production_pointer()
            active = pointer.get("active_version")

            doc = None
            if active:
                doc = await db.ml_models.find_one({"_id": active})

            if not doc or "model_data" not in doc:
                doc = await db.ml_models.find_one({"_id": self.LEGACY_ID})

            if not doc or "model_data" not in doc:
                return False

            model_bytes = base64.b64decode(doc["model_data"])
            self.model = pickle.loads(model_bytes)
            self.metadata = doc.get("metadata", {})
            self.is_trained = True
            self.current_version_id = doc.get("_id")
            print(f"  Trend model loaded from MongoDB ({self.current_version_id})")
            return True

        except Exception as e:
            print(f"  Load trend model error: {e}")
            return False

    # ------------------------------------------------------------------
    # Version management API
    # ------------------------------------------------------------------

    async def list_versions(self, limit: int = 10) -> list:
        db = get_db()
        cursor = db.ml_models.find(
            {"_id": {"$regex": f"^{self.MODEL_FAMILY}_v\\d+$"}},
            {"model_data": 0},
        ).sort("created_at", -1).limit(limit)
        docs = await cursor.to_list(length=limit)

        pointer = await self._get_production_pointer()
        active = pointer.get("active_version")
        previous = pointer.get("previous_version")

        result = []
        for d in docs:
            d["_id"] = str(d["_id"])
            d["is_production"] = d["_id"] == active
            d["is_previous"] = d["_id"] == previous
            result.append(d)
        return result

    async def promote(self, version_id: str) -> dict:
        db = get_db()
        doc = await db.ml_models.find_one({"_id": version_id})
        if not doc or "model_data" not in doc:
            return {"promoted": False, "error": f"version {version_id} not found"}

        await self._set_production(version_id)
        return {
            "promoted": True,
            "version_id": version_id,
            "message": f"{version_id} is now production",
        }

    async def rollback(self) -> dict:
        pointer = await self._get_production_pointer()
        previous = pointer.get("previous_version")
        if not previous:
            return {"rolled_back": False, "error": "no previous version recorded"}

        current = pointer.get("active_version")
        await self._set_production(previous, previous=current)
        return {
            "rolled_back": True,
            "from": current,
            "to": previous,
            "message": f"Rolled back to {previous}",
        }

    async def get_production_info(self) -> dict:
        pointer = await self._get_production_pointer()
        active = pointer.get("active_version")
        if not active:
            return {"active_version": self.LEGACY_ID, "legacy": True}

        db = get_db()
        doc = await db.ml_models.find_one(
            {"_id": active},
            {"model_data": 0},
        )
        if not doc:
            return {"active_version": active, "error": "document missing"}

        return {
            "active_version": active,
            "previous_version": pointer.get("previous_version"),
            "promoted_at": pointer.get("promoted_at"),
            "metadata": doc.get("metadata", {}),
            "status": doc.get("status"),
            "created_at": doc.get("created_at"),
        }

    # ------------------------------------------------------------------
    # Training & prediction (logic unchanged)
    # ------------------------------------------------------------------

    async def train(self):
        if not HAS_SKLEARN:
            return {"error": "scikit-learn not installed"}
        print("\nTREND MODEL TRAINING")
        print("=" * 50)
        db = get_db()
        all_features = []
        all_labels = []
        stocks_used = 0
        stocks_seen = 0
        cursor = db.stock_bars.find(
            {},
            {"ticker": 1, "bars": {"$slice": -TRAIN_BARS}},
        )
        async for stock in cursor:
            stocks_seen += 1
            bars = stock.get("bars", [])
            df = self._bars_to_df(bars)
            if df is None or len(df) < 50:
                continue
            stocks_used += 1
            close = df["Close"]
            for idx in range(max(20, len(df) - 60), len(df) - 5):
                features = extract_trend_features(df, idx)
                if features is None:
                    continue
                price_now = float(close.iloc[idx])
                price_future = float(close.iloc[idx + 5])
                if price_now <= 0:
                    continue
                change = ((price_future - price_now) / price_now) * 100
                if change > 2:
                    label = 2
                elif change < -2:
                    label = 0
                else:
                    label = 1
                all_features.append(features)
                all_labels.append(label)
            del df
        print(f"  Found {stocks_seen} stocks with bars")
        print(f"  Stocks used: {stocks_used}")
        print(f"  Total samples: {len(all_features)}")
        if len(all_features) < 100:
            return {"error": f"Not enough data: {len(all_features)} samples (need 100+)"}
        X = np.array(all_features)
        y = np.array(all_labels)
        up_count = int(np.sum(y == 2))
        flat_count = int(np.sum(y == 1))
        down_count = int(np.sum(y == 0))
        print(f"  Classes: UP={up_count}, FLAT={flat_count}, DOWN={down_count}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        if HAS_XGB:
            self.model = xgb.XGBClassifier(
                n_estimators=150, max_depth=5, learning_rate=0.1,
                use_label_encoder=False, eval_metric="mlogloss", random_state=42,
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42,
            )
        print("  Training model...")
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = round(accuracy_score(y_test, y_pred) * 100, 1)
        class_acc = {}
        for cls in [0, 1, 2]:
            mask = y_test == cls
            if mask.sum() > 0:
                class_acc[TREND_LABELS[cls]] = round(accuracy_score(y_test[mask], y_pred[mask]) * 100, 1)
        importances = self.model.feature_importances_
        importance_dict = {
            name: round(float(imp), 4)
            for name, imp in sorted(zip(TREND_FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)
        }
        self.is_trained = True
        self.metadata = {
            "accuracy": accuracy, "class_accuracy": class_acc,
            "n_samples": len(X), "n_stocks": stocks_used,
            "class_distribution": {"UP": up_count, "FLAT": flat_count, "DOWN": down_count},
            "train_date": datetime.utcnow().isoformat(),
            "feature_importance": importance_dict,
            "model_type": "xgboost" if HAS_XGB else "sklearn_gb",
        }
        save_res = await self.save_to_db(promote=True)
        print(f"  Model trained! Accuracy: {accuracy}%")
        print(f"  Class accuracy: {class_acc}")
        print(f"  Top features: {list(importance_dict.keys())[:5]}")
        print("=" * 50)
        return {
            "status": "trained",
            **self.metadata,
            "top_features": dict(list(importance_dict.items())[:5]),
            "version": save_res.get("version_id"),
        }

    async def predict_stock(self, ticker):
        if not self.is_trained:
            loaded = await self.load_from_db()
            if not loaded:
                return {"ticker": ticker, "prediction": None, "status": "not_trained"}
        try:
            db = get_db()
            stock = await db.stock_bars.find_one(
                {"ticker": ticker.upper()},
                {"ticker": 1, "bars": {"$slice": -PREDICT_BARS}},
            )
            if not stock or not stock.get("bars"):
                return {"ticker": ticker, "prediction": None, "status": "no_data"}
            df = self._bars_to_df(stock["bars"])
            if df is None or len(df) < 25:
                return {"ticker": ticker, "prediction": None, "status": "insufficient_data"}
            features = extract_trend_features(df, len(df) - 1)
            if features is None:
                return {"ticker": ticker, "prediction": None, "status": "feature_error"}
            X = np.array([features])
            probs = self.model.predict_proba(X)[0]
            down_prob = float(probs[0]) if len(probs) > 0 else 0
            flat_prob = float(probs[1]) if len(probs) > 1 else 0
            up_prob = float(probs[2]) if len(probs) > 2 else 0
            pred_class = int(np.argmax(probs))
            prediction = TREND_LABELS.get(pred_class, "FLAT")
            confidence = round(float(np.max(probs)) * 100, 1)
            return {
                "ticker": ticker.upper(), "prediction": prediction,
                "up_prob": round(up_prob * 100, 1), "flat_prob": round(flat_prob * 100, 1),
                "down_prob": round(down_prob * 100, 1), "confidence": confidence, "status": "ok",
            }
        except Exception as e:
            return {"ticker": ticker, "prediction": None, "status": "error", "error": str(e)}

    async def predict_all(self):
        if not self.is_trained:
            loaded = await self.load_from_db()
            if not loaded:
                return []
        db = get_db()
        results = []
        cursor = db.stock_bars.find(
            {},
            {"ticker": 1, "bars": {"$slice": -PREDICT_BARS}},
        )
        async for stock in cursor:
            ticker = stock.get("ticker", "")
            bars = stock.get("bars", [])
            df = self._bars_to_df(bars)
            if df is None or len(df) < 25:
                continue
            features = extract_trend_features(df, len(df) - 1)
            del df
            if features is None:
                continue
            try:
                X = np.array([features])
                probs = self.model.predict_proba(X)[0]
                down_prob = float(probs[0]) if len(probs) > 0 else 0
                flat_prob = float(probs[1]) if len(probs) > 1 else 0
                up_prob = float(probs[2]) if len(probs) > 2 else 0
                pred_class = int(np.argmax(probs))
                results.append({
                    "ticker": ticker, "prediction": TREND_LABELS.get(pred_class, "FLAT"),
                    "up_prob": round(up_prob * 100, 1), "flat_prob": round(flat_prob * 100, 1),
                    "down_prob": round(down_prob * 100, 1), "confidence": round(float(np.max(probs)) * 100, 1),
                })
            except Exception:
                continue
        results.sort(key=lambda x: x["up_prob"], reverse=True)
        return results

    async def get_status(self):
        if not self.is_trained:
            await self.load_from_db()
        if not self.is_trained:
            return {"is_trained": False, "status": "not_trained", "message": "Run /api/ml/trend/train"}
        importance = self.metadata.get("feature_importance", {})
        top5 = dict(list(importance.items())[:5])
        prod_info = await self.get_production_info()
        return {
            "is_trained": True, "status": "ready",
            "accuracy": self.metadata.get("accuracy", 0),
            "class_accuracy": self.metadata.get("class_accuracy", {}),
            "n_samples": self.metadata.get("n_samples", 0),
            "n_stocks": self.metadata.get("n_stocks", 0),
            "train_date": self.metadata.get("train_date", ""),
            "top_features": top5,
            "current_version": self.current_version_id or prod_info.get("active_version"),
            "production": prod_info,
        }


# ---- Singleton ----
trend_predictor = TrendPredictor()
