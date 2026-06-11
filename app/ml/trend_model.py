"""
SwingLab ML — Trend Prediction Model
Predicts 5-day price trend (UP/FLAT/DOWN) for each stock.
Uses historical price data from stock_bars in MongoDB.
"""

import numpy as np
import pandas as pd
import pickle
import base64
from datetime import datetime

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
class TrendPredictor:
    """Predicts 5-day price trend for each stock."""

    def __init__(self):
        self.model = None
        self.is_trained = False
        self.metadata = {}

    def _bars_to_df(self, bars):
        if not bars or len(bars) < 30:
            return None
        df = pd.DataFrame(bars)
        df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
        df = df.sort_values("date").reset_index(drop=True)
        return df

    async def train(self):
        if not HAS_SKLEARN:
            return {"error": "scikit-learn not installed"}
        print("\nTREND MODEL TRAINING")
        print("=" * 50)
        db = get_db()
        stocks = await db.stock_bars.find({}).to_list(length=300)
        print(f"  Found {len(stocks)} stocks with bars")
        all_features = []
        all_labels = []
        stocks_used = 0
        for stock in stocks:
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
        await self.save_to_db()
        print(f"  Model trained! Accuracy: {accuracy}%")
        print(f"  Class accuracy: {class_acc}")
        print(f"  Top features: {list(importance_dict.keys())[:5]}")
        print("=" * 50)
        return {"status": "trained", **self.metadata, "top_features": dict(list(importance_dict.items())[:5])}

    async def predict_stock(self, ticker):
        if not self.is_trained:
            loaded = await self.load_from_db()
            if not loaded:
                return {"ticker": ticker, "prediction": None, "status": "not_trained"}
        try:
            db = get_db()
            stock = await db.stock_bars.find_one({"ticker": ticker.upper()})
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
        stocks = await db.stock_bars.find({}, {"ticker": 1, "bars": 1}).to_list(length=300)
        results = []
        for stock in stocks:
            ticker = stock.get("ticker", "")
            bars = stock.get("bars", [])
            df = self._bars_to_df(bars)
            if df is None or len(df) < 25:
                continue
            features = extract_trend_features(df, len(df) - 1)
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

    async def save_to_db(self):
        if not self.model:
            return False
        try:
            db = get_db()
            model_bytes = pickle.dumps(self.model)
            model_b64 = base64.b64encode(model_bytes).decode("utf-8")
            await db.ml_models.update_one(
                {"_id": "trend_v1"},
                {"$set": {"model_data": model_b64, "metadata": self.metadata, "updated_at": datetime.utcnow().isoformat()}},
                upsert=True,
            )
            return True
        except Exception as e:
            print(f"  Save trend model error: {e}")
            return False

    async def load_from_db(self):
        try:
            db = get_db()
            doc = await db.ml_models.find_one({"_id": "trend_v1"})
            if not doc or "model_data" not in doc:
                return False
            model_bytes = base64.b64decode(doc["model_data"])
            self.model = pickle.loads(model_bytes)
            self.metadata = doc.get("metadata", {})
            self.is_trained = True
            return True
        except Exception as e:
            print(f"  Load trend model error: {e}")
            return False

    async def get_status(self):
        if not self.is_trained:
            await self.load_from_db()
        if not self.is_trained:
            return {"is_trained": False, "status": "not_trained", "message": "Run /api/ml/trend/train"}
        importance = self.metadata.get("feature_importance", {})
        top5 = dict(list(importance.items())[:5])
        return {
            "is_trained": True, "status": "ready",
            "accuracy": self.metadata.get("accuracy", 0),
            "class_accuracy": self.metadata.get("class_accuracy", {}),
            "n_samples": self.metadata.get("n_samples", 0),
            "n_stocks": self.metadata.get("n_stocks", 0),
            "train_date": self.metadata.get("train_date", ""),
            "top_features": top5,
        }


# ---- Singleton ----
trend_predictor = TrendPredictor()
