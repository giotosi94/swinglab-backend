"""
SwingLab ML — XGBoost Model
Trains on trade history, predicts WIN/LOSS probability for new candidates.
"""

import numpy as np
import pickle
import base64
from datetime import datetime

from app.db.mongodb import get_db
from app.ml.features import (
    extract_features_from_asset,
    extract_features_from_trade,
    features_to_array,
    get_feature_names,
)

# Graceful XGBoost import
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️ XGBoost not installed. Using sklearn fallback.")

# Fallback: sklearn
try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️ scikit-learn not installed. ML features disabled.")


class SwingLabModel:
    """XGBoost classifier for trade outcome prediction."""

    def __init__(self):
        self.model = None
        self.is_trained = False
        self.metadata = {}

    async def train(self, use_synthetic_if_needed=True):
        """Train the model on trade history."""
        if not HAS_SKLEARN:
            return {"error": "scikit-learn not installed"}

        print("\n🧠 ML MODEL TRAINING")
        print("=" * 50)

        db = get_db()

        # Fetch closed trades
        trades = await db.trade_history.find(
            {"side": "sell", "pnl_pct": {"$exists": True}}
        ).to_list(length=5000)

        print(f"  📊 Found {len(trades)} real trades")

        features_list = []
        labels = []

        if len(trades) >= 15:
            for t in trades:
                try:
                    feats = extract_features_from_trade(t)
                    arr = features_to_array(feats)
                    label = 1 if (t.get("pnl_pct", 0) > 0) else 0
                    features_list.append(arr)
                    labels.append(label)
                except Exception as e:
                    print(f"  ⚠️ Skip trade: {e}")
                    continue
            print(f"  ✅ Extracted {len(features_list)} feature vectors from real trades")

        elif use_synthetic_if_needed:
            print("  📦 Not enough real trades, generating synthetic data...")
            syn_features, syn_labels = await self.generate_synthetic_data(300)
            features_list = syn_features
            labels = syn_labels
            print(f"  ✅ Generated {len(features_list)} synthetic samples")

        else:
            return {"error": f"Not enough data: {len(trades)} trades (need 15+)"}

        if len(features_list) < 15:
            return {"error": f"Not enough valid data: {len(features_list)} samples"}

        X = np.array(features_list)
        y = np.array(labels)

        print(f"  📊 Dataset: {len(X)} samples, {sum(y)} wins, {len(y) - sum(y)} losses")
        print(f"  📊 Win rate: {sum(y) / len(y) * 100:.1f}%")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if len(set(y)) > 1 else None
        )

        if HAS_XGB:
            self.model = xgb.XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                use_label_encoder=False, eval_metric="logloss", random_state=42,
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42,
            )

        print("  🔄 Training model...")
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        accuracy = round(accuracy_score(y_test, y_pred) * 100, 1)
        precision = round(precision_score(y_test, y_pred, zero_division=0) * 100, 1)
        recall = round(recall_score(y_test, y_pred, zero_division=0) * 100, 1)

        importances = self.model.feature_importances_
        feature_names = get_feature_names()
        importance_dict = {
            name: round(float(imp), 4)
            for name, imp in sorted(
                zip(feature_names, importances), key=lambda x: x[1], reverse=True,
            )
        }

        self.is_trained = True
        self.metadata = {
            "accuracy": accuracy, "precision": precision, "recall": recall,
            "n_samples": len(X), "n_real_trades": len(trades),
            "n_wins": int(sum(y)), "n_losses": int(len(y) - sum(y)),
            "train_date": datetime.utcnow().isoformat(),
            "feature_importance": importance_dict,
            "model_type": "xgboost" if HAS_XGB else "sklearn_gb",
        }

        await self.save_to_db()

        print(f"  ✅ Model trained! Accuracy: {accuracy}%")
        print(f"  📊 Precision: {precision}%, Recall: {recall}%")
        print(f"  🏆 Top features: {list(importance_dict.keys())[:5]}")
        print("=" * 50)

        return {"status": "trained", **self.metadata, "top_features": dict(list(importance_dict.items())[:5])}
      async def predict(self, asset, market_context=None):
        """Predict WIN probability for a single asset."""
        if not self.is_trained:
            loaded = await self.load_from_db()
            if not loaded:
                return {"ml_score": None, "status": "not_trained"}

        try:
            features = extract_features_from_asset(asset, market_context)
            arr = np.array([features_to_array(features)])
            prob = self.model.predict_proba(arr)[0]
            win_prob = float(prob[1]) if len(prob) > 1 else float(prob[0])
            return {
                "ml_score": round(win_prob * 100, 1),
                "prediction": "WIN" if win_prob > 0.5 else "LOSS",
                "confidence": round(abs(win_prob - 0.5) * 200, 1),
                "status": "ok",
            }
        except Exception as e:
            print(f"  ⚠️ ML predict error: {e}")
            return {"ml_score": None, "status": "error", "error": str(e)}

    async def predict_batch(self, assets, market_context=None):
        """Predict for multiple assets at once."""
        if not self.is_trained:
            loaded = await self.load_from_db()
            if not loaded:
                return {}

        results = {}
        try:
            feature_arrays = []
            tickers = []
            for asset in assets:
                features = extract_features_from_asset(asset, market_context)
                feature_arrays.append(features_to_array(features))
                tickers.append(asset.get("ticker", "?"))

            if not feature_arrays:
                return {}

            X = np.array(feature_arrays)
            probs = self.model.predict_proba(X)

            for i, ticker in enumerate(tickers):
                win_prob = float(probs[i][1]) if probs[i].shape[0] > 1 else float(probs[i][0])
                results[ticker] = {
                    "ml_score": round(win_prob * 100, 1),
                    "prediction": "WIN" if win_prob > 0.5 else "LOSS",
                    "confidence": round(abs(win_prob - 0.5) * 200, 1),
                }
        except Exception as e:
            print(f"  ⚠️ ML batch predict error: {e}")

        return results

    async def save_to_db(self):
        """Save trained model to MongoDB."""
        if not self.model:
            return False
        try:
            db = get_db()
            model_bytes = pickle.dumps(self.model)
            model_b64 = base64.b64encode(model_bytes).decode("utf-8")
            await db.ml_models.update_one(
                {"_id": "xgboost_v1"},
                {"$set": {
                    "model_data": model_b64,
                    "metadata": self.metadata,
                    "updated_at": datetime.utcnow().isoformat(),
                }},
                upsert=True,
            )
            print("  💾 Model saved to MongoDB")
            return True
        except Exception as e:
            print(f"  ⚠️ Save model error: {e}")
            return False

    async def load_from_db(self):
        """Load model from MongoDB."""
        try:
            db = get_db()
            doc = await db.ml_models.find_one({"_id": "xgboost_v1"})
            if not doc or "model_data" not in doc:
                return False
            model_bytes = base64.b64decode(doc["model_data"])
            self.model = pickle.loads(model_bytes)
            self.metadata = doc.get("metadata", {})
            self.is_trained = True
            print("  📦 Model loaded from MongoDB")
            return True
        except Exception as e:
            print(f"  ⚠️ Load model error: {e}")
            return False

    async def get_status(self):
        """Return model status."""
        if not self.is_trained:
            await self.load_from_db()

        if not self.is_trained:
            return {
                "is_trained": False,
                "status": "not_trained",
                "message": "Run /api/ml/train to train the model",
            }

        importance = self.metadata.get("feature_importance", {})
        top5 = dict(list(importance.items())[:5])

        return {
            "is_trained": True,
            "status": "ready",
            "accuracy": self.metadata.get("accuracy", 0),
            "precision": self.metadata.get("precision", 0),
            "recall": self.metadata.get("recall", 0),
            "n_samples": self.metadata.get("n_samples", 0),
            "n_real_trades": self.metadata.get("n_real_trades", 0),
            "train_date": self.metadata.get("train_date", ""),
            "model_type": self.metadata.get("model_type", "unknown"),
            "top_features": top5,
        }

    async def generate_synthetic_data(self, n_samples=300):
        """
        Generate synthetic training data from current assets.
        Used for bootstrapping when no real trades exist.
        """
        import random
        db = get_db()

        assets = await db.assets.find({}).to_list(length=300)
        if not assets:
            return [], []

        features_list = []
        labels = []

        for _ in range(n_samples):
            asset = random.choice(assets)
            feats = extract_features_from_asset(asset)

            # Add noise
            for key in ["rsi", "relative_volume", "change_pct", "confluence_score"]:
                if key in feats:
                    feats[key] += random.gauss(0, feats[key] * 0.1 + 0.5)

            arr = features_to_array(feats)

            # Simulate outcome based on indicators
            win_prob = 0.45
            rsi = feats.get("rsi", 50)
            if 35 <= rsi <= 60:
                win_prob += 0.1
            if feats.get("ema_alignment", 0) == 2:
                win_prob += 0.12
            if feats.get("relative_volume", 1) >= 1.5:
                win_prob += 0.05
            if feats.get("poc_distance_pct", 50) <= 3:
                win_prob += 0.08
            if feats.get("confluence_score", 0) >= 65:
                win_prob += 0.1
            if feats.get("setup_type_encoded", 5) in [0, 2]:
                win_prob += 0.05
            if feats.get("regime_encoded", 1) == 0:
                win_prob += 0.05
            if feats.get("has_bullish_patterns", 0) == 1:
                win_prob += 0.05

            win_prob = min(max(win_prob, 0.1), 0.9)
            label = 1 if random.random() < win_prob else 0

            features_list.append(arr)
            labels.append(label)

        return features_list, labels


# ---- Singleton ----
ml_model = SwingLabModel()
