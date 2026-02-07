import logging
import pickle
from pathlib import Path
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains and manages gradient boosting models for price prediction."""

    def __init__(
        self, model_type="lightgbm", model_dir="./models/saved", training_windows=1000
    ):
        """
        Initialize model trainer.

        Args:
            model_type: Type of model (lightgbm, xgboost, catboost)
            model_dir: Directory to save trained models
            training_windows: Number of windows to use for training
        """
        self.model_type = model_type.lower()
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.training_windows = training_windows

        self.model = None
        self.feature_names = None
        self.training_history = []

        self._import_model_library()

        logger.info(f"Initialized {model_type} model trainer")

    def _import_model_library(self):
        """Import the appropriate gradient boosting library."""
        try:
            if self.model_type == "lightgbm":
                import lightgbm as lgb

                self.lgb = lgb
            elif self.model_type == "xgboost":
                import xgboost as xgb

                self.xgb = xgb
            elif self.model_type == "catboost":
                from catboost import CatBoostClassifier

                self.CatBoostClassifier = CatBoostClassifier
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
        except ImportError as e:
            logger.error(f"Failed to import {self.model_type}: {e}")
            logger.info(f"Install with: pip install {self.model_type}")
            raise

    def train(self, X, y, validation_split=0.2):
        """
        Train the model on provided data.

        Args:
            X: Feature data (list of dicts or 2D array)
            y: Labels (0 or 1 for down/up)
            validation_split: Fraction of data to use for validation

        Returns:
            dict: Training metrics
        """
        if len(X) < 10:
            logger.warning(f"Insufficient training data: {len(X)} samples")
            return None

        X_array, feature_names = self._prepare_features(X)
        y_array = np.array(y)

        self.feature_names = feature_names

        split_idx = int(len(X_array) * (1 - validation_split))
        X_train, X_val = X_array[:split_idx], X_array[split_idx:]
        y_train, y_val = y_array[:split_idx], y_array[split_idx:]

        logger.info(
            f"Training with {len(X_train)} samples, validating with {len(X_val)} samples"
        )

        metrics = {}

        if self.model_type == "lightgbm":
            metrics = self._train_lightgbm(X_train, y_train, X_val, y_val)
        elif self.model_type == "xgboost":
            metrics = self._train_xgboost(X_train, y_train, X_val, y_val)
        elif self.model_type == "catboost":
            metrics = self._train_catboost(X_train, y_train, X_val, y_val)

        self.training_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "n_samples": len(X_train),
                "metrics": metrics,
            }
        )

        logger.info(
            f"Training complete. Validation accuracy: {metrics.get('val_accuracy', 0):.4f}"
        )

        return metrics

    def _train_lightgbm(self, X_train, y_train, X_val, y_val):
        """Train LightGBM model."""
        train_data = self.lgb.Dataset(X_train, label=y_train)
        val_data = self.lgb.Dataset(X_val, label=y_val, reference=train_data)

        params = {
            "objective": "binary",
            "metric": ["binary_logloss", "auc"],
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "min_data_in_leaf": 20,
            "min_gain_to_split": 0.01,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
        }

        evals_result = {}
        self.model = self.lgb.train(
            params,
            train_data,
            num_boost_round=200,
            valid_sets=[train_data, val_data],
            valid_names=["train", "valid"],
            callbacks=[
                self.lgb.early_stopping(stopping_rounds=20),
                self.lgb.log_evaluation(period=50),
                self.lgb.record_evaluation(evals_result),
            ],
        )

        y_pred_train = (self.model.predict(X_train) > 0.5).astype(int)
        y_pred_val = (self.model.predict(X_val) > 0.5).astype(int)

        return {
            "train_accuracy": np.mean(y_pred_train == y_train),
            "val_accuracy": np.mean(y_pred_val == y_val),
            "best_iteration": self.model.best_iteration,
            "final_train_loss": evals_result["train"]["binary_logloss"][-1],
            "final_val_loss": evals_result["valid"]["binary_logloss"][-1],
        }

    def _train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model."""
        dtrain = self.xgb.DMatrix(
            X_train, label=y_train, feature_names=self.feature_names
        )
        dval = self.xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)

        params = {
            "objective": "binary:logistic",
            "eval_metric": ["logloss", "auc"],
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "gamma": 0.01,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
        }

        evals_result = {}
        self.model = self.xgb.train(
            params,
            dtrain,
            num_boost_round=200,
            evals=[(dtrain, "train"), (dval, "valid")],
            early_stopping_rounds=20,
            verbose_eval=50,
            evals_result=evals_result,
        )

        y_pred_train = (self.model.predict(dtrain) > 0.5).astype(int)
        y_pred_val = (self.model.predict(dval) > 0.5).astype(int)

        return {
            "train_accuracy": np.mean(y_pred_train == y_train),
            "val_accuracy": np.mean(y_pred_val == y_val),
            "best_iteration": self.model.best_iteration,
            "final_train_loss": evals_result["train"]["logloss"][-1],
            "final_val_loss": evals_result["valid"]["logloss"][-1],
        }

    def _train_catboost(self, X_train, y_train, X_val, y_val):
        """Train CatBoost model."""
        from catboost import Pool

        train_pool = Pool(X_train, y_train, feature_names=self.feature_names)
        val_pool = Pool(X_val, y_val, feature_names=self.feature_names)

        self.model = self.CatBoostClassifier(
            iterations=200,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            loss_function="Logloss",
            eval_metric="Accuracy",
            early_stopping_rounds=20,
            verbose=50,
        )

        self.model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        y_pred_train = self.model.predict(X_train)
        y_pred_val = self.model.predict(X_val)

        return {
            "train_accuracy": np.mean(y_pred_train == y_train),
            "val_accuracy": np.mean(y_pred_val == y_val),
            "best_iteration": self.model.get_best_iteration(),
        }

    def predict(self, X):
        """
        Make predictions on new data.

        Args:
            X: Feature data (dict or list of dicts)

        Returns:
            dict: Prediction with direction and confidence
        """
        if self.model is None:
            logger.error("Model not trained yet")
            return None

        if isinstance(X, dict):
            X = [X]

        X_array, _ = self._prepare_features(X)

        if self.model_type == "lightgbm":
            proba = self.model.predict(X_array)[0]
        elif self.model_type == "xgboost":
            dtest = self.xgb.DMatrix(X_array, feature_names=self.feature_names)
            proba = self.model.predict(dtest)[0]
        elif self.model_type == "catboost":
            proba = self.model.predict_proba(X_array)[0][1]

        direction = "up" if proba > 0.5 else "down"
        confidence = proba if proba > 0.5 else (1 - proba)

        return {
            "direction": direction,
            "confidence": confidence,
            "probability_up": proba,
        }

    def _prepare_features(self, X):
        """Convert feature dicts to numpy array."""
        if isinstance(X[0], dict):
            if self.feature_names is None:
                feature_names = sorted(set().union(*[set(x.keys()) for x in X]))
            else:
                feature_names = self.feature_names

            X_array = np.zeros((len(X), len(feature_names)))
            for i, features in enumerate(X):
                for j, name in enumerate(feature_names):
                    X_array[i, j] = features.get(name, 0)

            return X_array, feature_names
        else:
            return np.array(X), self.feature_names

    def save_model(self, filename=None):
        """Save trained model to disk."""
        if self.model is None:
            logger.warning("No model to save")
            return None

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.model_type}_{timestamp}.pkl"

        filepath = self.model_dir / filename

        model_data = {
            "model": self.model,
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "training_history": self.training_history,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {filepath}")
        return filepath

    def load_model(self, filepath):
        """Load a trained model from disk."""
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.model_type = model_data["model_type"]
        self.feature_names = model_data["feature_names"]
        self.training_history = model_data.get("training_history", [])

        self._import_model_library()

        logger.info(f"Model loaded from {filepath}")

    def get_feature_importance(self, top_n=20):
        """Get feature importance scores."""
        if self.model is None:
            return None

        if self.model_type == "lightgbm":
            importance = self.model.feature_importance(importance_type="gain")
        elif self.model_type == "xgboost":
            importance = self.model.get_score(importance_type="gain")
            importance = np.array([importance.get(f, 0) for f in self.feature_names])
        elif self.model_type == "catboost":
            importance = self.model.get_feature_importance()

        feature_importance = sorted(
            zip(self.feature_names, importance), key=lambda x: x[1], reverse=True
        )

        return feature_importance[:top_n]
