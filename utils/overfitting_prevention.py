import logging
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from collections import deque

logger = logging.getLogger(__name__)


class OverfittingPreventor:
    """Implements multiple techniques to prevent overfitting."""

    def __init__(self, patience=5, min_delta=0.001):
        """
        Initialize overfitting prevention.

        Args:
            patience: Number of iterations to wait for improvement
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.validation_history = deque(maxlen=100)
        self.best_score = None
        self.wait = 0

        logger.info(f"Initialized overfitting prevention (patience={patience})")

    def check_early_stopping(self, validation_score):
        """
        Check if training should stop early.

        Args:
            validation_score: Current validation score (higher is better)

        Returns:
            bool: True if training should stop
        """
        self.validation_history.append(validation_score)

        if self.best_score is None:
            self.best_score = validation_score
            return False

        if validation_score > self.best_score + self.min_delta:
            self.best_score = validation_score
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            logger.info(
                f"Early stopping triggered (no improvement for {self.patience} iterations)"
            )
            return True

        return False

    def detect_overfitting(self, train_score, val_score, threshold=0.1):
        """
        Detect if model is overfitting.

        Args:
            train_score: Training accuracy
            val_score: Validation accuracy
            threshold: Maximum acceptable gap

        Returns:
            dict: Overfitting status and metrics
        """
        gap = train_score - val_score
        is_overfitting = gap > threshold

        if is_overfitting:
            logger.warning(
                f"Overfitting detected! Train: {train_score:.3f}, Val: {val_score:.3f}, Gap: {gap:.3f}"
            )

        return {
            "is_overfitting": is_overfitting,
            "train_score": train_score,
            "val_score": val_score,
            "gap": gap,
            "severity": "high" if gap > 0.2 else "medium" if gap > 0.1 else "low",
        }

    def time_series_cv_score(self, X, y, model, n_splits=5):
        """
        Perform time-series cross-validation.

        Args:
            X: Feature data
            y: Labels
            model: Model instance with fit/predict methods
            n_splits: Number of CV splits

        Returns:
            dict: CV scores and statistics
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model.fit(X_train, y_train)

            predictions = model.predict(X_val)
            if hasattr(predictions, "shape") and len(predictions.shape) > 1:
                predictions = predictions[:, 1]

            accuracy = np.mean((predictions > 0.5) == y_val)
            scores.append(accuracy)

            logger.debug(f"Fold {fold + 1}/{n_splits}: {accuracy:.3f}")

        return {
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "scores": scores,
            "stability": (
                1 - (np.std(scores) / np.mean(scores)) if np.mean(scores) > 0 else 0
            ),
        }

    def get_validation_trends(self):
        """Analyze validation score trends."""
        if len(self.validation_history) < 3:
            return None

        recent_scores = list(self.validation_history)

        x = np.arange(len(recent_scores))
        coeffs = np.polyfit(x, recent_scores, 1)
        slope = coeffs[0]

        recent_avg = (
            np.mean(recent_scores[-10:])
            if len(recent_scores) >= 10
            else np.mean(recent_scores)
        )
        historical_avg = np.mean(recent_scores)

        return {
            "trend": (
                "improving"
                if slope > 0.001
                else "declining" if slope < -0.001 else "stable"
            ),
            "slope": slope,
            "recent_avg": recent_avg,
            "historical_avg": historical_avg,
            "volatility": np.std(recent_scores),
            "degradation": recent_avg < historical_avg - 0.05,
        }

    def suggest_regularization(self, overfitting_status):
        """
        Suggest regularization parameters based on overfitting status.

        Args:
            overfitting_status: Output from detect_overfitting

        Returns:
            dict: Suggested parameter adjustments
        """
        suggestions = {}

        if not overfitting_status["is_overfitting"]:
            return {"message": "No overfitting detected, current parameters are good"}

        severity = overfitting_status["severity"]

        if severity == "high":
            suggestions = {
                "learning_rate": "decrease to 0.03",
                "max_depth": "reduce by 1-2",
                "min_child_weight": "increase to 10",
                "subsample": "decrease to 0.7",
                "lambda_l1": "increase to 0.5",
                "lambda_l2": "increase to 0.5",
                "num_leaves": "reduce by 30%",
            }
        elif severity == "medium":
            suggestions = {
                "learning_rate": "decrease to 0.04",
                "max_depth": "reduce by 1",
                "subsample": "decrease to 0.8",
                "lambda_l1": "increase to 0.3",
                "lambda_l2": "increase to 0.3",
            }

        return suggestions

    def monitor_feature_importance_stability(self, importance_history):
        """
        Check if feature importance is stable across retrainings.

        Args:
            importance_history: List of feature importance dicts

        Returns:
            dict: Stability metrics
        """
        if len(importance_history) < 2:
            return {"stable": True, "reason": "insufficient history"}

        top_features_list = []
        for imp in importance_history[-5:]:
            top_features = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:10]
            top_features_list.append(set(f[0] for f in top_features))

        if len(top_features_list) >= 2:
            overlaps = []
            for i in range(len(top_features_list) - 1):
                overlap = len(top_features_list[i] & top_features_list[i + 1]) / 10
                overlaps.append(overlap)

            avg_overlap = np.mean(overlaps)

            return {
                "stable": avg_overlap >= 0.7,
                "overlap_score": avg_overlap,
                "message": f"Top features overlap: {avg_overlap:.1%}",
            }

        return {"stable": True, "reason": "insufficient data"}


class DataLeakageDetector:
    """Detect potential data leakage in features."""

    def __init__(self):
        self.suspicious_correlations = []

    def check_target_correlation(self, X, y, feature_names, threshold=0.9):
        """
        Check for suspiciously high correlations with target.

        Args:
            X: Feature array
            y: Target array
            feature_names: List of feature names
            threshold: Correlation threshold for concern

        Returns:
            list: Suspicious features
        """
        suspicious = []

        for i, name in enumerate(feature_names):
            correlation = np.corrcoef(X[:, i], y)[0, 1]

            if abs(correlation) > threshold:
                suspicious.append(
                    {
                        "feature": name,
                        "correlation": correlation,
                        "warning": "Extremely high correlation - possible data leakage",
                    }
                )

        if suspicious:
            logger.warning(
                f"Found {len(suspicious)} features with suspicious correlations"
            )
            for item in suspicious:
                logger.warning(f"  {item['feature']}: {item['correlation']:.3f}")

        self.suspicious_correlations = suspicious
        return suspicious

    def check_future_information(self, feature_dict, current_timestamp):
        """
        Check if features contain future information.

        Args:
            feature_dict: Dictionary of features
            current_timestamp: Current window timestamp

        Returns:
            list: Features that may contain future info
        """
        future_info_features = []

        suspicious_patterns = ["next_", "future_", "forward_"]

        for feature_name in feature_dict.keys():
            for pattern in suspicious_patterns:
                if pattern in feature_name.lower():
                    future_info_features.append(
                        {
                            "feature": feature_name,
                            "reason": f'Name contains "{pattern}"',
                        }
                    )

        if future_info_features:
            logger.warning(
                f"Found {len(future_info_features)} features that may contain future information"
            )

        return future_info_features
