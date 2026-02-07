import logging
import numpy as np
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Generates technical indicators and features from price data."""

    def __init__(self, lookback_windows=20, indicators=None):
        """
        Initialize feature engineer.

        Args:
            lookback_windows: Number of previous windows to use for features
            indicators: List of indicator names to calculate
        """
        self.lookback_windows = lookback_windows
        self.indicators = indicators or [
            "rsi",
            "macd",
            "bbands",
            "volume_ratio",
            "price_momentum",
        ]

        logger.info(
            f"Initialized feature engineer with {len(self.indicators)} indicators"
        )

    def create_features(self, window_data):
        """
        Create features from historical window data.

        Args:
            window_data: List of dicts with OHLCV data for each window

        Returns:
            dict: Feature dictionary
        """
        if len(window_data) < 2:
            logger.warning("Insufficient data for feature creation")
            return None

        df = pd.DataFrame(window_data)
        features = {}

        features.update(self._price_features(df))

        if "rsi" in self.indicators:
            features.update(self._calculate_rsi(df))

        if "macd" in self.indicators:
            features.update(self._calculate_macd(df))

        if "bbands" in self.indicators:
            features.update(self._calculate_bollinger_bands(df))

        if "volume_ratio" in self.indicators:
            features.update(self._calculate_volume_features(df))

        if "price_momentum" in self.indicators:
            features.update(self._calculate_momentum(df))

        features.update(self._time_features(df))

        features.update(self._volatility_features(df))

        return features

    def _price_features(self, df):
        """Calculate basic price-based features."""
        features = {}

        if len(df) < 2:
            return features

        close_prices = df["close"].values

        features["price_change_1"] = (
            close_prices[-1] - close_prices[-2]
        ) / close_prices[-2]

        if len(df) >= 5:
            features["price_change_5"] = (
                close_prices[-1] - close_prices[-5]
            ) / close_prices[-5]

        if len(df) >= 10:
            features["price_change_10"] = (
                close_prices[-1] - close_prices[-10]
            ) / close_prices[-10]

        if len(df) >= 20:
            recent_high = df["high"].tail(20).max()
            recent_low = df["low"].tail(20).min()
            if recent_high > recent_low:
                features["price_position"] = (close_prices[-1] - recent_low) / (
                    recent_high - recent_low
                )

        current = df.iloc[-1]
        if current["high"] != current["low"]:
            features["candle_body_ratio"] = abs(current["close"] - current["open"]) / (
                current["high"] - current["low"]
            )
            features["upper_shadow_ratio"] = (
                current["high"] - max(current["open"], current["close"])
            ) / (current["high"] - current["low"])
            features["lower_shadow_ratio"] = (
                min(current["open"], current["close"]) - current["low"]
            ) / (current["high"] - current["low"])

        features["is_green_candle"] = 1.0 if current["close"] > current["open"] else 0.0

        return features

    def _calculate_rsi(self, df, period=14):
        """Calculate Relative Strength Index."""
        features = {}

        if len(df) < period + 1:
            return features

        close = df["close"].values
        deltas = np.diff(close)

        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            features["rsi"] = 100.0
        else:
            rs = avg_gain / avg_loss
            features["rsi"] = 100 - (100 / (1 + rs))

        features["rsi_overbought"] = 1.0 if features["rsi"] > 70 else 0.0
        features["rsi_oversold"] = 1.0 if features["rsi"] < 30 else 0.0

        return features

    def _calculate_macd(self, df, fast=12, slow=26, signal=9):
        """Calculate MACD indicator."""
        features = {}

        if len(df) < slow + signal:
            return features

        close = df["close"].values

        ema_fast = self._ema(close, fast)
        ema_slow = self._ema(close, slow)

        macd_line = ema_fast[-1] - ema_slow[-1]

        macd_history = ema_fast[-signal:] - ema_slow[-signal:]
        signal_line = self._ema(macd_history, signal)[-1]

        features["macd"] = macd_line
        features["macd_signal"] = signal_line
        features["macd_histogram"] = macd_line - signal_line
        features["macd_bullish"] = 1.0 if macd_line > signal_line else 0.0

        return features

    def _calculate_bollinger_bands(self, df, period=20, num_std=2):
        """Calculate Bollinger Bands."""
        features = {}

        if len(df) < period:
            return features

        close = df["close"].values[-period:]

        sma = np.mean(close)
        std = np.std(close)

        upper_band = sma + (num_std * std)
        lower_band = sma - (num_std * std)

        current_price = close[-1]

        features["bb_upper"] = upper_band
        features["bb_lower"] = lower_band
        features["bb_middle"] = sma
        features["bb_width"] = (upper_band - lower_band) / sma if sma > 0 else 0

        if upper_band > lower_band:
            features["bb_position"] = (current_price - lower_band) / (
                upper_band - lower_band
            )

        features["price_above_bb_upper"] = 1.0 if current_price > upper_band else 0.0
        features["price_below_bb_lower"] = 1.0 if current_price < lower_band else 0.0

        return features

    def _calculate_volume_features(self, df):
        """Calculate volume-based features."""
        features = {}

        if len(df) < 10:
            return features

        volumes = df["volume"].values

        current_volume = volumes[-1]
        avg_volume_10 = np.mean(volumes[-10:])
        avg_volume_20 = np.mean(volumes[-20:]) if len(df) >= 20 else avg_volume_10

        features["volume_ratio_10"] = (
            current_volume / avg_volume_10 if avg_volume_10 > 0 else 1.0
        )
        features["volume_ratio_20"] = (
            current_volume / avg_volume_20 if avg_volume_20 > 0 else 1.0
        )

        if len(df) >= 5:
            volume_slope = np.polyfit(range(5), volumes[-5:], 1)[0]
            features["volume_trend"] = (
                volume_slope / avg_volume_10 if avg_volume_10 > 0 else 0
            )

        if "vwap" in df.columns:
            vwap = df["vwap"].values[-1]
            close = df["close"].values[-1]
            features["price_vwap_diff"] = (close - vwap) / vwap if vwap > 0 else 0

        return features

    def _calculate_momentum(self, df):
        """Calculate momentum indicators."""
        features = {}

        close = df["close"].values

        if len(df) >= 5:
            features["roc_5"] = (close[-1] - close[-5]) / close[-5]

        if len(df) >= 10:
            features["roc_10"] = (close[-1] - close[-10]) / close[-10]

        if len(df) >= 10:
            features["momentum_10"] = close[-1] - close[-10]

        if len(df) >= 3:
            first_diff = np.diff(close[-3:])
            if len(first_diff) >= 2:
                features["price_acceleration"] = first_diff[-1] - first_diff[-2]

        return features

    def _volatility_features(self, df):
        """Calculate volatility-based features."""
        features = {}

        if len(df) < 10:
            return features

        close = df["close"].values
        returns = np.diff(close) / close[:-1]

        features["volatility_10"] = np.std(returns[-10:])

        if len(df) >= 20:
            features["volatility_20"] = np.std(returns[-20:])
            features["volatility_ratio"] = (
                features["volatility_10"] / features["volatility_20"]
                if features["volatility_20"] > 0
                else 1.0
            )

        if "high" in df.columns and "low" in df.columns:
            highs = df["high"].values[-10:]
            lows = df["low"].values[-10:]
            closes = df["close"].values[-11:-1]

            tr = np.maximum(
                highs - lows, np.maximum(np.abs(highs - closes), np.abs(lows - closes))
            )
            features["atr"] = np.mean(tr)
            features["atr_ratio"] = features["atr"] / close[-1] if close[-1] > 0 else 0

        return features

    def _time_features(self, df):
        """Create time-based features."""
        features = {}

        if "timestamp" not in df.columns or df["timestamp"].iloc[-1] is None:
            return features

        timestamp = df["timestamp"].iloc[-1]

        features["hour"] = timestamp.hour
        features["hour_sin"] = np.sin(2 * np.pi * timestamp.hour / 24)
        features["hour_cos"] = np.cos(2 * np.pi * timestamp.hour / 24)

        features["day_of_week"] = timestamp.weekday()
        features["is_weekend"] = 1.0 if timestamp.weekday() >= 5 else 0.0

        return features

    def _ema(self, data, period):
        """Calculate Exponential Moving Average."""
        ema = np.zeros_like(data, dtype=float)
        ema[0] = data[0]
        multiplier = 2.0 / (period + 1)

        for i in range(1, len(data)):
            ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1]

        return ema

    def get_feature_names(self):
        """Return list of all possible feature names."""
        base_features = [
            "price_change_1",
            "price_change_5",
            "price_change_10",
            "price_position",
            "candle_body_ratio",
            "upper_shadow_ratio",
            "lower_shadow_ratio",
            "is_green_candle",
            "rsi",
            "rsi_overbought",
            "rsi_oversold",
            "macd",
            "macd_signal",
            "macd_histogram",
            "macd_bullish",
            "bb_upper",
            "bb_lower",
            "bb_middle",
            "bb_width",
            "bb_position",
            "price_above_bb_upper",
            "price_below_bb_lower",
            "volume_ratio_10",
            "volume_ratio_20",
            "volume_trend",
            "price_vwap_diff",
            "roc_5",
            "roc_10",
            "momentum_10",
            "price_acceleration",
            "volatility_10",
            "volatility_20",
            "volatility_ratio",
            "atr",
            "atr_ratio",
            "hour",
            "hour_sin",
            "hour_cos",
            "day_of_week",
            "is_weekend",
        ]

        sentiment_features = [
            "news_sentiment_24h",
            "news_confidence_24h",
            "news_positive_24h",
            "news_negative_24h",
            "news_sentiment_6h",
            "news_confidence_6h",
            "has_breaking_news",
            "breaking_news_sentiment",
            "news_volume_24h",
        ]

        return base_features + sentiment_features
