import logging
import json
import pickle
from pathlib import Path
from datetime import datetime
import pandas as pd
from utils.type_conversion import convert_to_native_types

logger = logging.getLogger(__name__)


class DataStorage:
    """Manages persistent storage of cryptocurrency data."""

    def __init__(self, data_dir="./data/storage"):
        """
        Initialize data storage.

        Args:
            data_dir: Directory to store data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.windows_file = self.data_dir / "window_data.json"
        self.features_file = self.data_dir / "features.pkl"
        self.metadata_file = self.data_dir / "metadata.json"

        logger.info(f"Initialized data storage in {data_dir}")

    def save_window(self, window_id, window_data, features=None, prediction=None):
        """
        Save a completed window's data.

        Args:
            window_id: Unique window identifier
            window_data: OHLCV data for the window
            features: Calculated features (optional)
            prediction: Prediction made for this window (optional)
        """
        windows = self.load_windows()

        entry = {
            "window_id": window_id,
            "timestamp": (
                window_data["timestamp"].isoformat()
                if isinstance(window_data["timestamp"], datetime)
                else window_data["timestamp"]
            ),
            "data": {k: v for k, v in window_data.items() if k != "timestamp"},
            "prediction": prediction,
        }

        windows[window_id] = entry

        with open(self.windows_file, "w") as f:
            json.dump(convert_to_native_types(windows), f, indent=2)

        if features:
            self.save_features(window_id, features)

        logger.debug(f"Saved window {window_id}")

    def load_windows(self, limit=None):
        """
        Load historical windows.

        Args:
            limit: Maximum number of windows to load (most recent)

        Returns:
            dict: Window data indexed by window_id
        """
        if not self.windows_file.exists():
            return {}

        try:
            with open(self.windows_file, "r") as f:
                windows = json.load(f)

            if limit:
                sorted_windows = sorted(
                    windows.items(), key=lambda x: x[1]["timestamp"], reverse=True
                )[:limit]
                windows = dict(sorted_windows)

            return windows
        except Exception as e:
            logger.error(f"Error loading windows: {e}")
            return {}

    def save_features(self, window_id, features):
        """Save features for a window."""
        all_features = self.load_all_features()
        all_features[window_id] = features

        with open(self.features_file, "wb") as f:
            pickle.dump(all_features, f)

    def load_all_features(self):
        """Load all stored features."""
        if not self.features_file.exists():
            return {}

        try:
            with open(self.features_file, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading features: {e}")
            return {}

    def get_recent_windows(self, n=100):
        """
        Get N most recent windows as a list.

        Args:
            n: Number of windows to retrieve

        Returns:
            list: Window data sorted by timestamp
        """
        windows = self.load_windows()

        window_list = list(windows.values())
        window_list.sort(key=lambda x: x["timestamp"], reverse=True)

        return window_list[:n]

    def get_training_data(self, n_windows=1000):
        """
        Prepare training data from stored windows.

        Args:
            n_windows: Number of windows to use

        Returns:
            tuple: (X, y) features and labels
        """
        windows = self.get_recent_windows(n_windows)
        features_dict = self.load_all_features()

        X = []
        y = []

        for i in range(len(windows) - 1):
            window = windows[i]
            next_window = windows[i + 1]

            window_id = window["window_id"]

            if window_id in features_dict:
                features = features_dict[window_id]

                current_close = window["data"]["close"]
                next_close = next_window["data"]["close"]
                label = 1 if next_close > current_close else 0

                X.append(features)
                y.append(label)

        return X, y

    def save_metadata(self, metadata):
        """Save metadata (model info, performance stats, etc.)."""
        with open(self.metadata_file, "w") as f:
            json.dump(convert_to_native_types(metadata), f, indent=2)

    def load_metadata(self):
        """Load metadata."""
        if not self.metadata_file.exists():
            return {}

        try:
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return {}

    def export_to_csv(self, output_file=None):
        """
        Export window data to CSV.

        Args:
            output_file: Path to output CSV file
        """
        if output_file is None:
            output_file = self.data_dir / "windows_export.csv"

        windows = self.load_windows()

        records = []
        for window_id, window in windows.items():
            record = {
                "window_id": window_id,
                "timestamp": window["timestamp"],
                **window["data"],
            }
            if window.get("prediction"):
                record.update(
                    {
                        "predicted_direction": window["prediction"].get("direction"),
                        "confidence": window["prediction"].get("confidence"),
                    }
                )
            records.append(record)

        df = pd.DataFrame(records)
        df.to_csv(output_file, index=False)

        logger.info(f"Exported {len(records)} windows to {output_file}")
        return output_file

    def get_statistics(self):
        """Get storage statistics."""
        windows = self.load_windows()
        features = self.load_all_features()

        stats = {
            "total_windows": len(windows),
            "total_features": len(features),
            "storage_size_mb": sum(f.stat().st_size for f in self.data_dir.glob("*"))
            / 1024
            / 1024,
        }

        if windows:
            timestamps = [w["timestamp"] for w in windows.values()]
            stats["earliest_window"] = min(timestamps)
            stats["latest_window"] = max(timestamps)

        return stats

    def cleanup_old_data(self, keep_n_windows=5000):
        """
        Remove old data to prevent unbounded growth.

        Args:
            keep_n_windows: Number of most recent windows to keep
        """
        windows = self.load_windows()

        if len(windows) <= keep_n_windows:
            logger.info(f"No cleanup needed ({len(windows)} windows)")
            return

        sorted_windows = sorted(
            windows.items(), key=lambda x: x[1]["timestamp"], reverse=True
        )

        windows_to_keep = dict(sorted_windows[:keep_n_windows])
        removed_count = len(windows) - len(windows_to_keep)

        with open(self.windows_file, "w") as f:
            json.dump(convert_to_native_types(windows_to_keep), f, indent=2)

        features = self.load_all_features()
        kept_window_ids = set(windows_to_keep.keys())
        features_to_keep = {k: v for k, v in features.items() if k in kept_window_ids}

        with open(self.features_file, "wb") as f:
            pickle.dump(features_to_keep, f)

        logger.info(f"Cleaned up {removed_count} old windows")
