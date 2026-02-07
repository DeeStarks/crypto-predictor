#!/usr/bin/env python3

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime

from utils.window_manager import WindowManager
from utils.logger_config import setup_logging
from data.collector import DataCollector
from data.preprocessor import FeatureEngineer
from data.storage import DataStorage
from models.trainer import ModelTrainer
from models.predictor import Predictor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time cryptocurrency price prediction using gradient boosting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Trading pair symbol (e.g., BTCUSDT, ETHUSDT)",
    )

    parser.add_argument(
        "--exchange",
        type=str,
        default="binance",
        choices=["binance", "coinbase", "kraken"],
        help="Cryptocurrency exchange to use",
    )

    parser.add_argument(
        "--window-minutes",
        type=int,
        default=15,
        help="Window size in minutes for predictions",
    )

    parser.add_argument(
        "--poll-interval", type=int, default=5, help="Data polling interval in seconds"
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default="lightgbm",
        choices=["lightgbm", "xgboost", "catboost"],
        help="Gradient boosting model type",
    )

    parser.add_argument(
        "--training-windows",
        type=int,
        default=1000,
        help="Number of historical windows to use for training",
    )

    parser.add_argument(
        "--retrain-interval", type=int, default=10, help="Retrain model every N windows"
    )

    parser.add_argument(
        "--min-samples",
        type=int,
        default=100,
        help="Minimum samples required before making predictions",
    )

    parser.add_argument(
        "--lookback-windows",
        type=int,
        default=1000,
        help="Number of previous windows to use for features",
    )

    parser.add_argument(
        "--technical-indicators",
        nargs="+",
        default=["rsi", "macd", "bbands", "volume_ratio", "price_momentum"],
        help="Technical indicators to calculate",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.6,
        help="Minimum confidence for predictions (0.5-1.0)",
    )

    parser.add_argument(
        "--predict-timing",
        type=str,
        default="end",
        choices=["start", "end"],
        help="When to make prediction: at window start or end",
    )

    parser.add_argument(
        "--prediction-offset",
        type=int,
        default=30,
        help="Seconds before window end to make prediction (if predict-timing=end)",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/storage",
        help="Directory to store historical data",
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default="./models/saved",
        help="Directory to save trained models",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="live",
        choices=["live", "backtest", "train"],
        help="Operating mode: live prediction, backtesting, or initial training",
    )

    parser.add_argument(
        "--backtest-start", type=str, help="Backtest start date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--backtest-end", type=str, help="Backtest end date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    parser.add_argument(
        "--log-file", type=str, default="./logs/predictor.log", help="Log file path"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output (log to file only)",
    )

    parser.add_argument(
        "--output-format",
        type=str,
        default="text",
        choices=["text", "json", "csv"],
        help="Output format for predictions",
    )

    parser.add_argument(
        "--max-position-size",
        type=float,
        default=1.0,
        help="Maximum position size as fraction of capital",
    )

    parser.add_argument(
        "--fee-rate",
        type=float,
        default=0.001,
        help="Trading fee rate (e.g., 0.001 for 0.1%%)",
    )

    parser.add_argument(
        "--enable-ensemble", action="store_true", help="Use ensemble of multiple models"
    )

    parser.add_argument(
        "--regime-detection", action="store_true", help="Enable market regime detection"
    )

    parser.add_argument(
        "--feature-selection",
        action="store_true",
        help="Enable automatic feature selection",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without making actual predictions (testing mode)",
    )

    parser.add_argument(
        "--simple-display",
        action="store_true",
        default=False,
        help="Use simple display (default: False)",
    )

    parser.add_argument(
        "--enable-sentiment",
        action="store_true",
        default=True,
        help="Enable sentiment analysis from news (default: True)",
    )

    parser.add_argument(
        "--enable-caching",
        action="store_true",
        default=True,
        help="Enable API call caching (default: True)",
    )

    return parser.parse_args()


def validate_args(args):
    """Validate command-line arguments."""
    if args.confidence_threshold < 0.5 or args.confidence_threshold > 1.0:
        raise ValueError("Confidence threshold must be between 0.5 and 1.0")

    if args.window_minutes < 1:
        raise ValueError("Window size must be at least 1 minute")

    if args.poll_interval < 1:
        raise ValueError("Poll interval must be at least 1 second")

    if args.poll_interval >= args.window_minutes * 60:
        raise ValueError("Poll interval must be less than window size")

    if args.training_windows < args.min_samples:
        raise ValueError("Training windows must be >= min_samples")

    if args.mode == "backtest":
        if not args.backtest_start or not args.backtest_end:
            raise ValueError(
                "Backtest mode requires --backtest-start and --backtest-end"
            )

    return True


def main():
    """Main entry point for the CLI tool."""
    args = parse_args()

    try:
        validate_args(args)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    setup_logging(args.log_level, args.log_file, args.quiet)
    logger = logging.getLogger(__name__)

    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Crypto Price Predictor - Starting")
    logger.info("=" * 60)
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Exchange: {args.exchange}")
    logger.info(f"Window: {args.window_minutes} minutes")
    logger.info(f"Model: {args.model_type}")
    logger.info(f"Mode: {args.mode}")
    logger.info("=" * 60)

    try:
        storage = DataStorage(args.data_dir)
        collector = DataCollector(args.exchange, args.symbol, args.poll_interval)
        feature_engineer = FeatureEngineer(
            lookback_windows=args.lookback_windows, indicators=args.technical_indicators
        )
        model_trainer = ModelTrainer(
            model_type=args.model_type,
            model_dir=args.model_dir,
            training_windows=args.training_windows,
        )
        predictor = Predictor(
            model_trainer=model_trainer,
            confidence_threshold=args.confidence_threshold,
            output_format=args.output_format,
        )
        window_manager = WindowManager(
            window_minutes=args.window_minutes,
            predict_timing=args.predict_timing,
            prediction_offset=args.prediction_offset,
        )

    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        sys.exit(1)

    try:
        if args.mode == "train":
            from modes.train_mode import run_training

            run_training(args, storage, collector, feature_engineer, model_trainer)

        elif args.mode == "backtest":
            from modes.backtest_mode import run_backtest

            run_backtest(args, storage, feature_engineer, model_trainer, predictor)

        elif args.mode == "live":
            if args.simple_display:
                from modes.live_mode import run_live_prediction

                run_live_prediction(
                    args,
                    storage,
                    collector,
                    feature_engineer,
                    model_trainer,
                    predictor,
                    window_manager,
                )
            else:
                from modes.live_mode_enhanced import run_live_prediction_enhanced

                run_live_prediction_enhanced(
                    args,
                    storage,
                    collector,
                    feature_engineer,
                    model_trainer,
                    predictor,
                    window_manager,
                )

    except KeyboardInterrupt:
        logger.info("\nShutdown requested by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
