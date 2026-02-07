import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def run_training(args, storage, collector, feature_engineer, model_trainer):
    """
    Run initial training mode using historical data.

    Args:
        args: Command-line arguments
        storage: Data storage instance
        collector: Data collector instance
        feature_engineer: Feature engineer instance
        model_trainer: Model trainer instance
    """
    logger.info("Starting TRAINING mode")
    logger.info(f"Symbol: {args.symbol}, Exchange: {args.exchange}")

    logger.info(f"Fetching historical data for training...")

    if args.window_minutes <= 5:
        interval = "1m"
    elif args.window_minutes <= 15:
        interval = "5m"
    elif args.window_minutes <= 60:
        interval = "15m"
    else:
        interval = "1h"

    limit = min(args.training_windows + args.lookback_windows + 10, 1000)
    logger.info(f"Fetching {limit} historical candles at {interval} interval...")

    klines = collector.fetch_historical_klines(interval=interval, limit=limit)

    if not klines or len(klines) < args.min_samples:
        logger.error(
            f"Insufficient historical data: {len(klines) if klines else 0} candles"
        )
        logger.info("Try reducing --training-windows or --min-samples")
        return

    logger.info(f"Fetched {len(klines)} historical candles")
    logger.info(f"Date range: {klines[0]['timestamp']} to {klines[-1]['timestamp']}")

    logger.info("Processing historical data into windows...")

    windows = []
    if interval != f"{args.window_minutes}m":
        candles_per_window = args.window_minutes // int(
            interval.replace("m", "").replace("h", "")
        )

        for i in range(0, len(klines), candles_per_window):
            window_candles = klines[i : i + candles_per_window]
            if len(window_candles) == candles_per_window:
                window_data = {
                    "timestamp": window_candles[-1]["timestamp"],
                    "open": window_candles[0]["open"],
                    "high": max(c["high"] for c in window_candles),
                    "low": min(c["low"] for c in window_candles),
                    "close": window_candles[-1]["close"],
                    "volume": sum(c["volume"] for c in window_candles),
                }
                windows.append(window_data)
    else:
        windows = [
            {
                "timestamp": k["timestamp"],
                "open": k["open"],
                "high": k["high"],
                "low": k["low"],
                "close": k["close"],
                "volume": k["volume"],
            }
            for k in klines
        ]

    logger.info(f"Processed {len(windows)} windows")

    logger.info("Creating features and labels...")

    X = []
    y = []

    for i in range(args.lookback_windows, len(windows) - 1):
        historical = windows[i - args.lookback_windows : i + 1]

        features = feature_engineer.create_features(historical)
        if features is None:
            continue

        current_close = windows[i]["close"]
        next_close = windows[i + 1]["close"]
        label = 1 if next_close > current_close else 0

        X.append(features)
        y.append(label)

        window_id = windows[i]["timestamp"].strftime("%Y%m%d_%H%M")
        storage.save_window(window_id, windows[i], features)

    logger.info(f"Created {len(X)} training samples")

    up_count = sum(y)
    down_count = len(y) - up_count
    logger.info(
        f"Label distribution: {up_count} up ({up_count/len(y)*100:.1f}%), "
        f"{down_count} down ({down_count/len(y)*100:.1f}%)"
    )

    if len(X) < args.min_samples:
        logger.error(f"Insufficient samples: {len(X)} < {args.min_samples}")
        return

    logger.info("Training model...")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Training samples: {len(X)}")

    metrics = model_trainer.train(X, y, validation_split=0.2)

    if metrics:
        logger.info("")
        logger.info("=" * 60)
        logger.info("TRAINING RESULTS")
        logger.info("=" * 60)
        logger.info(f"Training accuracy: {metrics['train_accuracy']:.4f}")
        logger.info(f"Validation accuracy: {metrics['val_accuracy']:.4f}")
        if "best_iteration" in metrics:
            logger.info(f"Best iteration: {metrics['best_iteration']}")
        logger.info("=" * 60)

        model_path = model_trainer.save_model()
        logger.info(f"Model saved to: {model_path}")

        importance = model_trainer.get_feature_importance(top_n=10)
        if importance:
            logger.info("")
            logger.info("Top 10 Feature Importance:")
            for i, (feature, score) in enumerate(importance, 1):
                logger.info(f"  {i}. {feature}: {score:.2f}")

        metadata = {
            "trained_at": datetime.now().isoformat(),
            "symbol": args.symbol,
            "exchange": args.exchange,
            "window_minutes": args.window_minutes,
            "model_type": args.model_type,
            "training_samples": len(X),
            "metrics": metrics,
            "feature_importance": importance[:10] if importance else None,
        }
        storage.save_metadata(metadata)

        logger.info("")
        logger.info("Training complete! Model ready for live prediction.")
        logger.info(f"Run with: python main.py --mode live --symbol {args.symbol}")
    else:
        logger.error("Training failed")
