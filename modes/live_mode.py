import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)


def run_live_prediction(
    args, storage, collector, feature_engineer, model_trainer, predictor, window_manager
):
    """
    Run live prediction mode with real-time data collection.

    Args:
        args: Command-line arguments
        storage: Data storage instance
        collector: Data collector instance
        feature_engineer: Feature engineer instance
        model_trainer: Model trainer instance
        predictor: Predictor instance
        window_manager: Window manager instance
    """
    logger.info("Starting LIVE PREDICTION mode")
    logger.info(f"Symbol: {args.symbol}, Exchange: {args.exchange}")
    logger.info(f"Window: {args.window_minutes} minutes")
    logger.info(f"Retrain interval: {args.retrain_interval} windows")

    model_files = list(model_trainer.model_dir.glob(f"{args.model_type}_*.pkl"))
    if model_files:
        latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Loading existing model: {latest_model}")
        model_trainer.load_model(latest_model)
    else:
        logger.info("No existing model found - will train on first window completion")

    windows_processed = 0
    predictions_made = 0
    last_retrain = 0

    try:
        while True:
            window_start, window_end = window_manager.get_current_window()
            window_id = window_manager.get_window_id(window_start)

            logger.info("")
            logger.info(
                f"Current window: {window_manager.format_window_display(window_start, window_end)}"
            )
            logger.info(f"Window ID: {window_id}")

            window_duration = args.window_minutes * 60
            ticks = collector.collect_window_data(window_duration)

            if not ticks:
                logger.error("No data collected - skipping window")
                window_manager.wait_for_next_window()
                continue

            window_data = collector.aggregate_window_ticks(ticks)
            logger.info(
                f"Window data: O={window_data['open']:.2f} H={window_data['high']:.2f} "
                f"L={window_data['low']:.2f} C={window_data['close']:.2f}"
            )

            historical_windows = storage.get_recent_windows(args.lookback_windows + 1)

            historical_windows.insert(
                0, {"data": window_data, "timestamp": window_data["timestamp"]}
            )
            window_history = [w["data"] for w in historical_windows]

            features = feature_engineer.create_features(window_history)

            prediction = None
            if (
                len(historical_windows) >= args.min_samples
                and model_trainer.model is not None
            ):
                window_info = {
                    "window_id": window_id,
                    "current_price": window_data["close"],
                    "window_start": window_start.isoformat(),
                    "window_end": window_end.isoformat(),
                }

                prediction = predictor.predict(features, window_info)

                if prediction:
                    predictions_made += 1
                    output = predictor.format_output(prediction)
                    print(output)

                    if args.output_format in ["json", "csv"]:
                        output_file = (
                            storage.data_dir / f"predictions.{args.output_format}"
                        )
                        with open(output_file, "a") as f:
                            if args.output_format == "csv" and predictions_made == 1:
                                f.write(predictor.get_csv_header() + "\n")
                            f.write(output + "\n")
            else:
                logger.info(
                    f"Not enough data for prediction yet ({len(historical_windows)}/{args.min_samples})"
                )

            storage.save_window(window_id, window_data, features, prediction)
            windows_processed += 1

            should_retrain = (
                windows_processed - last_retrain >= args.retrain_interval
                and windows_processed >= args.min_samples
            )

            if should_retrain:
                logger.info(
                    f"Retraining model (every {args.retrain_interval} windows)..."
                )
                X, y = storage.get_training_data(args.training_windows)

                if len(X) >= args.min_samples:
                    metrics = model_trainer.train(X, y)
                    if metrics:
                        logger.info(
                            f"Model retrained - Val Accuracy: {metrics['val_accuracy']:.4f}"
                        )
                        model_trainer.save_model()
                        last_retrain = windows_processed
                else:
                    logger.warning(f"Insufficient training data: {len(X)} samples")

            if windows_processed % 10 == 0:
                stats = storage.get_statistics()
                logger.info("")
                logger.info("=" * 60)
                logger.info("SESSION STATISTICS")
                logger.info(f"Windows processed: {windows_processed}")
                logger.info(f"Predictions made: {predictions_made}")
                logger.info(f"Total stored windows: {stats['total_windows']}")
                logger.info(f"Storage size: {stats['storage_size_mb']:.2f} MB")

                perf = predictor.get_recent_performance(20)
                if perf:
                    logger.info(f"Recent performance (last 20):")
                    logger.info(
                        f"  - Traded: {perf['traded']}, Skipped: {perf['skipped']}"
                    )
                    logger.info(f"  - Avg confidence: {perf['avg_confidence']:.2%}")

                logger.info("=" * 60)
                logger.info("")

            if windows_processed % 100 == 0:
                storage.cleanup_old_data(keep_n_windows=5000)

            logger.info("Waiting for next window...")
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info("\nStopping live prediction...")
        logger.info("Saving final model and predictions...")
        model_trainer.save_model()
        predictor.save_predictions(storage.data_dir / "predictions_final.json")

        logger.info("")
        logger.info("=" * 60)
        logger.info("FINAL STATISTICS")
        logger.info(f"Total windows processed: {windows_processed}")
        logger.info(f"Total predictions made: {predictions_made}")
        logger.info("=" * 60)
