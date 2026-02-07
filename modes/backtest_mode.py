import logging
from datetime import datetime, timedelta
import numpy as np
from utils.type_conversion import convert_to_native_types

logger = logging.getLogger(__name__)


def run_backtest(args, storage, feature_engineer, model_trainer, predictor):
    """
    Run backtest mode to evaluate model on historical data.

    Args:
        args: Command-line arguments
        storage: Data storage instance
        feature_engineer: Feature engineer instance
        model_trainer: Model trainer instance
        predictor: Predictor instance
    """
    logger.info("Starting BACKTEST mode")

    model_files = list(model_trainer.model_dir.glob(f"{args.model_type}_*.pkl"))
    if not model_files:
        logger.error("No trained model found. Run training mode first.")
        return

    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Loading model: {latest_model}")
    model_trainer.load_model(latest_model)

    start_date = (
        datetime.fromisoformat(args.backtest_start) if args.backtest_start else None
    )
    end_date = datetime.fromisoformat(args.backtest_end) if args.backtest_end else None

    logger.info("Loading historical data...")
    all_windows = storage.get_recent_windows(n=10000)

    if start_date or end_date:
        filtered_windows = []
        for window in all_windows:
            window_time = datetime.fromisoformat(window["timestamp"]).replace(
                tzinfo=None
            )
            if start_date and window_time < start_date:
                continue
            if end_date and window_time > end_date:
                continue
            filtered_windows.append(window)
        windows = filtered_windows
    else:
        windows = all_windows

    if len(windows) < args.min_samples:
        logger.error(f"Insufficient historical data: {len(windows)} windows")
        return

    logger.info(f"Backtesting on {len(windows)} windows")
    if windows:
        logger.info(
            f"Date range: {windows[-1]['timestamp']} to {windows[0]['timestamp']}"
        )

    logger.info("Running backtest...")

    predictions = []
    actuals = []
    features_list = []

    all_features = storage.load_all_features()

    for i in range(args.lookback_windows, len(windows) - 1):
        historical = windows[i - args.lookback_windows : i + 1]
        historical_data = [w["data"] for w in historical]

        window_id = windows[i]["window_id"]
        if window_id in all_features:
            features = all_features[window_id]
        else:
            features = feature_engineer.create_features(historical_data)
            if features is None:
                continue

        prediction = model_trainer.predict(features)
        if prediction:
            predictions.append(prediction)

            current_close = windows[i]["data"]["close"]
            next_close = windows[i + 1]["data"]["close"]
            actual = "up" if next_close > current_close else "down"
            actuals.append(actual)

            features_list.append(
                {
                    "window_id": window_id,
                    "timestamp": windows[i]["timestamp"],
                    "current_price": current_close,
                    "next_price": next_close,
                    "price_change": ((next_close - current_close) / current_close)
                    * 100,
                }
            )

    if not predictions:
        logger.error("No predictions generated")
        return

    logger.info(f"Generated {len(predictions)} predictions")

    logger.info("Evaluating performance...")

    correct = sum(1 for p, a in zip(predictions, actuals) if p["direction"] == a)
    accuracy = correct / len(predictions)

    high_conf_predictions = [
        (p, a)
        for p, a in zip(predictions, actuals)
        if p["confidence"] >= args.confidence_threshold
    ]

    if high_conf_predictions:
        high_conf_correct = sum(
            1 for p, a in high_conf_predictions if p["direction"] == a
        )
        high_conf_accuracy = high_conf_correct / len(high_conf_predictions)
    else:
        high_conf_accuracy = 0
        high_conf_correct = 0

    returns = []
    for pred, actual, info in zip(predictions, actuals, features_list):
        if pred["confidence"] >= args.confidence_threshold:
            price_change = info["price_change"]

            if pred["direction"] == actual:
                returns.append(abs(price_change) - args.fee_rate * 200)
            else:
                returns.append(-abs(price_change) - args.fee_rate * 200)

    total_return = sum(returns) if returns else 0
    avg_return = np.mean(returns) if returns else 0
    sharpe_ratio = (
        (
            np.mean(returns)
            / np.std(returns)
            * np.sqrt(365 * 24 * (60 / args.window_minutes))
        )
        if returns and np.std(returns) > 0
        else 0
    )

    wins = sum(1 for r in returns if r > 0)
    losses = sum(1 for r in returns if r < 0)
    win_rate = wins / len(returns) if returns else 0

    logger.info("")
    logger.info("=" * 60)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total predictions: {len(predictions)}")
    logger.info(f"Overall accuracy: {accuracy:.2%} ({correct}/{len(predictions)})")
    logger.info("")
    logger.info(
        f"High confidence predictions (>={args.confidence_threshold}): {len(high_conf_predictions)}"
    )
    logger.info(
        f"High confidence accuracy: {high_conf_accuracy:.2%} ({high_conf_correct}/{len(high_conf_predictions)})"
    )
    logger.info("")
    logger.info("TRADING PERFORMANCE (High Confidence Only):")
    logger.info(f"  Total trades: {len(returns)}")
    logger.info(f"  Total return: {total_return:.2f}%")
    logger.info(f"  Average return per trade: {avg_return:.3f}%")
    logger.info(f"  Win rate: {win_rate:.2%} ({wins}/{len(returns)})")
    logger.info(f"  Sharpe ratio: {sharpe_ratio:.2f}")
    logger.info("")

    up_preds = sum(1 for p in predictions if p["direction"] == "up")
    down_preds = len(predictions) - up_preds
    logger.info(f"Prediction distribution:")
    logger.info(f"  Up predictions: {up_preds} ({up_preds/len(predictions)*100:.1f}%)")
    logger.info(
        f"  Down predictions: {down_preds} ({down_preds/len(predictions)*100:.1f}%)"
    )
    logger.info("")

    confidences = [p["confidence"] for p in predictions]
    logger.info(f"Confidence statistics:")
    logger.info(f"  Mean: {np.mean(confidences):.3f}")
    logger.info(f"  Median: {np.median(confidences):.3f}")
    logger.info(f"  Std: {np.std(confidences):.3f}")
    logger.info("=" * 60)

    results_file = storage.data_dir / "backtest_results.json"
    import json

    detailed_results = {
        "backtest_date": datetime.now().isoformat(),
        "parameters": {
            "symbol": args.symbol,
            "window_minutes": args.window_minutes,
            "confidence_threshold": args.confidence_threshold,
            "date_range": {"start": args.backtest_start, "end": args.backtest_end},
        },
        "metrics": {
            "total_predictions": len(predictions),
            "overall_accuracy": accuracy,
            "high_confidence_predictions": len(high_conf_predictions),
            "high_confidence_accuracy": high_conf_accuracy,
            "total_return_pct": total_return,
            "avg_return_pct": avg_return,
            "win_rate": win_rate,
            "sharpe_ratio": sharpe_ratio,
            "total_trades": len(returns),
            "wins": wins,
            "losses": losses,
        },
        "predictions": [
            {
                **pred,
                "actual": actual,
                "window_info": info,
                "return_pct": ret if i < len(returns) else None,
            }
            for i, (pred, actual, info, ret) in enumerate(
                zip(
                    predictions,
                    actuals,
                    features_list,
                    returns + [None] * (len(predictions) - len(returns)),
                )
            )
        ],
    }

    with open(results_file, "w") as f:
        json.dump(convert_to_native_types(detailed_results), f, indent=2)

    logger.info(f"Detailed results saved to: {results_file}")

    csv_file = storage.data_dir / "backtest_trades.csv"
    with open(csv_file, "w") as f:
        f.write(
            "timestamp,window_id,predicted,actual,confidence,current_price,next_price,price_change_pct,return_pct\n"
        )
        for pred, actual, info in zip(predictions, actuals, features_list):
            if pred["confidence"] >= args.confidence_threshold:
                idx = features_list.index(info)
                ret = (
                    returns[
                        sum(
                            1
                            for p in predictions[:idx]
                            if p["confidence"] >= args.confidence_threshold
                        )
                    ]
                    if idx < len(returns)
                    else 0
                )
                f.write(
                    f"{info['timestamp']},{info['window_id']},{pred['direction']},{actual},"
                    f"{pred['confidence']:.4f},{info['current_price']:.2f},{info['next_price']:.2f},"
                    f"{info['price_change']:.4f},{ret:.4f}\n"
                )

    logger.info(f"Trade log saved to: {csv_file}")
