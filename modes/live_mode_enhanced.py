import logging
import time
from datetime import datetime
import numpy as np
from rich.live import Live
from rich.console import Console

from utils.cache_manager import CacheManager, CachedDataCollector
from utils.overfitting_prevention import OverfittingPreventor, DataLeakageDetector
from utils.sentiment_news import NewsChecker, SentimentAnalyzer, MarketSentimentFeatures
from utils.rich_display import RichDisplay

logger = logging.getLogger(__name__)
console = Console()


def calculate_market_insights(window_data_history, features):
    """Calculate market insights from data."""
    if len(window_data_history) < 5:
        return {}

    insights = {}

    recent_closes = [w["close"] for w in window_data_history[-10:]]
    price_change = (recent_closes[-1] - recent_closes[0]) / recent_closes[0]

    if price_change > 0.02:
        insights["trend"] = "bullish"
    elif price_change < -0.02:
        insights["trend"] = "bearish"
    else:
        insights["trend"] = "neutral"

    returns = np.diff(recent_closes) / recent_closes[:-1]
    insights["volatility"] = np.std(returns)

    insights["momentum"] = price_change

    recent_highs = [w["high"] for w in window_data_history[-20:]]
    recent_lows = [w["low"] for w in window_data_history[-20:]]
    insights["resistance"] = max(recent_highs)
    insights["support"] = min(recent_lows)

    if "rsi" in features:
        rsi = features["rsi"]
        if rsi > 70:
            insights["rsi_signal"] = "Overbought"
        elif rsi < 30:
            insights["rsi_signal"] = "Oversold"
        else:
            insights["rsi_signal"] = "Neutral"

    return insights


def calculate_probability_of_beating(
    price_to_beat, current_price, prediction, historical_accuracy
):
    """Calculate probability of beating the price to beat."""
    if not prediction:
        return None

    base_prob = (
        prediction["probability_up"]
        if prediction["direction"] == "up"
        else (1 - prediction["probability_up"])
    )

    current_diff = current_price - price_to_beat

    if prediction["direction"] == "up" and current_diff > 0:
        adjustment = min(0.1, abs(current_diff / price_to_beat))
        prob_beating = min(0.95, base_prob + adjustment)
    elif prediction["direction"] == "down" and current_diff < 0:
        adjustment = min(0.1, abs(current_diff / price_to_beat))
        prob_beating = min(0.95, base_prob + adjustment)
    else:
        prob_beating = base_prob * 0.95

    if historical_accuracy:
        prob_beating = prob_beating * (historical_accuracy / 0.5)

    return max(0.05, min(0.95, prob_beating))


def calculate_confidence_interval(prediction, historical_std=0.1):
    """Calculate confidence interval for prediction probability."""
    if not prediction:
        return None

    prob = prediction["probability_up"]

    margin = 1.96 * historical_std

    lower = max(0.0, prob - margin)
    upper = min(1.0, prob + margin)

    return (lower, upper)


def run_live_prediction_enhanced(
    args, storage, collector, feature_engineer, model_trainer, predictor, window_manager
):
    """
    Run enhanced live prediction mode with rich display and advanced features.

    Args:
        args: Command-line arguments
        storage: Data storage instance
        collector: Data collector instance
        feature_engineer: Feature engineer instance
        model_trainer: Model trainer instance
        predictor: Predictor instance
        window_manager: Window manager instance
    """
    logger.info("Starting ENHANCED LIVE PREDICTION mode")
    logger.info(f"Symbol: {args.symbol}, Exchange: {args.exchange}")
    logger.info(f"Window: {args.window_minutes} minutes")

    cache_manager = CacheManager(
        cache_dir="./data/cache", max_age_seconds=args.poll_interval
    )
    cached_collector = CachedDataCollector(collector, cache_manager)

    overfitting_preventor = OverfittingPreventor(patience=10, min_delta=0.001)
    data_leakage_detector = DataLeakageDetector()

    symbol_base = args.symbol.replace("USDT", "").replace("USD", "")
    sentiment_analyzer = SentimentAnalyzer(cache_manager)
    news_checker = NewsChecker(cache_manager, sentiment_analyzer)
    sentiment_features = MarketSentimentFeatures(news_checker)

    rich_display = RichDisplay()

    model_files = list(model_trainer.model_dir.glob(f"{args.model_type}_*.pkl"))
    if model_files:
        latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Loading existing model: {latest_model}")
        model_trainer.load_model(latest_model)
    else:
        logger.info("No existing model found - will train on first window completion")

    windows_processed = 0
    predictions_made = 0
    correct_predictions = 0
    last_retrain = 0
    prediction_outcomes = []

    current_window_start_price = None
    latest_tick_price = None

    try:
        with Live(
            rich_display.create_header(args.symbol, {}),
            refresh_per_second=1,
            console=console,
            screen=True,
        ) as live:
            while True:
                window_start, window_end = window_manager.get_current_window()
                window_id = window_manager.get_window_id(window_start)

                logger.debug(
                    f"\nCurrent window: {window_manager.format_window_display(window_start, window_end)}"
                )

                ticks = []
                end_time = window_end.timestamp()

                current_window_start_price = None
                try:
                    interval_map = {1: "1m", 5: "5m", 15: "15m", 30: "30m", 60: "1h"}
                    interval = interval_map.get(args.window_minutes, "15m")

                    klines = collector.fetch_historical_klines(
                        interval=interval, limit=2
                    )
                    if klines:
                        for kline in klines:
                            time_diff = abs(
                                (kline["timestamp"] - window_start).total_seconds()
                            )
                            if time_diff < 60:
                                current_window_start_price = kline["open"]
                                logger.debug(
                                    f"Found window opening price from historical data: ${current_window_start_price:,.2f}"
                                )
                                break
                except Exception as e:
                    logger.warning(f"Could not fetch historical opening price: {e}")

                if current_window_start_price is None:
                    first_tick = cached_collector.fetch_ticker()
                    if first_tick:
                        current_window_start_price = first_tick["price"]
                        ticks.append(first_tick)
                        logger.warning(
                            f"Using first tick as price to beat (historical data unavailable): ${current_window_start_price:,.2f}"
                        )

                if current_window_start_price is not None:
                    logger.debug(
                        f"Price to beat this window: ${current_window_start_price:,.2f}"
                    )
                else:
                    logger.warning("Price to beat unavailable for this window")

                while time.time() < end_time:
                    tick = cached_collector.fetch_ticker()
                    if tick:
                        ticks.append(tick)
                        latest_tick_price = tick["price"]

                        historical_windows = storage.get_recent_windows(
                            args.lookback_windows + 1
                        )
                        window_history = [w["data"] for w in historical_windows]

                        if len(window_history) >= 5:
                            features = feature_engineer.create_features(window_history)

                            try:
                                sent_features = (
                                    sentiment_features.create_sentiment_features(
                                        symbol_base
                                    )
                                )
                                if features and sent_features:
                                    features.update(sent_features)
                            except Exception as e:
                                logger.warning(
                                    f"Could not fetch sentiment features: {e}"
                                )

                            prediction = None
                            prob_beating = None
                            confidence_interval = None

                            if (
                                len(historical_windows) >= args.min_samples
                                and model_trainer.model is not None
                            ):
                                window_info = {
                                    "window_id": window_id,
                                    "current_price": latest_tick_price,
                                    "window_start": window_start.isoformat(),
                                    "window_end": window_end.isoformat(),
                                }

                                if (
                                    "current_window_prediction" not in locals()
                                    or current_window_id != window_id
                                ):
                                    current_window_id = window_id
                                    current_window_prediction = model_trainer.predict(
                                        features
                                    )

                                    if current_window_prediction:
                                        current_window_prediction["timestamp"] = (
                                            datetime.now().isoformat()
                                        )
                                        current_window_prediction["window_info"] = (
                                            window_info
                                        )

                                        if (
                                            current_window_prediction["confidence"]
                                            < predictor.confidence_threshold
                                        ):
                                            current_window_prediction["action"] = "skip"
                                            current_window_prediction["reason"] = (
                                                "low_confidence"
                                            )
                                        else:
                                            current_window_prediction["action"] = (
                                                "trade"
                                            )
                                            predictions_made += 1

                                        predictor.prediction_history.append(
                                            current_window_prediction
                                        )

                                prediction = current_window_prediction

                                if prediction:
                                    historical_accuracy = correct_predictions / max(
                                        1, predictions_made
                                    )
                                    prob_beating = calculate_probability_of_beating(
                                        current_window_start_price,
                                        latest_tick_price,
                                        prediction,
                                        historical_accuracy,
                                    )
                                    prediction["prob_beating"] = prob_beating

                                    historical_std = (
                                        np.std(
                                            [
                                                p.get("probability_up", 0.5)
                                                for p in predictor.prediction_history[
                                                    -20:
                                                ]
                                            ]
                                        )
                                        if len(predictor.prediction_history) >= 20
                                        else 0.1
                                    )
                                    confidence_interval = calculate_confidence_interval(
                                        prediction, historical_std
                                    )

                            insights = calculate_market_insights(
                                window_history, features
                            )

                            sentiment_data = None
                            try:
                                sentiment_data = news_checker.analyze_news_sentiment(
                                    symbol_base, hours=24
                                )
                            except:
                                pass

                            display_data = {
                                "symbol": args.symbol,
                                "window_info": {
                                    "window_id": window_manager.format_window_display(
                                        window_start, window_end
                                    )
                                },
                                "price_to_beat": current_window_start_price,
                                "current_price": latest_tick_price,
                                "prediction": prediction,
                                "prob_beating": prob_beating,
                                "confidence_interval": confidence_interval,
                                "stats": {
                                    "windows_processed": windows_processed,
                                    "predictions_made": predictions_made,
                                    "accuracy": correct_predictions
                                    / max(1, predictions_made),
                                    "win_rate": correct_predictions
                                    / max(1, predictions_made),
                                    "avg_confidence": (
                                        np.mean(
                                            [
                                                p.get("confidence", 0)
                                                for p in predictor.prediction_history[
                                                    -20:
                                                ]
                                            ]
                                        )
                                        if predictor.prediction_history
                                        else 0
                                    ),
                                },
                                "insights": insights,
                                "sentiment": sentiment_data,
                                "history": predictor.prediction_history,
                            }

                            live.update(
                                rich_display.display_live_dashboard(display_data)
                            )

                    time_remaining = end_time - time.time()
                    if time_remaining > args.poll_interval:
                        time.sleep(args.poll_interval)
                    elif time_remaining > 0:
                        time.sleep(time_remaining)

                window_data = collector.aggregate_window_ticks(ticks)
                logger.info(
                    f"Window complete: O={window_data['open']:.2f} H={window_data['high']:.2f} L={window_data['low']:.2f} C={window_data['close']:.2f}"
                )

                if len(predictor.prediction_history) > 0:
                    last_prediction = predictor.prediction_history[-1]
                    actual_direction = (
                        "up"
                        if window_data["close"] > current_window_start_price
                        else "down"
                    )

                    if last_prediction["direction"] == actual_direction:
                        correct_predictions += 1

                    last_prediction["actual"] = actual_direction
                    prediction_outcomes.append(
                        {
                            "predicted": last_prediction["direction"],
                            "actual": actual_direction,
                            "correct": last_prediction["direction"] == actual_direction,
                        }
                    )

                storage.save_window(
                    window_id,
                    window_data,
                    features,
                    prediction if "prediction" in locals() else None,
                )
                windows_processed += 1

                if windows_processed > 0 and predictions_made > 10:
                    recent_accuracy = sum(
                        1 for o in prediction_outcomes[-10:] if o["correct"]
                    ) / min(10, len(prediction_outcomes))
                    if model_trainer.model:
                        overfitting_status = overfitting_preventor.detect_overfitting(
                            train_score=0.7,
                            val_score=recent_accuracy,
                        )

                        if overfitting_status["is_overfitting"]:
                            logger.warning(
                                "Overfitting detected - consider retraining with regularization"
                            )

                should_retrain = (
                    windows_processed - last_retrain >= args.retrain_interval
                    and windows_processed >= args.min_samples
                )

                if should_retrain:
                    logger.info(f"Retraining model...")
                    X, y = storage.get_training_data(args.training_windows)

                    if len(X) >= args.min_samples:
                        metrics = model_trainer.train(X, y)
                        if metrics:
                            overfitting_status = (
                                overfitting_preventor.detect_overfitting(
                                    metrics["train_accuracy"], metrics["val_accuracy"]
                                )
                            )

                            if overfitting_status["is_overfitting"]:
                                suggestions = (
                                    overfitting_preventor.suggest_regularization(
                                        overfitting_status
                                    )
                                )
                                logger.warning(
                                    f"Overfitting detected. Suggestions: {suggestions}"
                                )

                            model_trainer.save_model()
                            last_retrain = windows_processed

                if windows_processed % 20 == 0:
                    cache_manager.clear_expired()

                time.sleep(2)
    except KeyboardInterrupt:
        logger.info("\nStopping live prediction...")

        model_trainer.save_model()
        predictor.save_predictions(storage.data_dir / "predictions_final.json")

        logger.info("\n" + "=" * 60)
        logger.info("FINAL STATISTICS")
        logger.info(f"Total windows processed: {windows_processed}")
        logger.info(f"Total predictions made: {predictions_made}")
        logger.info(f"Correct predictions: {correct_predictions}")
        logger.info(
            f"Final accuracy: {correct_predictions / max(1, predictions_made):.2%}"
        )
        logger.info("=" * 60)
