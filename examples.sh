#!/bin/bash

echo "Crypto Price Predictor - Example Commands"
echo "=========================================="
echo ""

echo "1. Train a model on Bitcoin with 15-minute windows:"
echo "   python main.py --mode train --symbol BTCUSDT --window-minutes 15"
echo ""

echo "2. Run live predictions on Ethereum:"
echo "   python main.py --mode live --symbol ETHUSDT --window-minutes 15 --confidence-threshold 0.65"
echo ""

echo "3. Backtest on February 2026 data:"
echo "   python main.py --mode backtest --backtest-start 2026-02-01 --backtest-end 2026-02-06"
echo ""

echo "4. Train with XGBoost on 1-hour windows:"
echo "   python main.py --mode train --symbol BTCUSDT --window-minutes 60 --model-type xgboost --training-windows 2000"
echo ""

echo "5. Live prediction with JSON output:"
echo "   python main.py --mode live --symbol BTCUSDT --output-format json --log-level DEBUG"
echo ""

echo "6. High-frequency with 5-minute windows:"
echo "   python main.py --mode live --symbol BTCUSDT --window-minutes 5 --poll-interval 2 --confidence-threshold 0.7"
echo ""

echo "7. Conservative trading with high confidence:"
echo "   python main.py --mode live --symbol BTCUSDT --confidence-threshold 0.75 --retrain-interval 20"
echo ""

echo "8. Multiple cryptocurrencies (run in separate terminals):"
echo "   python main.py --mode live --symbol BTCUSDT --data-dir ./data/btc"
echo "   python main.py --mode live --symbol ETHUSDT --data-dir ./data/eth"
echo ""

echo "9. Backtest with detailed analysis:"
echo "   python main.py --mode backtest --backtest-start 2026-02-01 --backtest-end 2026-02-06 --confidence-threshold 0.7 --output-format csv"
echo ""

echo "10. Training with custom features:"
echo "   python main.py --mode train --symbol BTCUSDT --lookback-windows 30 --technical-indicators rsi macd bbands volume_ratio"
echo ""

echo ""
echo "For full list of options: python main.py --help"
