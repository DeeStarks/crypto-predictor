# Crypto Price Predictor

A CLI tool for predicting cryptocurrency price movements using gradient boosting models (LightGBM, XGBoost, CatBoost) with real-time data polling and incremental learning.

> **Note:** I am not responsible for any losses incurred by using this tool. Use at your own risk.

## Features

- **Real-time prediction** on fixed time windows (e.g., 15-min, 1-hour)
- **Live data polling** from major exchanges (Binance, Coinbase, Kraken)
- **Incremental learning** - model updates with new data
- **Multiple gradient boosting models** - LightGBM, XGBoost, CatBoost
- **Comprehensive technical indicators** - RSI, MACD, Bollinger Bands, momentum, volatility
- **Confidence-based filtering** - only trade high-confidence predictions
- **Backtesting** - validate on historical data
- **Multiple output formats** - text, JSON, CSV

## Installation

```bash
# Clone the repository
git clone git@github.com:DeeStarks/crypto-predictor.git
cd crypto-predictor

# Install dependencies
pip install -r requirements.txt

# Make main.py executable (optional)
chmod +x main.py
```

## Quick Start

### 1. Initial Training

Train a model on historical data before running live predictions:

```bash
python main.py --mode train --symbol BTCUSDT --window-minutes 15 --model-type lightgbm
```

This will:
- Fetch historical price data from Binance
- Create features from 1000+ historical windows
- Train a LightGBM model
- Save the trained model to `./models/saved/`

### 2. Live Prediction

Run real-time predictions:

```bash
python main.py --mode live --symbol BTCUSDT --window-minutes 15
```

This will:
- Poll live price data every 5 seconds
- Aggregate data at the end of each 15-minute window
- Make predictions for the next window
- Retrain the model every 10 windows
- Display predictions in the console

### 3. Backtesting

Evaluate model performance on historical data:

```bash
python main.py --mode backtest \
  --backtest-start 2026-01-01 \
  --backtest-end 2026-01-31 \
  --confidence-threshold 0.65
```

## Usage Examples

### Basic Commands

**Train with XGBoost on 1-hour windows:**
```bash
python main.py --mode train --symbol ETHUSDT --window-minutes 60 --model-type xgboost
```

**Live prediction with custom confidence threshold:**
```bash
python main.py --mode live --symbol BTCUSDT --confidence-threshold 0.7
```

**Backtest with CatBoost:**
```bash
python main.py --mode backtest --model-type catboost --backtest-start 2026-01-01 --backtest-end 2026-02-01
```

### Advanced Configuration

**Custom feature engineering:**
```bash
python main.py --mode train \
  --symbol BTCUSDT \
  --lookback-windows 30 \
  --technical-indicators rsi macd bbands volume_ratio price_momentum
```

**Live prediction with frequent retraining:**
```bash
python main.py --mode live \
  --symbol BTCUSDT \
  --window-minutes 15 \
  --retrain-interval 5 \
  --training-windows 500
```

**Output to JSON file:**
```bash
python main.py --mode live \
  --symbol BTCUSDT \
  --output-format json \
  --log-file ./logs/btc_predictions.log
```

**Different exchange:**
```bash
python main.py --mode train --symbol BTC-USD --exchange coinbase
```

## Command-Line Arguments

### Core Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--symbol` | BTCUSDT | Trading pair symbol |
| `--exchange` | binance | Exchange (binance, coinbase, kraken) |
| `--mode` | live | Operating mode (live, train, backtest) |
| `--window-minutes` | 15 | Window size in minutes |

### Model Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-type` | lightgbm | Model type (lightgbm, xgboost, catboost) |
| `--training-windows` | 1000 | Historical windows for training |
| `--retrain-interval` | 10 | Retrain every N windows |
| `--min-samples` | 100 | Minimum samples for predictions |

### Feature Engineering

| Argument | Default | Description |
|----------|---------|-------------|
| `--lookback-windows` | 20 | Previous windows for features |
| `--technical-indicators` | rsi macd bbands... | Indicators to calculate |

### Prediction Settings

| Argument | Default | Description |
|----------|---------|-------------|
| `--confidence-threshold` | 0.6 | Minimum prediction confidence |
| `--predict-timing` | end | When to predict (start/end) |
| `--prediction-offset` | 30 | Seconds before window end |

### Data Management

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | ./data/storage | Data storage directory |
| `--model-dir` | ./models/saved | Model save directory |
| `--poll-interval` | 5 | Polling interval (seconds) |

### Output & Logging

| Argument | Default | Description |
|----------|---------|-------------|
| `--output-format` | text | Output format (text, json, csv) |
| `--log-level` | INFO | Logging level |
| `--log-file` | ./logs/predictor.log | Log file path |
| `--quiet` | False | Suppress console output |

### Backtesting

| Argument | Default | Description |
|----------|---------|-------------|
| `--backtest-start` | - | Start date (YYYY-MM-DD) |
| `--backtest-end` | - | End date (YYYY-MM-DD) |

### Advanced Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--enable-ensemble` | False | Use ensemble of models |
| `--regime-detection` | False | Detect market regimes |
| `--feature-selection` | False | Auto feature selection |
| `--dry-run` | False | Testing mode (no predictions) |

## How It Works

### 1. Data Collection
- Polls exchange API at regular intervals (default: 5 seconds)
- Aggregates tick data into OHLCV windows
- Stores historical data locally

### 2. Feature Engineering
Creates 40+ features including:
- **Price features**: changes, position in range, candle patterns
- **Technical indicators**: RSI, MACD, Bollinger Bands
- **Volume features**: ratios, trends, VWAP deviation
- **Momentum indicators**: ROC, price acceleration
- **Volatility measures**: historical volatility, ATR
- **Time features**: hour, day of week, cyclical encoding

### 3. Model Training
- Uses gradient boosting (LightGBM/XGBoost/CatBoost)
- Binary classification: will price go up or down?
- Time-series aware validation
- Early stopping to prevent overfitting
- Regular retraining on recent data

### 4. Prediction
- Makes predictions at window boundaries
- Provides confidence scores (0.5-1.0)
- Filters low-confidence predictions
- Tracks prediction history

## Performance Considerations

### Accuracy Expectations
- Short windows (5-15 min): 52-58% accuracy
- Medium windows (1 hour): 55-62% accuracy
- Long windows (4+ hours): 58-65% accuracy

Higher confidence thresholds improve accuracy but reduce trade frequency.

### Trading Viability
- Account for exchange fees (typically 0.1%)
- Consider slippage in volatile markets
- Use confidence filtering to improve edge
- Focus on Sharpe ratio, not just accuracy

## Troubleshooting

**API rate limits**
- Increase `--poll-interval` to 10-15 seconds
- Use fewer requests per window

**Insufficient training data**
- Reduce `--min-samples` to 50
- Reduce `--training-windows` to 500
- Use longer time intervals for historical data

**Low accuracy**
- Increase `--confidence-threshold` to 0.7+
- Use longer windows (60+ minutes)
- Add more lookback windows
- Try different model types

**Model not improving**
- Collect more data (5000+ windows)
- Check feature importance
- Verify data quality (no gaps/errors)
- Try ensemble mode

## Best Practices

1. **Start with training mode** to build initial model
2. **Use confidence thresholds** ≥ 0.65 for live trading
3. **Backtest thoroughly** before risking capital
4. **Monitor performance** - retrain if accuracy drops
5. **Account for fees** in profit calculations
6. **Start with longer windows** (1+ hours) for better signal
7. **Keep historical data** - enables better retraining

## Limitations

- **Not financial advice** - use at your own risk
- **Past performance ≠ future results**
- **Market conditions change** - requires retraining
- **Short windows are noisy** - limited predictability
- **No guarantee of profit** - trading is risky

## License

MIT License - See LICENSE file
