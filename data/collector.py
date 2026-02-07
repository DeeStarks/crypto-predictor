import logging
import time
from datetime import datetime
import requests
import json
from collections import deque
import pytz

logger = logging.getLogger(__name__)


class DataCollector:
    """Collects real-time cryptocurrency data from exchanges."""

    def __init__(self, exchange="binance", symbol="BTCUSDT", poll_interval=5):
        """
        Initialize data collector.

        Args:
            exchange: Exchange name (binance, coinbase, kraken)
            symbol: Trading pair symbol
            poll_interval: Polling interval in seconds
        """
        self.exchange = exchange.lower()
        self.symbol = symbol
        self.poll_interval = poll_interval
        self.tick_buffer = deque(maxlen=10000)
        self.api_config = {
            "binance": {
                "ticker": "https://api.binance.com/api/v3/ticker/24hr",
                "orderbook": "https://api.binance.com/api/v3/depth",
                "trades": "https://api.binance.com/api/v3/trades",
                "klines": "https://api.binance.com/api/v3/klines",
            },
            "coinbase": {
                "ticker": "https://api.exchange.coinbase.com/products/{symbol}/ticker",
                "orderbook": "https://api.exchange.coinbase.com/products/{symbol}/book",
                "trades": "https://api.exchange.coinbase.com/products/{symbol}/trades",
            },
            "kraken": {
                "ticker": "https://api.kraken.com/0/public/Ticker",
                "orderbook": "https://api.kraken.com/0/public/Depth",
                "trades": "https://api.kraken.com/0/public/Trades",
            },
        }

        if self.exchange not in self.api_config:
            raise ValueError(f"Unsupported exchange: {exchange}")

        logger.info(f"Initialized {exchange} data collector for {symbol}")

    def fetch_ticker(self):
        """
        Fetch current ticker data (price, volume, etc.).

        Returns:
            dict: Ticker data with price, volume, high, low, etc.
        """
        try:
            if self.exchange == "binance":
                url = self.api_config["binance"]["ticker"]
                params = {"symbol": self.symbol}
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                return {
                    "timestamp": datetime.now(pytz.UTC),
                    "price": float(data["lastPrice"]),
                    "bid": float(data["bidPrice"]),
                    "ask": float(data["askPrice"]),
                    "volume": float(data["volume"]),
                    "high_24h": float(data["highPrice"]),
                    "low_24h": float(data["lowPrice"]),
                    "price_change_24h": float(data["priceChangePercent"]),
                    "quote_volume": float(data["quoteVolume"]),
                }

            elif self.exchange == "coinbase":
                url = self.api_config["coinbase"]["ticker"].format(
                    symbol=self._format_symbol_coinbase()
                )
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()

                return {
                    "timestamp": datetime.now(pytz.UTC),
                    "price": float(data["price"]),
                    "bid": float(data["bid"]),
                    "ask": float(data["ask"]),
                    "volume": float(data["volume"]),
                }

            elif self.exchange == "kraken":
                url = self.api_config["kraken"]["ticker"]
                params = {"pair": self._format_symbol_kraken()}
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                pair_data = list(data["result"].values())[0]

                return {
                    "timestamp": datetime.now(pytz.UTC),
                    "price": float(pair_data["c"][0]),
                    "bid": float(pair_data["b"][0]),
                    "ask": float(pair_data["a"][0]),
                    "volume": float(pair_data["v"][1]),
                    "high_24h": float(pair_data["h"][1]),
                    "low_24h": float(pair_data["l"][1]),
                }

        except Exception as e:
            logger.error(f"Error fetching ticker data: {e}")
            return None

    def fetch_orderbook(self, depth=20):
        """
        Fetch order book data.

        Args:
            depth: Number of order book levels to fetch

        Returns:
            dict: Order book with bids and asks
        """
        try:
            if self.exchange == "binance":
                url = self.api_config["binance"]["orderbook"]
                params = {"symbol": self.symbol, "limit": depth}
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                return {
                    "timestamp": datetime.now(pytz.UTC),
                    "bids": [[float(p), float(q)] for p, q in data["bids"]],
                    "asks": [[float(p), float(q)] for p, q in data["asks"]],
                }

        except Exception as e:
            logger.error(f"Error fetching orderbook: {e}")
            return None

    def fetch_recent_trades(self, limit=100):
        """Fetch recent trades."""
        try:
            if self.exchange == "binance":
                url = self.api_config["binance"]["trades"]
                params = {"symbol": self.symbol, "limit": limit}
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                return [
                    {
                        "timestamp": datetime.fromtimestamp(t["time"] / 1000, pytz.UTC),
                        "price": float(t["price"]),
                        "quantity": float(t["qty"]),
                        "is_buyer_maker": t["isBuyerMaker"],
                    }
                    for t in data
                ]

        except Exception as e:
            logger.error(f"Error fetching trades: {e}")
            return []

    def fetch_historical_klines(self, interval="1m", limit=500):
        """
        Fetch historical candlestick data.

        Args:
            interval: Candlestick interval (1m, 5m, 15m, 1h, etc.)
            limit: Number of candles to fetch

        Returns:
            list: Historical OHLCV data
        """
        try:
            if self.exchange == "binance":
                url = self.api_config["binance"]["klines"]
                params = {"symbol": self.symbol, "interval": interval, "limit": limit}
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                return [
                    {
                        "timestamp": datetime.fromtimestamp(k[0] / 1000, pytz.UTC),
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5]),
                        "close_time": datetime.fromtimestamp(k[6] / 1000, pytz.UTC),
                        "quote_volume": float(k[7]),
                        "trades": int(k[8]),
                    }
                    for k in data
                ]

        except Exception as e:
            logger.error(f"Error fetching historical klines: {e}")
            return []

    def collect_tick(self):
        """
        Collect a single tick of data (price, volume, orderbook).

        Returns:
            dict: Tick data
        """
        ticker = self.fetch_ticker()
        if ticker:
            self.tick_buffer.append(ticker)
            return ticker
        return None

    def collect_window_data(self, window_duration_seconds):
        """
        Collect data for an entire window by polling at intervals.

        Args:
            window_duration_seconds: Duration of window in seconds

        Returns:
            list: All ticks collected during the window
        """
        ticks = []
        start_time = time.time()
        end_time = start_time + window_duration_seconds

        logger.info(f"Collecting data for {window_duration_seconds}s window...")

        while time.time() < end_time:
            tick = self.collect_tick()
            if tick:
                ticks.append(tick)

            # sleep until next poll
            time_remaining = end_time - time.time()
            if time_remaining > self.poll_interval:
                time.sleep(self.poll_interval)
            elif time_remaining > 0:
                time.sleep(time_remaining)

        logger.info(f"Collected {len(ticks)} ticks")
        return ticks

    def aggregate_window_ticks(self, ticks):
        """
        Aggregate tick data into window-level OHLCV.

        Args:
            ticks: List of tick data

        Returns:
            dict: Aggregated window data
        """
        if not ticks:
            return None

        prices = [t["price"] for t in ticks]
        volumes = [t.get("volume", 0) for t in ticks]

        return {
            "timestamp": ticks[-1]["timestamp"],
            "open": prices[0],
            "high": max(prices),
            "low": min(prices),
            "close": prices[-1],
            "volume": sum(volumes) / len(volumes),
            "vwap": (
                sum(p * v for p, v in zip(prices, volumes)) / sum(volumes)
                if sum(volumes) > 0
                else prices[-1]
            ),
            "num_ticks": len(ticks),
            "price_range": max(prices) - min(prices),
            "spread": ticks[-1].get("ask", ticks[-1]["price"])
            - ticks[-1].get("bid", ticks[-1]["price"]),
        }

    def _format_symbol_coinbase(self):
        if "USDT" in self.symbol:
            base = self.symbol.replace("USDT", "")
            return f"{base}-USD"
        return self.symbol

    def _format_symbol_kraken(self):
        if "USDT" in self.symbol:
            base = self.symbol.replace("USDT", "")
            return f"X{base}ZUSD"
        return self.symbol
