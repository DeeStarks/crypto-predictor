import logging
import time
import hashlib
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from collections import OrderedDict
from utils.type_conversion import convert_to_native_types

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages caching for API calls and expensive computations."""

    def __init__(self, cache_dir="./data/cache", max_age_seconds=300):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory to store cache files
            max_age_seconds: Maximum age of cached data in seconds
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age = max_age_seconds
        self.memory_cache = OrderedDict()
        self.max_memory_items = 1000

        logger.info(f"Initialized cache manager (max age: {max_age_seconds}s)")

    def _generate_key(self, prefix, params):
        """Generate cache key from parameters."""
        param_str = json.dumps(convert_to_native_types(params), sort_keys=True)
        hash_key = hashlib.md5(param_str.encode()).hexdigest()
        return f"{prefix}_{hash_key}"

    def get(self, key, check_memory=True, check_disk=True):
        """
        Get cached value.

        Args:
            key: Cache key
            check_memory: Check in-memory cache
            check_disk: Check disk cache

        Returns:
            Cached value or None if not found/expired
        """
        if check_memory and key in self.memory_cache:
            data, timestamp = self.memory_cache[key]
            if time.time() - timestamp < self.max_age:
                self.memory_cache.move_to_end(key)
                logger.debug(f"Cache hit (memory): {key}")
                return data
            else:
                del self.memory_cache[key]

        if check_disk:
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, "rb") as f:
                        cached = pickle.load(f)

                    if time.time() - cached["timestamp"] < self.max_age:
                        self._store_memory(key, cached["data"])
                        logger.debug(f"Cache hit (disk): {key}")
                        return cached["data"]
                    else:
                        cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Error reading cache {key}: {e}")

        return None

    def set(self, key, data, store_memory=True, store_disk=True):
        """
        Store value in cache.

        Args:
            key: Cache key
            data: Data to cache
            store_memory: Store in memory cache
            store_disk: Store on disk
        """
        timestamp = time.time()

        if store_memory:
            self._store_memory(key, data)

        if store_disk:
            cache_file = self.cache_dir / f"{key}.pkl"
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump({"data": data, "timestamp": timestamp}, f)
                logger.debug(f"Cached to disk: {key}")
            except Exception as e:
                logger.warning(f"Error writing cache {key}: {e}")

    def _store_memory(self, key, data):
        """Store in memory cache with LRU eviction."""
        self.memory_cache[key] = (data, time.time())

        if len(self.memory_cache) > self.max_memory_items:
            self.memory_cache.popitem(last=False)

    def invalidate(self, key=None, prefix=None):
        """
        Invalidate cache entries.

        Args:
            key: Specific key to invalidate
            prefix: Invalidate all keys with this prefix
        """
        if key:
            self.memory_cache.pop(key, None)
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                cache_file.unlink()
            logger.debug(f"Invalidated cache: {key}")

        elif prefix:
            to_remove = [k for k in self.memory_cache.keys() if k.startswith(prefix)]
            for k in to_remove:
                del self.memory_cache[k]

            for cache_file in self.cache_dir.glob(f"{prefix}_*.pkl"):
                cache_file.unlink()

            logger.debug(f"Invalidated cache prefix: {prefix}")

    def clear_expired(self):
        """Remove all expired cache entries."""
        current_time = time.time()

        to_remove = [
            k
            for k, (_, ts) in self.memory_cache.items()
            if current_time - ts >= self.max_age
        ]
        for k in to_remove:
            del self.memory_cache[k]

        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                with open(cache_file, "rb") as f:
                    cached = pickle.load(f)
                if current_time - cached["timestamp"] >= self.max_age:
                    cache_file.unlink()
            except:
                pass

        logger.info(f"Cleared {len(to_remove)} expired cache entries")

    def get_stats(self):
        """Get cache statistics."""
        return {
            "memory_entries": len(self.memory_cache),
            "disk_entries": len(list(self.cache_dir.glob("*.pkl"))),
            "cache_size_mb": sum(f.stat().st_size for f in self.cache_dir.glob("*.pkl"))
            / 1024
            / 1024,
        }


class CachedDataCollector:
    """Wrapper for DataCollector with caching."""

    def __init__(self, data_collector, cache_manager):
        """
        Initialize cached data collector.

        Args:
            data_collector: DataCollector instance
            cache_manager: CacheManager instance
        """
        self.collector = data_collector
        self.cache = cache_manager

    def fetch_ticker(self):
        """Fetch ticker with caching."""
        cache_key = self.cache._generate_key(
            "ticker",
            {"exchange": self.collector.exchange, "symbol": self.collector.symbol},
        )

        cached = self.cache.get(cache_key)
        if cached:
            return cached

        data = self.collector.fetch_ticker()
        if data:
            self.cache.set(cache_key, data)

        return data

    def fetch_orderbook(self, depth=20):
        """Fetch orderbook with caching."""
        cache_key = self.cache._generate_key(
            "orderbook",
            {
                "exchange": self.collector.exchange,
                "symbol": self.collector.symbol,
                "depth": depth,
            },
        )

        cached = self.cache.get(cache_key)
        if cached:
            return cached

        data = self.collector.fetch_orderbook(depth)
        if data:
            self.cache.set(cache_key, data)

        return data

    def fetch_historical_klines(self, interval="1m", limit=500):
        """Fetch historical klines with longer caching."""
        cache_key = self.cache._generate_key(
            "klines",
            {
                "exchange": self.collector.exchange,
                "symbol": self.collector.symbol,
                "interval": interval,
                "limit": limit,
            },
        )

        cached = self.cache.get(cache_key, check_disk=True)
        if cached:
            return cached

        data = self.collector.fetch_historical_klines(interval, limit)
        if data:
            self.cache.set(cache_key, data, store_memory=False, store_disk=True)

        return data
