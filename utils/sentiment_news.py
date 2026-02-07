import logging
import requests
from datetime import datetime, timedelta
import re
from collections import Counter

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Analyzes sentiment from news and social media."""

    def __init__(self, cache_manager=None):
        """
        Initialize sentiment analyzer.

        Args:
            cache_manager: Optional cache manager for API calls
        """
        self.cache = cache_manager
        self.sentiment_keywords = {
            "positive": [
                "bullish",
                "moon",
                "rally",
                "surge",
                "breakout",
                "gain",
                "profit",
                "adoption",
                "partnership",
                "upgrade",
                "innovation",
                "breakthrough",
                "optimistic",
                "positive",
                "growth",
                "strong",
                "buy",
                "accumulate",
            ],
            "negative": [
                "bearish",
                "crash",
                "dump",
                "drop",
                "fall",
                "loss",
                "decline",
                "scam",
                "hack",
                "regulation",
                "ban",
                "crisis",
                "concern",
                "pessimistic",
                "negative",
                "sell",
                "warning",
                "risk",
                "collapse",
            ],
            "neutral": ["stable", "consolidate", "sideways", "range", "hold", "watch"],
        }

        logger.info("Initialized sentiment analyzer")

    def analyze_text(self, text):
        """
        Analyze sentiment of text using keyword matching.

        Args:
            text: Text to analyze

        Returns:
            dict: Sentiment scores and classification
        """
        if not text:
            return {"sentiment": "neutral", "score": 0, "confidence": 0}

        text_lower = text.lower()

        positive_count = sum(
            1 for word in self.sentiment_keywords["positive"] if word in text_lower
        )
        negative_count = sum(
            1 for word in self.sentiment_keywords["negative"] if word in text_lower
        )
        neutral_count = sum(
            1 for word in self.sentiment_keywords["neutral"] if word in text_lower
        )

        total = positive_count + negative_count + neutral_count

        if total == 0:
            return {"sentiment": "neutral", "score": 0, "confidence": 0}

        score = (positive_count - negative_count) / total if total > 0 else 0

        if score > 0.2:
            sentiment = "positive"
        elif score < -0.2:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        confidence = abs(score)

        return {
            "sentiment": sentiment,
            "score": score,
            "confidence": confidence,
            "positive_mentions": positive_count,
            "negative_mentions": negative_count,
            "neutral_mentions": neutral_count,
        }

    def get_aggregate_sentiment(self, texts):
        """
        Get aggregate sentiment from multiple texts.

        Args:
            texts: List of text strings

        Returns:
            dict: Aggregate sentiment metrics
        """
        if not texts:
            return {"sentiment": "neutral", "score": 0, "confidence": 0}

        sentiments = [self.analyze_text(text) for text in texts]

        avg_score = sum(s["score"] for s in sentiments) / len(sentiments)
        avg_confidence = sum(s["confidence"] for s in sentiments) / len(sentiments)

        sentiment_counts = Counter(s["sentiment"] for s in sentiments)
        dominant_sentiment = sentiment_counts.most_common(1)[0][0]

        return {
            "sentiment": dominant_sentiment,
            "score": avg_score,
            "confidence": avg_confidence,
            "distribution": dict(sentiment_counts),
            "sample_size": len(texts),
        }


class NewsChecker:
    """Fetches and analyzes crypto news."""

    def __init__(self, cache_manager=None, sentiment_analyzer=None):
        """
        Initialize news checker.

        Args:
            cache_manager: Optional cache manager
            sentiment_analyzer: SentimentAnalyzer instance
        """
        self.cache = cache_manager
        self.sentiment = sentiment_analyzer or SentimentAnalyzer()

        self.news_sources = {
            "cryptocompare": "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
        }

        logger.info("Initialized news checker")

    def fetch_recent_news(self, symbol="BTC", limit=10):
        """
        Fetch recent news articles.

        Args:
            symbol: Cryptocurrency symbol
            limit: Number of articles to fetch

        Returns:
            list: News articles with metadata
        """
        cache_key = f"news_{symbol}_{limit}"

        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                return cached

        articles = []

        try:
            url = self.news_sources["cryptocompare"]
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()

                if "Data" in data:
                    for article in data["Data"][:limit]:
                        title = article.get("title", "")
                        body = article.get("body", "")

                        if (
                            symbol.upper() in title.upper()
                            or symbol.upper() in body.upper()
                        ):
                            articles.append(
                                {
                                    "title": title,
                                    "body": body[:500],
                                    "source": article.get("source", "Unknown"),
                                    "published": datetime.fromtimestamp(
                                        article.get("published_on", 0)
                                    ),
                                    "url": article.get("url", ""),
                                    "categories": article.get("categories", "").split(
                                        "|"
                                    ),
                                }
                            )

                    logger.info(f"Fetched {len(articles)} news articles for {symbol}")

        except Exception as e:
            logger.warning(f"Error fetching news: {e}")
            articles = []

        if self.cache and articles:
            self.cache.set(cache_key, articles)

        return articles

    def analyze_news_sentiment(self, symbol="BTC", hours=24):
        """
        Analyze sentiment from recent news.

        Args:
            symbol: Cryptocurrency symbol
            hours: Look back this many hours

        Returns:
            dict: Sentiment analysis results
        """
        articles = self.fetch_recent_news(symbol, limit=20)

        if not articles:
            return {
                "sentiment": "neutral",
                "score": 0,
                "confidence": 0,
                "articles_analyzed": 0,
                "recent_headlines": [],
            }

        cutoff = datetime.now() - timedelta(hours=hours)
        recent_articles = [a for a in articles if a["published"] > cutoff]

        texts = [a["title"] + " " + a["body"] for a in recent_articles]
        sentiment = self.sentiment.get_aggregate_sentiment(texts)

        recent_headlines = [
            {
                "title": a["title"],
                "source": a["source"],
                "published": a["published"].isoformat(),
                "sentiment": self.sentiment.analyze_text(a["title"])["sentiment"],
            }
            for a in recent_articles[:5]
        ]

        return {
            **sentiment,
            "articles_analyzed": len(recent_articles),
            "recent_headlines": recent_headlines,
            "lookback_hours": hours,
        }

    def check_breaking_news(self, symbol="BTC", minutes=60):
        """
        Check for breaking news in last N minutes.

        Args:
            symbol: Cryptocurrency symbol
            minutes: Look back this many minutes

        Returns:
            dict: Breaking news status
        """
        articles = self.fetch_recent_news(symbol, limit=30)

        cutoff = datetime.now() - timedelta(minutes=minutes)
        breaking = [a for a in articles if a["published"] > cutoff]

        if breaking:
            logger.info(
                f"Found {len(breaking)} breaking news in last {minutes} minutes"
            )

            texts = [a["title"] + " " + a["body"] for a in breaking]
            sentiment = self.sentiment.get_aggregate_sentiment(texts)

            return {
                "has_breaking_news": True,
                "count": len(breaking),
                "sentiment": sentiment["sentiment"],
                "score": sentiment["score"],
                "headlines": [a["title"] for a in breaking[:3]],
                "sources": list(set(a["source"] for a in breaking)),
            }

        return {
            "has_breaking_news": False,
            "count": 0,
            "sentiment": "neutral",
            "score": 0,
        }


class MarketSentimentFeatures:
    """Generate features from market sentiment and news."""

    def __init__(self, news_checker):
        """
        Initialize market sentiment features.

        Args:
            news_checker: NewsChecker instance
        """
        self.news_checker = news_checker

    def create_sentiment_features(self, symbol="BTC"):
        """
        Create sentiment-based features for model.

        Args:
            symbol: Cryptocurrency symbol

        Returns:
            dict: Sentiment features
        """
        features = {}

        news_24h = self.news_checker.analyze_news_sentiment(symbol, hours=24)
        features["news_sentiment_24h"] = news_24h["score"]
        features["news_confidence_24h"] = news_24h["confidence"]
        features["news_positive_24h"] = (
            1.0 if news_24h["sentiment"] == "positive" else 0.0
        )
        features["news_negative_24h"] = (
            1.0 if news_24h["sentiment"] == "negative" else 0.0
        )

        news_6h = self.news_checker.analyze_news_sentiment(symbol, hours=6)
        features["news_sentiment_6h"] = news_6h["score"]
        features["news_confidence_6h"] = news_6h["confidence"]

        breaking = self.news_checker.check_breaking_news(symbol, minutes=60)
        features["has_breaking_news"] = 1.0 if breaking["has_breaking_news"] else 0.0
        features["breaking_news_sentiment"] = breaking["score"]

        features["news_volume_24h"] = news_24h.get("articles_analyzed", 0) / 20.0

        return features
