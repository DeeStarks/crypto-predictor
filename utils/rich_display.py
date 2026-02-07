import logging
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn
import time

logger = logging.getLogger(__name__)


class RichDisplay:
    """Rich terminal display with colors and formatting."""

    def __init__(self):
        """Initialize rich display."""
        self.console = Console()
        self.prediction_history = []

    def create_header(self, symbol, window_info):
        """Create header panel."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        header_text = Text()
        header_text.append("Crypto Price Predictor\n", style="bold cyan")
        header_text.append(f"Symbol: {symbol} | ", style="bold white")
        header_text.append(
            f"Window: {window_info.get('window_id', 'N/A')}\n", style="bold white"
        )
        header_text.append(f"Time: {timestamp}", style="dim white")

        return Panel(header_text, box=box.DOUBLE, border_style="cyan")

    def create_price_panel(self, price_to_beat, current_price, prediction=None):
        """Create price comparison panel."""
        table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="white", width=20)
        table.add_column("Status", style="white", width=30)

        table.add_row("Price to Beat", f"${price_to_beat:,.2f}", "Window Opening Price")

        if current_price:
            change = current_price - price_to_beat
            change_pct = (change / price_to_beat) * 100

            if change > 0:
                status_text = Text(
                    f"↗ +${abs(change):,.2f} (+{change_pct:.2f}%)", style="bold green"
                )
            elif change < 0:
                status_text = Text(
                    f"↘ -${abs(change):,.2f} ({change_pct:.2f}%)", style="bold red"
                )
            else:
                status_text = Text("➡ No change", style="yellow")

            table.add_row("Current Price", f"${current_price:,.2f}", status_text)

            if prediction:
                if prediction["direction"] == "up":
                    table.add_row("Target", f"> ${price_to_beat:,.2f}", "Predicting UP")
                else:
                    table.add_row(
                        "Target", f"< ${price_to_beat:,.2f}", "Predicting DOWN"
                    )

        return Panel(table, title="[bold]Price Tracker[/bold]", border_style="magenta")

    def create_prediction_panel(self, prediction, prob_beating, confidence_interval):
        """Create prediction panel with probabilities."""
        if not prediction:
            return Panel(
                Text("Waiting for prediction...", style="dim yellow"),
                title="[bold]Current Prediction[/bold]",
                border_style="yellow",
            )

        table = Table(box=box.SIMPLE, show_header=False)
        table.add_column("", style="cyan", width=25)
        table.add_column("", style="white", width=35)

        direction_color = "green" if prediction["direction"] == "up" else "red"
        table.add_row(
            "Direction",
            Text(
                prediction["direction"].upper(),
                style=f"bold {direction_color}",
            ),
        )

        conf = prediction["confidence"]
        conf_color = "green" if conf > 0.7 else "yellow" if conf > 0.6 else "red"
        table.add_row("Confidence", Text(f"{conf:.1%}", style=f"bold {conf_color}"))

        prob_up = prediction["probability_up"]
        table.add_row("Probability UP", f"{prob_up:.1%}")

        if prob_beating:
            beat_color = (
                "green"
                if prob_beating > 0.6
                else "yellow" if prob_beating > 0.5 else "red"
            )
            table.add_row(
                "Prob. Beating Price",
                Text(f"{prob_beating:.1%}", style=f"bold {beat_color}"),
            )

        if confidence_interval:
            table.add_row(
                "95% Confidence",
                f"{confidence_interval[0]:.1%} - {confidence_interval[1]:.1%}",
            )

        action = prediction.get("action", "unknown")
        if action == "trade":
            action_text = Text("TRADE", style="bold green")
        elif action == "skip":
            action_text = Text("SKIP (Low Confidence)", style="bold yellow")
        else:
            action_text = Text("UNKNOWN", style="dim")

        table.add_row("Action", action_text)

        border_color = "green" if action == "trade" else "yellow"
        return Panel(
            table, title="[bold]Current Prediction[/bold]", border_style=border_color
        )

    def create_history_table(self, history):
        """Create table of recent predictions."""
        table = Table(
            title="Recent Predictions (Last 5 Windows)",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )

        table.add_column("Time", style="dim", width=16)
        table.add_column("Direction", width=10)
        table.add_column("Confidence", width=12)
        table.add_column("Outcome", width=15)
        table.add_column("Actual", width=15)

        for pred in history[-5:]:
            timestamp = pred.get("timestamp", "N/A")
            if isinstance(timestamp, str):
                timestamp = timestamp[11:16]

            direction = pred.get("direction", "N/A")
            direction_text = Text(
                f"{direction.upper()}",
                style="green" if direction == "up" else "red",
            )

            confidence = pred.get("confidence", 0)
            conf_text = Text(
                f"{confidence:.1%}",
                style=(
                    "green"
                    if confidence > 0.7
                    else "yellow" if confidence > 0.6 else "red"
                ),
            )

            actual = pred.get("actual", None)
            if actual:
                correct = actual == direction
                outcome_text = Text(
                    "Correct" if correct else "Wrong",
                    style="green" if correct else "red",
                )
                actual_text = Text(
                    actual.upper(),
                    style="green" if actual == "up" else "red",
                )
            else:
                outcome_text = Text("Pending", style="dim yellow")
                actual_text = Text("N/A", style="dim")

            table.add_row(
                timestamp, direction_text, conf_text, outcome_text, actual_text
            )

        return table

    def create_stats_panel(self, stats):
        """Create statistics panel."""
        table = Table(box=box.SIMPLE, show_header=False)
        table.add_column("", style="cyan", width=25)
        table.add_column("", style="white", width=20)

        table.add_row("Windows Processed", str(stats.get("windows_processed", 0)))
        table.add_row("Predictions Made", str(stats.get("predictions_made", 0)))

        if "accuracy" in stats:
            acc = stats["accuracy"]
            acc_color = "green" if acc > 0.6 else "yellow" if acc > 0.5 else "red"
            table.add_row("Accuracy", Text(f"{acc:.1%}", style=f"bold {acc_color}"))

        if "win_rate" in stats:
            win_rate = stats["win_rate"]
            win_color = (
                "green" if win_rate > 0.6 else "yellow" if win_rate > 0.5 else "red"
            )
            table.add_row(
                "Win Rate", Text(f"{win_rate:.1%}", style=f"bold {win_color}")
            )

        if "avg_confidence" in stats:
            table.add_row("Avg Confidence", f"{stats['avg_confidence']:.1%}")

        return Panel(
            table, title="[bold]Session Statistics[/bold]", border_style="blue"
        )

    def create_insights_panel(self, insights):
        """Create market insights panel."""
        if not insights:
            return Panel(
                Text("No insights available", style="dim"),
                title="[bold]Market Insights[/bold]",
                border_style="blue",
            )

        text = Text()

        if "trend" in insights:
            trend = insights["trend"]
            if trend == "bullish":
                text.append("Trend: BULLISH\n", style="bold green")
            elif trend == "bearish":
                text.append("Trend: BEARISH\n", style="bold red")
            else:
                text.append("Trend: NEUTRAL\n", style="bold yellow")

        if "volatility" in insights:
            vol = insights["volatility"]
            vol_level = "HIGH" if vol > 0.05 else "MEDIUM" if vol > 0.02 else "LOW"
            vol_color = "red" if vol > 0.05 else "yellow" if vol > 0.02 else "green"
            text.append(f"Volatility: {vol_level} ({vol:.2%})\n", style=vol_color)

        if "momentum" in insights:
            mom = insights["momentum"]
            if mom > 0:
                text.append(f"Momentum: POSITIVE (+{mom:.2%})\n", style="green")
            else:
                text.append(f"Momentum: NEGATIVE ({mom:.2%})\n", style="red")

        if "sentiment" in insights:
            sent = insights["sentiment"]
            if sent == "positive":
                text.append("Sentiment: POSITIVE\n", style="green")
            elif sent == "negative":
                text.append("Sentiment: NEGATIVE\n", style="red")
            else:
                text.append("Sentiment: NEUTRAL\n", style="yellow")

        if "support" in insights and "resistance" in insights:
            text.append(f"\nSupport: ${insights['support']:,.2f}\n", style="cyan")
            text.append(f"Resistance: ${insights['resistance']:,.2f}", style="cyan")

        return Panel(text, title="[bold]Market Insights[/bold]", border_style="blue")

    def create_sentiment_panel(self, sentiment_data):
        """Create news sentiment panel."""
        if not sentiment_data:
            return Panel(
                Text("No news data available", style="dim"),
                title="[bold]News Sentiment[/bold]",
                border_style="magenta",
            )

        table = Table(box=box.SIMPLE, show_header=False)
        table.add_column("", style="cyan", width=20)
        table.add_column("", style="white", width=35)

        sentiment = sentiment_data.get("sentiment", "neutral")
        score = sentiment_data.get("score", 0)

        if sentiment == "positive":
            sent_text = Text(f"POSITIVE ({score:+.2f})", style="bold green")
        elif sentiment == "negative":
            sent_text = Text(f"NEGATIVE ({score:+.2f})", style="bold red")
        else:
            sent_text = Text(f"NEUTRAL ({score:+.2f})", style="bold yellow")

        table.add_row("Overall", sent_text)

        articles = sentiment_data.get("articles_analyzed", 0)
        table.add_row("Articles (24h)", str(articles))

        if sentiment_data.get("has_breaking_news"):
            table.add_row("Breaking News", Text("YES", style="bold red"))

        headlines = sentiment_data.get("recent_headlines", [])
        if headlines:
            table.add_row("", "")
            for i, headline in enumerate(headlines[:2], 1):
                title = headline.get("title", "")[:40] + "..."
                table.add_row(f"Headline {i}", title)

        return Panel(table, title="[bold]News Sentiment[/bold]", border_style="magenta")

    def display_live_dashboard(self, data):
        """Display complete live dashboard."""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=4),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=10),
        )

        layout["main"].split_row(Layout(name="left"), Layout(name="right"))

        layout["left"].split_column(Layout(name="price"), Layout(name="prediction"))

        layout["right"].split_column(
            Layout(name="stats"), Layout(name="insights"), Layout(name="sentiment")
        )

        layout["header"].update(
            self.create_header(data.get("symbol", "N/A"), data.get("window_info", {}))
        )

        layout["price"].update(
            self.create_price_panel(
                data.get("price_to_beat", 0),
                data.get("current_price", 0),
                data.get("prediction"),
            )
        )

        layout["prediction"].update(
            self.create_prediction_panel(
                data.get("prediction"),
                data.get("prob_beating"),
                data.get("confidence_interval"),
            )
        )

        layout["stats"].update(self.create_stats_panel(data.get("stats", {})))
        layout["insights"].update(self.create_insights_panel(data.get("insights", {})))
        layout["sentiment"].update(
            self.create_sentiment_panel(data.get("sentiment", {}))
        )
        layout["footer"].update(self.create_history_table(data.get("history", [])))

        return layout

    def print_simple_prediction(self, prediction, price_info):
        """Print simple prediction output (fallback)."""
        self.console.print("\n" + "=" * 60)
        self.console.print(
            f"[bold cyan]PREDICTION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/bold cyan]"
        )
        self.console.print("=" * 60)

        if price_info:
            self.console.print(
                f"\n[bold]Price to Beat:[/bold] ${price_info.get('price_to_beat', 0):,.2f}"
            )
            self.console.print(
                f"[bold]Current Price:[/bold] ${price_info.get('current_price', 0):,.2f}"
            )

        if prediction:
            direction = prediction["direction"]
            color = "green" if direction == "up" else "red"
            self.console.print(
                f"\n[bold {color}]Direction: {direction.upper()}[/bold {color}]"
            )
            self.console.print(
                f"[bold]Confidence:[/bold] {prediction['confidence']:.1%}"
            )

            if "prob_beating" in prediction:
                self.console.print(
                    f"[bold]Probability of Beating:[/bold] {prediction['prob_beating']:.1%}"
                )

        self.console.print("=" * 60 + "\n")
