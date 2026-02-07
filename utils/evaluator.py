import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def load_backtest_results(results_file="./data/storage/backtest_results.json"):
    with open(results_file, "r") as f:
        return json.load(f)


def analyze_predictions(results):
    """Analyze prediction patterns and accuracy."""
    predictions = results["predictions"]

    df = pd.DataFrame(predictions)

    confidence_bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    df["confidence_bin"] = pd.cut(df["confidence"], bins=confidence_bins)

    accuracy_by_confidence = df.groupby("confidence_bin").apply(
        lambda x: (x["direction"] == x["actual"]).mean()
    )

    print("Accuracy by Confidence Level:")
    print(accuracy_by_confidence)
    print()

    accuracy_by_direction = df.groupby("direction").apply(
        lambda x: (x["direction"] == x["actual"]).mean()
    )

    print("Accuracy by Predicted Direction:")
    print(accuracy_by_direction)
    print()

    correct = (df["direction"] == df["actual"]).astype(int)

    streaks = []
    current_streak = 0
    for c in correct:
        if c == 1:
            current_streak += 1
        else:
            if current_streak > 0:
                streaks.append(current_streak)
            current_streak = 0

    if current_streak > 0:
        streaks.append(current_streak)

    print(f"Longest winning streak: {max(streaks) if streaks else 0}")
    print(f"Average winning streak: {np.mean(streaks) if streaks else 0:.2f}")
    print()

    return df


def calculate_risk_metrics(results):
    """Calculate risk-adjusted performance metrics."""
    predictions = results["predictions"]

    traded = [p for p in predictions if p.get("return_pct") is not None]

    if not traded:
        print("No trades to analyze")
        return

    returns = [p["return_pct"] for p in traded]

    total_return = sum(returns)
    avg_return = np.mean(returns)
    std_return = np.std(returns)

    sharpe = (avg_return / std_return * np.sqrt(8760)) if std_return > 0 else 0

    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    max_drawdown = np.min(drawdown)

    wins = sum(1 for r in returns if r > 0)
    win_rate = wins / len(returns)

    gross_profit = sum(r for r in returns if r > 0)
    gross_loss = abs(sum(r for r in returns if r < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    print("Risk Metrics:")
    print(f"  Total Return: {total_return:.2f}%")
    print(f"  Average Return: {avg_return:.3f}%")
    print(f"  Std Deviation: {std_return:.3f}%")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Max Drawdown: {max_drawdown:.2f}%")
    print(f"  Win Rate: {win_rate:.2%}")
    print(f"  Profit Factor: {profit_factor:.2f}")
    print()


def plot_cumulative_returns(results, save_path="./cumulative_returns.png"):
    """Plot cumulative returns over time."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping plot.")
        return

    predictions = results["predictions"]
    traded = [p for p in predictions if p.get("return_pct") is not None]

    if not traded:
        print("No trades to plot")
        return

    returns = [p["return_pct"] for p in traded]
    cumulative = np.cumsum(returns)

    plt.figure(figsize=(12, 6))
    plt.plot(cumulative)
    plt.title("Cumulative Returns Over Time")
    plt.xlabel("Trade Number")
    plt.ylabel("Cumulative Return (%)")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color="r", linestyle="--", alpha=0.5)

    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")


def export_detailed_stats(results, output_file="./detailed_stats.csv"):
    """Export detailed statistics to CSV."""
    predictions = results["predictions"]
    df = pd.DataFrame(predictions)

    df["correct"] = (df["direction"] == df["actual"]).astype(int)

    df.to_csv(output_file, index=False)
    print(f"Detailed stats exported to {output_file}")


if __name__ == "__main__":
    import sys

    results_file = (
        sys.argv[1] if len(sys.argv) > 1 else "./data/storage/backtest_results.json"
    )

    if not Path(results_file).exists():
        print(f"Results file not found: {results_file}")
        print("Run backtest first: python main.py --mode backtest")
        sys.exit(1)

    print("Loading backtest results...")
    results = load_backtest_results(results_file)

    print()
    print("=" * 60)
    print("DETAILED ANALYSIS")
    print("=" * 60)
    print()

    df = analyze_predictions(results)
    calculate_risk_metrics(results)

    try:
        plot_cumulative_returns(results)
    except Exception as e:
        print(f"Could not create plot: {e}")

    export_detailed_stats(results)

    print()
    print("Analysis complete!")
