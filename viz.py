import marimo

__generated_with = "0.13.9"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import json
    import matplotlib.pyplot as plt
    from datetime import datetime
    from typing import Optional
    import os
    from pathlib import Path
    return Optional, Path, datetime, json, mo, pd, plt


@app.cell
def _():
    experiment = 'GRPO_Qwen2.5-Coder-1.5B-Instruct'
    results_dir = 'qwen3-rust-finetune/outputs'
    return experiment, results_dir


@app.cell
def _(experiment, plot_file):
    plot_file(experiment, 'cargo_test_rewards.jsonl', f'{experiment}\n\nCargo Test Reward (max score=2.0)')
    return


@app.cell
def _(experiment, plot_file):
    plot_file(experiment, 'cargo_build_rewards.jsonl', f'{experiment}\n\nCargo Build Reward (max score=1.0)')
    return


@app.cell
def _(experiment, plot_file):
    plot_file(experiment, 'cargo_clippy_rewards.jsonl', f'{experiment}\n\nCargo Clippy Reward (max score=1.0)')
    return


@app.cell
def _(experiment, plot_file):
    plot_file(experiment, 'non_empty_rewards.jsonl', f'{experiment}\n\nNon-Empty Reward (max score=1.0)')
    return


@app.cell
def _(experiment, plot_file):
    plot_file(experiment, 'test_block_count_rewards.jsonl', f'{experiment}\n\nTest Block Count Reward (max score=1.0)')
    return


@app.cell
def _(Path, mo, plot_rolling_average, results_dir):
    def plot_file(experiment: str, filename: str, title: str):
        # Ensure the results directory exists
        os.makedirs(results_dir, exist_ok=True)

        output_file = Path(results_dir) / experiment / filename
        print(f"Reading file: {output_file}")

        image_name = filename.replace('.jsonl', '') + "_rolling_average.png"
        output_image_file = Path(results_dir) / image_name
        plot_rolling_average(output_file, title=title, window_size=100, save_path=output_image_file)
        return mo.image(src=output_image_file, rounded=True)
    return (plot_file,)


@app.cell
def _(Optional, json, pd, plt):
    def plot_rolling_average(filepath: str, title='Rolling Average Score Over Time', window_size: int = 100, save_path: Optional[str] = None) -> None:
        """
        Reads a JSONL file containing timestamp and score columns, calculates and plots
        the rolling average of scores over specified window size.

        Args:
            filepath (str): Path to the JSONL file
            window_size (int): Size of the rolling window for averaging (default: 100)
            save_path (Optional[str]): If provided, saves the plot to this path instead of displaying

        Returns:
            None
        """
        # Read JSONL file line by line into a list
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                data.append(json.loads(line))

        # Convert to DataFrame
        # df = pd.DataFrame(data)[window_size:]
        df = pd.DataFrame(data)

        # Calculate rolling average
        rolling_avg = df['score'].rolling(window=window_size, min_periods=1).mean()

        # Create the plot using row indices
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(rolling_avg)), rolling_avg, 'b-',
                label=f'Rolling Average (window={window_size})')

        # Customize the plot
        plt.title(title)
        plt.xlabel('Step')
        plt.ylabel('Average Score')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        # Either save or display the plot
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    return (plot_rolling_average,)


@app.cell
def _(experiment, plot_metrics_from_jsonl):
    plot_metrics_from_jsonl(f'qwen3-rust-finetune/outputs/{experiment}/logs.jsonl', metrics=["loss", "reward", "learning_rate"])
    return


@app.cell
def _(datetime, json, plt):
    def plot_metrics_from_jsonl(
        jsonl_path,
        metrics=None,
        time_key="timestamp",
        time_format="%Y-%m-%d %H:%M:%S",
    ):
        """
        Reads a JSONL file and plots specified metrics over time.

        :param jsonl_path: Path to the .jsonl file.
        :param metrics: List of metric keys to plot. If None, plots all numeric fields except the time field.
        :param time_key: Key in JSON entries representing the timestamp.
        :param time_format: Format string for parsing timestamps.
        """
        # Load entries
        entries = []
        with open(jsonl_path, "r") as f:
            for line in f:
                entries.append(json.loads(line))

        # Parse times
        times = [datetime.strptime(e[time_key], time_format) for e in entries]

        # Determine which metrics to plot
        sample = entries[0]
        if metrics is None:
            metrics = [
                k
                for k, v in sample.items()
                if k != time_key and isinstance(v, (int, float))
            ]

          # Plot all metrics side-by-side
        n = len(metrics)
        fig, axes = plt.subplots(1, n, figsize=(4*n, 4), sharex=True)
        for ax, metric in zip(axes, metrics):
            vals = [e.get(metric) for e in entries]
            ax.plot(times, vals)
            ax.set_title(metric)
            ax.set_xlabel('Time')
            ax.set_ylabel(metric)
        fig.tight_layout()
        plt.show()
    
        print(sample.keys())
    return (plot_metrics_from_jsonl,)


if __name__ == "__main__":
    app.run()
