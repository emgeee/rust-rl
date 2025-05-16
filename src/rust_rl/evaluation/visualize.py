"""
Visualization utilities for Rust evaluation results
"""

import matplotlib.pyplot as plt


def plot_results(results, as_html_func=None):
    """
    Plot the results of Rust code evaluation.
    
    Args:
        results: DataFrame with evaluation results
        as_html_func: Optional function to convert matplotlib figure to HTML
        
    Returns:
        If as_html_func is provided, returns a layout object with the plots
        Otherwise, returns a dictionary of matplotlib figures
    """
    def _plot(df, column_name, title):
        build_passed_counts = results[column_name].value_counts()
        fig = plt.figure(figsize=(4, 3))
        num_correct = build_passed_counts[True] if True in build_passed_counts else 0.0
        num_incorrect = build_passed_counts[False] if False in build_passed_counts else 0.0
        total = num_correct + num_incorrect
        percentage = (num_correct / total) * 100
        plt.title(f"{title}: {num_correct}/{total} = {percentage:.2f}%")

        # Create ordered index and corresponding values
        ordered_index = ["True", "False"]
        ordered_values = [
            build_passed_counts.get(True, 0),
            build_passed_counts.get(False, 0),
        ]

        # Create color map
        # Retro color palette
        # https://www.color-hex.com/color-palette/165
        colors = ["#6fcb9f", "#fb2e01"]

        # Plot with fixed order and colors
        plt.bar(ordered_index, ordered_values, color=colors)
        return fig, plt.gca()
    
    build_fig, build_ax = _plot(results, "build_passed", "Build Passed")
    clippy_fig, clippy_ax = _plot(results, "clippy_passed", "Clippy Passed")
    test_fig, test_ax = _plot(results, "test_passed", "Test Passed")
    
    if as_html_func:
        import marimo as mo
        return mo.vstack(
            [
                mo.md("# Results"),
                mo.hstack(
                    [
                        as_html_func(build_ax),
                        as_html_func(clippy_ax),
                        as_html_func(test_ax),
                    ],
                ),
            ]
        )
    else:
        return {
            "build": build_fig,
            "clippy": clippy_fig,
            "test": test_fig
        }