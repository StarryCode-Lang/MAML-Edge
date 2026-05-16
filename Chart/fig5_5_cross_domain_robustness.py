from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from plot_common import apply_clean_axes, format_split_label, read_csv_rows, save_figure


SPLIT_KEYS = [("0,1,2", "3"), ("0,1,3", "2"), ("0,2,3", "1"), ("1,2,3", "0")]


def main() -> None:
    rows = read_csv_rows("logs/thesis_tables/paper_balanced/table3_domain_robustness.csv")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["train_domains"], row["test_domain"])].append(float(row["accuracy_percent"]))

    labels = [format_split_label(train_domains, test_domain) for train_domains, test_domain in SPLIT_KEYS]
    distributions = [grouped[key] for key in SPLIT_KEYS]
    means = [sum(values) / len(values) for values in distributions]

    fig = plt.figure(figsize=(14.5, 5.9))
    fig.suptitle("Cross-Domain Robustness: Generalization across Different Domain Splits", fontsize=13, fontweight="bold", y=0.99)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.16)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    colors = ["#4477AA", "#76B7B2", "#66BB55", "#EE6677"]
    bp = ax_left.boxplot(
        distributions,
        patch_artist=True,
        widths=0.55,
        medianprops={"color": "#333333", "linewidth": 1.5},
        boxprops={"linewidth": 1.2},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)

    rng = np.random.default_rng(42)
    for idx, values in enumerate(distributions, start=1):
        jitter = rng.normal(0, 0.045, len(values))
        ax_left.scatter(np.full(len(values), idx) + jitter, values, s=28, color="#333333", alpha=0.45, zorder=3)
        ax_left.scatter(idx, means[idx - 1], s=90, color="#111111", marker="D", zorder=4)

    ax_left.set_xticks(np.arange(1, 5))
    ax_left.set_xticklabels(labels)
    ax_left.set_ylabel("准确率 (%)")
    ax_left.set_ylim(84.5, 100.8)
    apply_clean_axes(ax_left, grid_axis="y")

    for idx, mean in enumerate(means):
        ax_left.text(idx + 1, 100.3, f"{mean:.2f}", ha="center", fontsize=9.0, color="#333333", fontweight="bold")

    ranking = sorted(zip(labels, means), key=lambda item: item[1], reverse=True)
    rank_labels = [item[0] for item in ranking]
    rank_values = [item[1] for item in ranking]
    bar_colors = ["#4477AA", "#76B7B2", "#66BB55", "#EE6677"]
    bars = ax_right.barh(rank_labels, rank_values, color=bar_colors, height=0.64)
    ax_right.invert_yaxis()
    for bar, value in zip(bars, rank_values):
        ax_right.text(value + 0.05, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=9.0)
    hardest_index = rank_labels.index("123→0")
    ax_right.annotate(
        "最难目标域",
        xy=(min(rank_values), hardest_index),
        xytext=(min(rank_values) - 0.5, hardest_index),
        arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#333333"},
        fontsize=9.6,
        ha="right",
        va="center",
    )
    ax_right.set_xlim(min(rank_values) - 1.5, max(rank_values) + 1.0)
    ax_right.set_xlabel("平均准确率 (%)")
    apply_clean_axes(ax_right, grid_axis="x")
    save_figure(fig, "fig5_5_cross_domain_robustness")


if __name__ == "__main__":
    main()
