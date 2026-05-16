from __future__ import annotations

from collections import defaultdict

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from plot_common import (
    SHOT_ORDER, apply_clean_axes, build_matrix_mean_std_lookup,
    build_shot_seed_summary, format_split_label, read_csv_rows, save_figure
)


SPLIT_KEYS = [("0,1,2", "3"), ("0,1,3", "2"), ("0,2,3", "1"), ("1,2,3", "0")]


def main() -> None:
    seed_rows = build_shot_seed_summary()
    lookup = build_matrix_mean_std_lookup()

    rows = read_csv_rows("logs/thesis_tables/paper_balanced/table3_domain_robustness.csv")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["train_domains"], row["test_domain"])].append(float(row["accuracy_percent"]))

    labels = [format_split_label(train_domains, test_domain) for train_domains, test_domain in SPLIT_KEYS]
    distributions = [grouped[key] for key in SPLIT_KEYS]
    means = [sum(values) / len(values) for values in distributions]

    fig = plt.figure(figsize=(16, 6.0))
    fig.suptitle(
        "Stability & Robustness Analysis",
        fontsize=14, fontweight="bold", y=0.98
    )

    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.25)

    ax_left = fig.add_subplot(gs[0, 0])
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
    ax_left.set_ylabel("Accuracy (%)", fontsize=11)
    ax_left.set_xlabel("Domain Split", fontsize=11)
    ax_left.set_ylim(84.5, 100.8)
    apply_clean_axes(ax_left, grid_axis="y")

    for idx, mean in enumerate(means):
        ax_left.text(idx + 1, 100.3, f"{mean:.2f}", ha="center", fontsize=9.0, color="#333333", fontweight="bold")

    ranking = sorted(zip(labels, means), key=lambda item: item[1], reverse=True)
    rank_labels = [item[0] for item in ranking]
    rank_values = [item[1] for item in ranking]
    bar_colors = ["#4477AA", "#76B7B2", "#66BB55", "#EE6677"]
    bars = ax_left.barh(rank_labels, rank_values, color=bar_colors, height=0.64)
    ax_left.invert_yaxis()
    for bar, value in zip(bars, rank_values):
        ax_left.text(value + 0.05, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=9.0)
    ax_left.set_xlim(min(rank_values) - 1.5, max(rank_values) + 1.0)
    ax_left.set_xlabel("Mean Accuracy (%)", fontsize=11)
    apply_clean_axes(ax_left, grid_axis="x")

    ax_right = fig.add_subplot(gs[0, 1])

    x = np.arange(len(SHOT_ORDER))
    seed_styles = {
        42: {"marker": "o", "color": "#4477AA"},
        43: {"marker": "s", "color": "#CCBB44"},
        44: {"marker": "^", "color": "#66BB55"},
    }

    for seed in [42, 43, 44]:
        points = [row for row in seed_rows if row["seed"] == seed]
        points.sort(key=lambda item: item["shots"])
        ax_right.plot(x, [p["accuracy_percent"] for p in points], linewidth=1.35, alpha=0.78, color=seed_styles[seed]["color"])
        ax_right.scatter(
            x,
            [p["accuracy_percent"] for p in points],
            s=72,
            marker=seed_styles[seed]["marker"],
            color=seed_styles[seed]["color"],
            edgecolors="white",
            linewidth=0.8,
            zorder=3,
        )

    mean_values = [lookup[("STFT", "MAML", shot)]["accuracy_mean_percent"] for shot in SHOT_ORDER]
    std_values = [lookup[("STFT", "MAML", shot)]["accuracy_std_percent"] for shot in SHOT_ORDER]
    lower = [m - s for m, s in zip(mean_values, std_values)]
    upper = [m + s for m, s in zip(mean_values, std_values)]
    ax_right.plot(x, mean_values, color="#1F6F8B", linewidth=2.8, marker="o", markersize=8, label="Mean", zorder=2)
    ax_right.fill_between(x, lower, upper, color="#A0CBE8", alpha=0.28, zorder=1)
    ax_right.annotate(
        "10-shot Best",
        xy=(1, mean_values[1]),
        xytext=(0.55, 98.65),
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#333333"},
        fontsize=10,
    )
    ax_right.set_xticks(x)
    ax_right.set_xticklabels([f"{shot}-shot" for shot in SHOT_ORDER])
    ax_right.set_ylabel("Accuracy (%)", fontsize=11)
    ax_right.set_xlabel("Shot Settings", fontsize=11)
    ax_right.set_ylim(97.0, 100.15)
    ax_right.set_xlim(-0.4, len(SHOT_ORDER) - 0.6)
    apply_clean_axes(ax_right, grid_axis="y")

    handles = [
        mlines.Line2D([], [], color=seed_styles[seed]["color"], marker=seed_styles[seed]["marker"], linestyle="-", label=f"seed {seed}")
        for seed in [42, 43, 44]
    ]
    handles.append(mlines.Line2D([], [], color="#1F6F8B", marker="o", linewidth=2.8, label="Mean"))
    ax_right.legend(handles=handles, loc="lower left", fontsize=8.5, ncol=2)

    save_figure(fig, "fig5_stability_robustness_combined")

if __name__ == "__main__":
    main()