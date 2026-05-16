from __future__ import annotations

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from plot_common import MODEL_COLORS, MODEL_ORDER, apply_clean_axes, build_model_seed_summary, build_matrix_mean_std_lookup, save_figure


def main() -> None:
    seed_rows = build_model_seed_summary()
    lookup = build_matrix_mean_std_lookup()

    fig = plt.figure(figsize=(14.2, 5.6))
    fig.suptitle("Model Stability Analysis: Accuracy and Latency Variance", fontsize=13, fontweight="bold", y=0.98)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.1], wspace=0.22)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    x = np.arange(len(MODEL_ORDER))
    seed_offsets = {42: -0.11, 43: 0.0, 44: 0.11}

    for idx, model in enumerate(MODEL_ORDER):
        items = [row for row in seed_rows if row["model"] == model]
        mean_item = lookup[("STFT", model, 5)]
        ax_left.errorbar(
            idx,
            mean_item["accuracy_mean_percent"],
            yerr=mean_item["accuracy_std_percent"],
            fmt="none",
            ecolor=MODEL_COLORS[model],
            elinewidth=2.2,
            capsize=4,
            zorder=2,
        )
        for item in items:
            marker = {42: "o", 43: "s", 44: "^"}[item["seed"]]
            ax_left.scatter(
                idx + seed_offsets[item["seed"]],
                item["accuracy_percent"],
                s=74,
                color=MODEL_COLORS[model],
                marker=marker,
                edgecolors="white",
                linewidth=0.8,
                zorder=3,
            )
        ax_left.scatter(idx, mean_item["accuracy_mean_percent"], s=165, color=MODEL_COLORS[model], edgecolors="black", linewidth=0.6, zorder=4)
        ax_left.text(
            idx,
            mean_item["accuracy_mean_percent"] + mean_item["accuracy_std_percent"] + 2.5,
            f"{mean_item['accuracy_mean_percent']:.2f}% ±{mean_item['accuracy_std_percent']:.2f}",
            ha="center",
            va="bottom",
            fontsize=9.4,
            color=MODEL_COLORS[model],
            fontweight="bold",
        )

    ax_left.set_xticks(x)
    ax_left.set_xticklabels(MODEL_ORDER)
    ax_left.set_ylabel("准确率 (%)")
    ax_left.set_ylim(65, 108)
    apply_clean_axes(ax_left, grid_axis="y")

    seed_handles = [
        mlines.Line2D([], [], color="#666666", marker=marker, linestyle="None", markersize=6, label=f"seed {seed}")
        for seed, marker in zip([42, 43, 44], ["o", "s", "^"])
    ]
    ax_left.legend(handles=seed_handles, loc="lower right", fontsize=8.8, ncol=3)

    accuracy_std = [lookup[("STFT", model, 5)]["accuracy_std_percent"] for model in MODEL_ORDER]
    latency_std = [lookup[("STFT", model, 5)]["latency_std_ms"] for model in MODEL_ORDER]
    x2 = np.arange(len(MODEL_ORDER))
    width = 0.36
    ax_right.bar(x2, accuracy_std, width=width, color="#EE6677", alpha=0.90, label="准确率标准差")
    for idx, value in enumerate(accuracy_std):
        ax_right.text(idx, value + max(accuracy_std) * 0.08, f"{value:.2f}", ha="center", fontsize=9.0)
    ax_r2 = ax_right.twinx()
    ax_r2.plot(x2, latency_std, color="#4477AA", marker="o", linewidth=2.0, markersize=6, label="时延标准差")
    for idx, value in enumerate(latency_std):
        y_offset = max(latency_std) * 0.15 if value == max(latency_std) else max(latency_std) * 0.30
        ax_r2.text(idx, value + y_offset, f"{value:.3f}", ha="center", fontsize=8.6, color="#4477AA")

    ax_right.set_xticks(x2)
    ax_right.set_xticklabels(MODEL_ORDER)
    ax_right.set_ylabel("准确率标准差 (%)")
    ax_r2.set_ylabel("时延标准差 (ms)")
    ax_right.set_ylim(0, max(accuracy_std) * 1.25)
    ax_r2.set_ylim(0, max(latency_std) * 1.45)
    apply_clean_axes(ax_right, grid_axis="y")
    ax_r2.spines["top"].set_visible(False)

    legend_handles = [
        mlines.Line2D([], [], color="#EE6677", linewidth=8, label="准确率标准差"),
        mlines.Line2D([], [], color="#4477AA", marker="o", linewidth=2.0, label="时延标准差"),
    ]
    ax_right.legend(handles=legend_handles, loc="upper right", fontsize=8.8)
    save_figure(fig, "fig5_3_model_stability_detail")


if __name__ == "__main__":
    main()
