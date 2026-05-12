from __future__ import annotations

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from plot_common import SHOT_ORDER, apply_clean_axes, build_matrix_mean_std_lookup, build_shot_seed_summary, save_figure


def main() -> None:
    seed_rows = build_shot_seed_summary()
    lookup = build_matrix_mean_std_lookup()

    fig = plt.figure(figsize=(14.2, 5.6))
    fig.suptitle("Shot Stability Analysis: Performance across Different Shot Settings", fontsize=13, fontweight="bold", y=0.98)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.65, 1.0], wspace=0.22)
    ax_left = fig.add_subplot(gs[0, 0])
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
        ax_left.plot(x, [p["accuracy_percent"] for p in points], linewidth=1.35, alpha=0.78, color=seed_styles[seed]["color"])
        ax_left.scatter(
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
    ax_left.plot(x, mean_values, color="#1F6F8B", linewidth=2.8, marker="o", markersize=8, label="多 seed 均值", zorder=2)
    ax_left.fill_between(x, lower, upper, color="#A0CBE8", alpha=0.28, zorder=1)
    ax_left.annotate(
        "10-shot 最优且最稳",
        xy=(1, mean_values[1]),
        xytext=(0.55, 98.65),
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#333333"},
        fontsize=10,
    )
    ax_left.set_xticks(x)
    ax_left.set_xticklabels([f"{shot}-shot" for shot in SHOT_ORDER])
    ax_left.set_ylabel("准确率 (%)")
    ax_left.set_ylim(97.0, 100.15)
    ax_left.set_xlim(-0.4, len(SHOT_ORDER) - 0.6)
    apply_clean_axes(ax_left, grid_axis="y")

    handles = [
        mlines.Line2D([], [], color=seed_styles[seed]["color"], marker=seed_styles[seed]["marker"], linestyle="-", label=f"seed {seed}")
        for seed in [42, 43, 44]
    ]
    handles.append(mlines.Line2D([], [], color="#1F6F8B", marker="o", linewidth=2.8, label="多 seed 均值"))
    ax_left.legend(handles=handles, loc="lower left", fontsize=8.8, ncol=2)

    std_bars = [lookup[("STFT", "MAML", shot)]["accuracy_std_percent"] for shot in SHOT_ORDER]
    lat_vals = [lookup[("STFT", "MAML", shot)]["latency_mean_ms"] for shot in SHOT_ORDER]
    ax_right.bar(x, std_bars, width=0.44, color="#EE6677", alpha=0.88, label="准确率标准差")
    for idx, value in enumerate(std_bars):
        ax_right.text(idx, value + max(std_bars) * 0.08, f"{value:.2f}", ha="center", fontsize=8.9)
    ax_r2 = ax_right.twinx()
    ax_r2.plot(x, lat_vals, color="#4477AA", marker="o", linewidth=2.2, markersize=6, label="平均时延")
    for idx, value in enumerate(lat_vals):
        ax_r2.text(idx, value + (max(lat_vals) - min(lat_vals)) * 0.15, f"{value:.2f}", ha="center", fontsize=8.6, color="#4477AA")

    ax_right.set_xticks(x)
    ax_right.set_xticklabels([f"{shot}-shot" for shot in SHOT_ORDER])
    ax_right.set_ylabel("准确率标准差 (%)")
    ax_r2.set_ylabel("平均时延 (ms)")
    ax_right.set_ylim(0, max(std_bars) * 1.35)
    ax_r2.set_ylim(min(lat_vals) * 0.98, max(lat_vals) * 1.15)
    apply_clean_axes(ax_right, grid_axis="y")
    ax_r2.spines["top"].set_visible(False)
    legend_handles = [
        mlines.Line2D([], [], color="#EE6677", linewidth=8, label="准确率标准差"),
        mlines.Line2D([], [], color="#4477AA", marker="o", linewidth=2.2, label="平均时延"),
    ]
    ax_right.legend(handles=legend_handles, loc="upper right", fontsize=8.8)
    save_figure(fig, "fig5_4_shot_stability_detail")


if __name__ == "__main__":
    main()
