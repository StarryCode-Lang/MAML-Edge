from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from plot_common import (
    MODEL_COLORS,
    MODEL_ORDER,
    PREPROCESS_COLORS,
    PREPROCESS_ORDER,
    SHOT_ORDER,
    apply_clean_axes,
    build_main_controlled_lookup,
    short_shot_labels,
    save_figure,
)


def main() -> None:
    lookup = build_main_controlled_lookup()
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15.4, 8.5),
        sharex="col",
        gridspec_kw={"height_ratios": [3.0, 1.35], "hspace": 0.12, "wspace": 0.18},
    )

    x = np.arange(len(SHOT_ORDER))
    width = 0.21
    offsets = [-width, 0.0, width]

    for col, preprocess in enumerate(PREPROCESS_ORDER):
        acc_ax = axes[0, col]
        lat_ax = axes[1, col]

        all_acc = [lookup[(preprocess, model, shot)]["accuracy_percent"] for model in MODEL_ORDER for shot in SHOT_ORDER]
        all_lat = [lookup[(preprocess, model, shot)]["latency_ms"] for model in MODEL_ORDER for shot in SHOT_ORDER]

        if preprocess == "FFT":
            acc_bottom, acc_top = 84.0, 100.8
        elif preprocess == "STFT":
            acc_bottom, acc_top = 97.8, 100.15
        else:
            acc_bottom, acc_top = 99.0, 100.05

        for offset, model in zip(offsets, MODEL_ORDER):
            accuracies = [lookup[(preprocess, model, shot)]["accuracy_percent"] for shot in SHOT_ORDER]
            latencies = [lookup[(preprocess, model, shot)]["latency_ms"] for shot in SHOT_ORDER]
            bars = acc_ax.bar(
                x + offset,
                accuracies,
                width=width,
                color=MODEL_COLORS[model],
                edgecolor="white",
                linewidth=0.8,
                alpha=0.95,
                label=model if col == 0 else None,
            )
            lat_ax.plot(
                x,
                latencies,
                color=MODEL_COLORS[model],
                marker="o",
                markersize=5,
                linewidth=2.0,
                alpha=0.95,
                label=model if col == 0 else None,
            )

            label_offset = 0.4 if preprocess == "FFT" else 0.2
            if preprocess == "FFT":
                highest_bar = max(zip(accuracies, bars), key=lambda item: item[0])[1]
                acc_ax.text(
                    highest_bar.get_x() + highest_bar.get_width() / 2,
                    highest_bar.get_height() + label_offset,
                    model,
                    ha="center",
                    va="bottom",
                    fontsize=8.6,
                    color=MODEL_COLORS[model],
                )

        acc_ax.set_ylim(acc_bottom, acc_top)
        lat_ax.set_ylim(min(all_lat) - 0.03, max(all_lat) + 0.04)
        acc_ax.set_title(preprocess, fontsize=13.5, color=PREPROCESS_COLORS[preprocess], pad=6, fontweight="bold")
        lat_ax.set_xticks(x)
        lat_ax.set_xticklabels(short_shot_labels())
        apply_clean_axes(acc_ax, grid_axis="y")
        apply_clean_axes(lat_ax, grid_axis="y")

        if col == 0:
            acc_ax.set_ylabel("准确率 (%)")
            lat_ax.set_ylabel("时延 (ms)")
        lat_ax.set_xlabel("Shot")

    fig.suptitle("Main Experiment Overview", fontsize=15, fontweight="bold", y=0.985)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.955), fontsize=10.5)
    fig.subplots_adjust(top=0.88)
    save_figure(fig, "fig5_1_main_experiment_overview")


if __name__ == "__main__":
    main()
