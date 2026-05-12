from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from plot_common import MODEL_ORDER, PREPROCESS_ORDER, SHOT_ORDER, build_matrix_mean_std_lookup, save_figure


def text_color(value: float) -> str:
    return "white" if value < 85 else "#111111"


def main() -> None:
    lookup = build_matrix_mean_std_lookup()
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.7), constrained_layout=True)
    values = [lookup[(p, m, s)]["accuracy_mean_percent"] for p in PREPROCESS_ORDER for m in MODEL_ORDER for s in SHOT_ORDER]
    vmin = min(values)
    vmax = max(values)
    cmap = plt.cm.YlGnBu

    fig.suptitle("Accuracy Matrix Heatmap across Preprocessing, Model, and Shot Settings", fontsize=13, fontweight="bold", y=1.02)

    for ax, shot in zip(axes, SHOT_ORDER):
        matrix = np.array(
            [[lookup[(preprocess, model, shot)]["accuracy_mean_percent"] for model in MODEL_ORDER] for preprocess in PREPROCESS_ORDER]
        )
        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(f"{shot}-shot", fontsize=13, pad=7)
        ax.set_xticks(np.arange(len(MODEL_ORDER)))
        ax.set_xticklabels(MODEL_ORDER)
        ax.set_yticks(np.arange(len(PREPROCESS_ORDER)))
        ax.set_yticklabels(PREPROCESS_ORDER)

        for row_idx, preprocess in enumerate(PREPROCESS_ORDER):
            for col_idx, model in enumerate(MODEL_ORDER):
                item = lookup[(preprocess, model, shot)]
                ax.text(
                    col_idx,
                    row_idx,
                    f"{item['accuracy_mean_percent']:.2f}%\n±{item['accuracy_std_percent']:.2f} | {item['latency_mean_ms']:.2f}ms",
                    ha="center",
                    va="center",
                    fontsize=8.8,
                    color=text_color(item["accuracy_mean_percent"]),
                    fontweight="bold" if item["accuracy_mean_percent"] >= 99.5 else None,
                )

        ax.set_xticks(np.arange(-0.5, len(MODEL_ORDER), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(PREPROCESS_ORDER), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=axes, shrink=0.92, pad=0.012)
    cbar.set_label("准确率均值 (%)")
    save_figure(fig, "fig5_2_main_experiment_matrix_heatmap")


if __name__ == "__main__":
    main()
