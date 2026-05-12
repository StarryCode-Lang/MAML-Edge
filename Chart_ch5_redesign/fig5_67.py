from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from plot_common import (
    PROFILE_COLORS, STAGE_MARKERS, STAGE_LABELS,
    stage_sorted, read_csv_rows, apply_clean_axes, save_figure
)

def main() -> None:
    rows = stage_sorted(read_csv_rows("logs/thesis_tables/paper_balanced/table4_compression_ablation.csv"))

    profiles = []
    for row in rows:
        if row["profile"] not in profiles and "MAML" in row["profile"]:
            profiles.append(row["profile"])

    fig = plt.figure(figsize=(12, 5.0))
    fig.suptitle(
        "Model Compression: Accuracy vs. Model Size & Latency",
        fontsize=14, fontweight="bold", y=0.98
    )
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.24)

    ax1 = fig.add_subplot(gs[0, 0])

    label_offsets_ms = {
        ("STFT + MAML + 5-shot", "baseline"): (0.008, 0.08),
        ("STFT + MAML + 5-shot", "int8_only"): (0.008, -0.10),
        ("STFT + MAML + 5-shot", "prune_only"): (0.008, -0.18),
        ("STFT + MAML + 5-shot", "prune_recovery_float"): (0.008, -0.26),
        ("STFT + MAML + 5-shot", "prune_recovery_int8"): (0.008, -0.34),
    }

    for profile in profiles:
        pr = [r for r in rows if r["profile"] == profile]
        x = [float(r["model_size_mb"]) for r in pr]
        y = [float(r["accuracy_percent"]) for r in pr]
        ax1.plot(x, y, color=PROFILE_COLORS[profile], linewidth=2.4, alpha=0.95, label=profile)

        for r in pr:
            xv, yv = float(r["model_size_mb"]), float(r["accuracy_percent"])
            ax1.scatter(xv, yv, s=130, color=PROFILE_COLORS[profile],
                        marker=STAGE_MARKERS[r["variant"]],
                        edgecolors="white", linewidth=0.8, zorder=3)

            dx, dy = label_offsets_ms[(profile, r["variant"])]
            label_map = {
                "baseline": "baseline",
                "int8_only": "INT8",
                "prune_only": "P",
                "prune_recovery_float": "PR_F",
                "prune_recovery_int8": "PR_INT8",
            }
            ax1.text(xv + dx, yv + dy, label_map[r["variant"]],
                     fontsize=9, color="#333", ha="left", va="center")

    ax1.set_xlabel("Model Size (MB)", fontsize=11)
    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.set_xlim(0.01, 0.46)
    ax1.set_ylim(98.0, 99.5)
    apply_clean_axes(ax1, grid_axis="both")
    ax1.legend(loc="lower right", fontsize=9, framealpha=0.9)

    ax2 = fig.add_subplot(gs[0, 1])
    label_offsets_lat = {
        ("STFT + MAML + 5-shot", "baseline"): (0.08, 0.00),
        ("STFT + MAML + 5-shot", "int8_only"): (0.08, -0.04),
        ("STFT + MAML + 5-shot", "prune_only"): (0.08, -0.08),
        ("STFT + MAML + 5-shot", "prune_recovery_float"): (0.08, -0.12),
        ("STFT + MAML + 5-shot", "prune_recovery_int8"): (0.08, -0.16),
    }

    summary_rows = []
    for profile in profiles:
        pr = [r for r in rows if r["profile"] == profile]
        x = [float(r["latency_ms"]) for r in pr]
        y = [float(r["accuracy_percent"]) for r in pr]
        ax2.plot(x, y, color=PROFILE_COLORS[profile], linewidth=2.4, alpha=0.95)

        for r in pr:
            xv, yv = float(r["latency_ms"]), float(r["accuracy_percent"])
            ax2.scatter(xv, yv, s=130, color=PROFILE_COLORS[profile],
                        marker=STAGE_MARKERS[r["variant"]],
                        edgecolors="white", linewidth=0.8, zorder=3)

            dx, dy = label_offsets_lat[(profile, r["variant"])]
            lbl = STAGE_LABELS[r["variant"]]
            ax2.text(xv + dx, yv + dy, lbl, fontsize=8.8, color="#333")

        fastest = min(pr, key=lambda r: float(r["latency_ms"]))
        most_acc = max(pr, key=lambda r: float(r["accuracy_percent"]))
        summary_rows.append((
            profile,
            STAGE_LABELS[fastest["variant"]], float(fastest["latency_ms"]),
            STAGE_LABELS[most_acc["variant"]], float(most_acc["accuracy_percent"])
        ))

    ax2.set_xlabel("Latency (ms)", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_xlim(2.7, 3.4)
    ax2.set_ylim(98.0, 99.5)
    apply_clean_axes(ax2, grid_axis="both")

    ax2_text = ax2.inset_axes([0.62, 0.65, 0.36, 0.3])
    ax2_text.axis("off")
    ypos = 0.9
    for prof, fast_lbl, fast_val, best_lbl, best_val in summary_rows:
        ax2_text.text(0, ypos, prof, fontsize=10, fontweight="bold", color=PROFILE_COLORS[prof], transform=ax2_text.transAxes)
        ax2_text.text(0, ypos - 0.15, f"Fastest: {fast_lbl} ({fast_val:.2f}ms)", fontsize=9, transform=ax2_text.transAxes)
        ax2_text.text(0, ypos - 0.28, f"Best Acc: {best_lbl} ({best_val:.2f}%)", fontsize=9, transform=ax2_text.transAxes)
        ypos -= 0.4

    save_figure(fig, "fig5_combined_compression_size_latency")

if __name__ == "__main__":
    main()