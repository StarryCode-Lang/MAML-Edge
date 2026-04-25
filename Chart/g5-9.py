from collections import defaultdict

from _paper_plot import (
    PROFILE_COLORS,
    VARIANT_LABELS,
    VARIANT_MARKERS,
    normalize_variant_rows,
    read_csv_rows,
    save_figure,
    to_float,
)

import matplotlib.pyplot as plt


SOURCE_FILES = [
    "logs/thesis_tables/paper_balanced/table4_compression_ablation.csv",
    "logs/thesis_tables/paper_balanced/paper_balanced_tables.json",
    "logs/thesis_tables/controlled/benchmark_rows.csv",
    "logs/thesis_tables/paper_balanced/paper_balanced_report.md",
    "test_layer/paper_tables.py",
    "test_layer/benchmark.py",
    "deploy_layer/runtime_backends.py",
    "deploy_layer/compression.py",
    "test_layer/run_compression_ablation.sh",
]

OUTPUT_STEM = "fig5_9_accuracy_vs_latency"


def load_ablation_rows():
    rows = read_csv_rows("logs/thesis_tables/paper_balanced/table4_compression_ablation.csv")
    parsed = []
    for row in rows:
        parsed.append({
            "profile": row["profile"],
            "variant": row["variant"],
            "accuracy_percent": to_float(row["accuracy_percent"]),
            "latency_ms": to_float(row["latency_ms"]),
        })
    grouped = defaultdict(list)
    for row in parsed:
        grouped[row["profile"]].append(row)
    return {profile: normalize_variant_rows(profile_rows) for profile, profile_rows in grouped.items()}

LABEL_OFFSETS = {
    "baseline": (10, -4),
    "prune_only": (6, -10),
    "prune_recovery": (-20, 8),
    "prune_recovery_int8": (6, 10),
    "int8_only": (6, -6),
}

def plot_profile(ax, profile, rows):
    color = PROFILE_COLORS.get(profile, "#345995")
    x_values = [row["latency_ms"] for row in rows]
    y_values = [row["accuracy_percent"] for row in rows]

    ax.plot(x_values, y_values, color=color, linewidth=1.5, alpha=0.85)
    for index, row in enumerate(rows):
        ax.scatter(
            row["latency_ms"],
            row["accuracy_percent"],
            color=color,
            marker=VARIANT_MARKERS[row["variant"]],
            s=78,
            zorder=3,
            label=profile if index == 0 else None,
        )
        offset = LABEL_OFFSETS.get(row["variant"], (4, 6))
        ax.annotate(
            VARIANT_LABELS[row["variant"]],
            xy=(row["latency_ms"], row["accuracy_percent"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            ha="left",
            va="bottom",
        )

DISPLAY_PROFILES = {"STFT"}

def main():
    grouped_rows = load_ablation_rows()
    grouped_rows = {
        profile: rows
        for profile, rows in grouped_rows.items()
        if any(keyword in profile for keyword in DISPLAY_PROFILES)
    }

    fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    for profile, rows in grouped_rows.items():
        plot_profile(ax, profile, rows)

    ax.set_xlabel("Deployment-layer inference latency (ms)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy-Latency Trade-off Across Compression Stages", fontsize=12)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=False, loc="lower left")

    png_path, pdf_path = save_figure(fig, OUTPUT_STEM)
    print("Generated:")
    print(png_path)
    print(pdf_path)
    print("Source files:")
    for source in SOURCE_FILES:
        print(source)


if __name__ == "__main__":
    main()
