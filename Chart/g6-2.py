import json

from _paper_plot import read_csv_rows, save_figure, to_float

import matplotlib.pyplot as plt


SOURCE_FILES = [
    "logs/thesis_tables/paper_balanced/system_benchmark.json",
    "logs/thesis_tables/paper_balanced/table5_system_performance.csv",
    "logs/thesis_tables/paper_balanced/paper_balanced_tables.json",
    "test_layer/system_benchmark.py",
    "test_layer/paper_tables.py",
    "system_layer/backend/service_stats.py",
    "system_layer/backend/predictor.py",
    "system_layer/backend/main.py",
    "test_layer/run_system_benchmark.sh",
]

OUTPUT_STEM = "fig6_2_system_latency_breakdown"


def load_system_rows():
    rows = read_csv_rows("logs/thesis_tables/paper_balanced/table5_system_performance.csv")
    parsed = []
    for row in rows:
        parsed.append({
            "stage": row["stage"],
            "latency_ms": to_float(row["latency_ms"]),
        })
    return parsed


def main():
    rows = load_system_rows()
    stage_order = ["preprocess", "inference", "end_to_end"]
    stage_labels = ["Preprocess", "Inference", "End-to-End"]
    stage_values = [
        next(row["latency_ms"] for row in rows if row["stage"] == stage_name)
        for stage_name in stage_order
    ]
    inference_ratio = stage_values[1] / stage_values[2] * 100.0

    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    colors = ["#a05a2c", "#2b7a78", "#3c6eaf"]
    bars = ax.bar(stage_labels, stage_values, color=colors, edgecolor="black", linewidth=0.5)
    for bar, value in zip(bars, stage_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.9,
            f"{value:.4f} ms",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.annotate(
        f"Inference accounts for only {inference_ratio:.2f}% of end-to-end latency",
        xy=(1, stage_values[1]),
        xytext=(0.22, max(stage_values) * 0.63),
        textcoords="data",
        arrowprops={"arrowstyle": "->", "lw": 0.9},
        fontsize=9,
    )

    ax.set_ylabel("Latency (ms)")
    ax.set_xlabel("System stage")
    ax.set_title("System-layer Latency Breakdown on Direct Channel", fontsize=12)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)

    png_path, pdf_path = save_figure(fig, OUTPUT_STEM)
    print("Generated:")
    print(png_path)
    print(pdf_path)
    print("Source files:")
    for source in SOURCE_FILES:
        print(source)


if __name__ == "__main__":
    main()
