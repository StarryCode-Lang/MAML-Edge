from _paper_plot import (
    SHOT_ORDER,
    percent_label,
    read_csv_rows,
    save_figure,
    safe_ylim,
    shot_tick_labels,
    to_float,
    to_int,
)

import matplotlib.pyplot as plt


SOURCE_FILES = [
    "logs/thesis_tables/paper_balanced/table2_few_shot_mean_std.csv",
    "logs/thesis_tables/paper_balanced/paper_balanced_tables.json",
    "logs/thesis_tables/paper_balanced/paper_balanced_report.md",
    "logs/thesis_tables/controlled/benchmark_rows.csv",
    "test_layer/paper_tables.py",
    "test_layer/thesis_tables.py",
    "test_layer/thesis_config.py",
]

OUTPUT_STEM = "fig5_2_shot_trend"


def load_shot_rows():
    rows = read_csv_rows("logs/thesis_tables/paper_balanced/table2_few_shot_mean_std.csv")
    parsed = []
    for row in rows:
        parsed.append({
            "shots": to_int(row["shots"]),
            "accuracy_mean_percent": to_float(row["accuracy_mean_percent"]),
            "accuracy_std_percent": to_float(row["accuracy_std_percent"]),
        })
    parsed.sort(key=lambda item: item["shots"])
    return parsed


def main():
    rows = load_shot_rows()
    x_values = list(range(len(rows)))
    shot_values = [row["shots"] for row in rows]
    means = [row["accuracy_mean_percent"] for row in rows]
    stds = [row["accuracy_std_percent"] for row in rows]

    best_mean_index = max(range(len(rows)), key=lambda index: means[index])
    most_stable_index = min(range(len(rows)), key=lambda index: stds[index])

    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    ax.errorbar(
        x_values,
        means,
        yerr=stds,
        fmt="-o",
        color="#1f5aa6",
        ecolor="#4c78a8",
        elinewidth=1.2,
        capsize=5,
        linewidth=2.0,
        markersize=7,
    )
    ax.scatter(
        [best_mean_index, most_stable_index],
        [means[best_mean_index], means[most_stable_index]],
        s=72,
        color=["#c56b18", "#2d8b57"],
        zorder=4,
    )

    ax.annotate(
        f"Highest mean: {percent_label(means[best_mean_index])}",
        xy=(best_mean_index, means[best_mean_index]),
        xytext=(best_mean_index - 0.35, means[best_mean_index] + 0.55),
        arrowprops={"arrowstyle": "->", "lw": 0.9},
        fontsize=9,
    )
    ax.annotate(
        f"Lowest std: {stds[most_stable_index]:.2f}%",
        xy=(most_stable_index, means[most_stable_index]),
        xytext=(most_stable_index - 0.25, means[most_stable_index] - 1.1),
        arrowprops={"arrowstyle": "->", "lw": 0.9},
        fontsize=9,
    )

    ax.set_xticks(x_values)
    ax.set_xticklabels(shot_tick_labels())
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Few-shot setting")
    ax.set_title("STFT + MAML Accuracy Trend Across Few-Shot Settings", fontsize=12)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
    ymin, ymax = safe_ylim([value - error for value, error in zip(means, stds)] + [value + error for value, error in zip(means, stds)])
    ax.set_ylim(ymin, ymax)

    png_path, pdf_path = save_figure(fig, OUTPUT_STEM)
    print("Generated:")
    print(png_path)
    print(pdf_path)
    print("Source files:")
    for source in SOURCE_FILES:
        print(source)


if __name__ == "__main__":
    main()
