from _paper_plot import read_csv_rows, save_figure, safe_ylim, to_float

import matplotlib.pyplot as plt


SOURCE_FILES = [
    "logs/thesis_tables/paper_balanced/table1_model_performance_mean_std.csv",
    "logs/thesis_tables/paper_balanced/paper_balanced_tables.json",
    "logs/thesis_tables/paper_balanced/paper_balanced_report.md",
    "test_layer/paper_tables.py",
    "test_layer/thesis_tables.py",
    "logs/thesis_tables/paper_balanced/seed/seed43/thesis_tables.json",
    "logs/thesis_tables/paper_balanced/seed/seed44/thesis_tables.json",
]

OUTPUT_STEM = "fig5_4_model_stability_errorbar"


def load_model_rows():
    rows = read_csv_rows("logs/thesis_tables/paper_balanced/table1_model_performance_mean_std.csv")
    parsed = []
    for row in rows:
        parsed.append({
            "model": row["model"],
            "accuracy_mean_percent": to_float(row["accuracy_mean_percent"]),
            "accuracy_std_percent": to_float(row["accuracy_std_percent"]),
        })
    return parsed


def main():
    rows = load_model_rows()
    x_values = list(range(len(rows)))
    means = [row["accuracy_mean_percent"] for row in rows]
    stds = [row["accuracy_std_percent"] for row in rows]
    labels = [row["model"] for row in rows]

    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    ax.errorbar(
        x_values,
        means,
        yerr=stds,
        fmt="o",
        linestyle="none",
        color="#123b68",
        ecolor="#123b68",
        elinewidth=1.5,
        capsize=6,
        markersize=8,
    )
    ax.plot(x_values, means, color="#6b7c93", linewidth=1.2, alpha=0.8)

    for x_value, mean_value, std_value, label in zip(x_values, means, stds, labels):
        ax.text(
            x_value,
            mean_value + std_value + 0.6,
            f"{label}\n{mean_value:.2f}±{std_value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x_values)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Model")
    ax.set_title("Model Stability Under STFT + 5-shot Setting", fontsize=12)
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
