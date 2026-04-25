from _paper_plot import read_csv_rows, save_figure, safe_ylim, shot_tick_labels, to_float, to_int

import matplotlib.pyplot as plt


SOURCE_FILES = [
    "logs/thesis_tables/paper_balanced/table2_few_shot_mean_std.csv",
    "logs/thesis_tables/paper_balanced/paper_balanced_tables.json",
    "logs/thesis_tables/paper_balanced/paper_balanced_report.md",
    "test_layer/paper_tables.py",
    "test_layer/thesis_tables.py",
    "test_layer/thesis_config.py",
    "logs/thesis_tables/paper_balanced/seed/seed43/table2_few_shot.csv",
    "logs/thesis_tables/paper_balanced/seed/seed44/table2_few_shot.csv",
    "logs/thesis_tables/controlled/table2_few_shot.csv",
]

OUTPUT_STEM = "fig5_5_few_shot_stability"


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
    means = [row["accuracy_mean_percent"] for row in rows]
    stds = [row["accuracy_std_percent"] for row in rows]
    lower = [mean - std for mean, std in zip(means, stds)]
    upper = [mean + std for mean, std in zip(means, stds)]

    most_stable_index = min(range(len(rows)), key=lambda index: stds[index])

    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    ax.plot(x_values, means, color="#146c6e", linewidth=2.1, marker="o", markersize=7)
    ax.fill_between(x_values, lower, upper, color="#6fc2c0", alpha=0.25, linewidth=0)
    ax.scatter(
        [most_stable_index],
        [means[most_stable_index]],
        s=84,
        color="#cf5f34",
        zorder=4,
    )
    ax.annotate(
        f"Most stable: {rows[most_stable_index]['shots']}-shot\nstd = {stds[most_stable_index]:.2f}%",
        xy=(most_stable_index, means[most_stable_index]),
        xytext=(most_stable_index - 0.45, means[most_stable_index] - 1.3),
        arrowprops={"arrowstyle": "->", "lw": 0.9},
        fontsize=9,
    )

    ax.set_xticks(x_values)
    ax.set_xticklabels(shot_tick_labels())
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Few-shot setting")
    ax.set_title("Few-Shot Stability Band for STFT + MAML", fontsize=12)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
    ymin, ymax = safe_ylim(lower + upper)
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
