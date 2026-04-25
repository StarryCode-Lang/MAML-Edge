from collections import defaultdict
import statistics

from _paper_plot import read_csv_rows, save_figure, to_float, to_int

import matplotlib.pyplot as plt


SOURCE_FILES = [
    "logs/thesis_tables/paper_balanced/table3_domain_robustness.csv",
    "logs/thesis_tables/paper_balanced/paper_balanced_tables.json",
    "logs/thesis_tables/paper_balanced/paper_balanced_report.md",
    "test_layer/paper_tables.py",
    "test_layer/thesis_config.py",
    "logs/thesis_tables/paper_balanced/domain/source013_target2/thesis_tables.json",
    "logs/thesis_tables/paper_balanced/domain/source023_target1/thesis_tables.json",
    "logs/thesis_tables/paper_balanced/domain/source123_target0/thesis_tables.json",
    "logs/thesis_tables/controlled/benchmark_rows.csv",
]

OUTPUT_STEM = "fig5_6_domain_robustness_and_difficulty"


def load_domain_rows():
    rows = read_csv_rows("logs/thesis_tables/paper_balanced/table3_domain_robustness.csv")
    parsed = []
    for row in rows:
        parsed.append({
            "train_domains": row["train_domains"],
            "test_domain": to_int(row["test_domain"]),
            "accuracy_percent": to_float(row["accuracy_percent"]),
        })
    return parsed


def summarize_domain_means(rows):
    grouped = defaultdict(list)
    for row in rows:
        split_label = f"{row['train_domains'].replace(',', '')}->{row['test_domain']}"
        grouped[split_label].append(row["accuracy_percent"])
    split_means = []
    for split_label, values in grouped.items():
        split_means.append({
            "split": split_label,
            "test_domain": int(split_label.split("->")[1]),
            "mean_accuracy": statistics.mean(values),
        })
    split_means.sort(key=lambda item: item["test_domain"], reverse=True)
    return split_means


def main():
    rows = load_domain_rows()
    split_means = summarize_domain_means(rows)
    difficulty_ranking = sorted(split_means, key=lambda item: item["mean_accuracy"])

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.8), constrained_layout=True, gridspec_kw={"width_ratios": [1.25, 1.0]})

    left_axis, right_axis = axes

    split_labels = [item["split"] for item in split_means]
    split_values = [item["mean_accuracy"] for item in split_means]
    bar_colors = ["#5b8def" if item["test_domain"] != 0 else "#cf5f34" for item in split_means]
    left_axis.bar(split_labels, split_values, color=bar_colors, edgecolor="black", linewidth=0.5)
    for x_index, value in enumerate(split_values):
        left_axis.text(x_index, value + 0.35, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    left_axis.set_ylabel("Mean accuracy (%)")
    left_axis.set_xlabel("Held-out target split")
    left_axis.set_title("Cross-domain Robustness by Held-out Target", fontsize=11)
    left_axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
    left_axis.set_ylim(min(split_values) - 2.5, 101.0)

    ranking_labels = [f"Target {item['test_domain']}" for item in difficulty_ranking]
    ranking_values = [item["mean_accuracy"] for item in difficulty_ranking]
    right_axis.barh(ranking_labels, ranking_values, color="#7a9f35", edgecolor="black", linewidth=0.5)
    for y_index, value in enumerate(ranking_values):
        right_axis.text(value + 0.25, y_index, f"{value:.2f}", va="center", fontsize=8)
    right_axis.set_xlabel("Mean accuracy (%)")
    right_axis.set_title("Target-domain Difficulty Ranking", fontsize=11)
    right_axis.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.4)
    right_axis.set_xlim(min(ranking_values) - 2.5, 101.0)
    hardest = difficulty_ranking[0]
    right_axis.annotate(
        "Hardest domain",
        xy=(hardest["mean_accuracy"], 0),
        xytext=(hardest["mean_accuracy"] + 3.2, 0.35),
        arrowprops={"arrowstyle": "->", "lw": 0.9},
        fontsize=9,
    )

    fig.suptitle("Cross-domain Generalization and Target-domain Difficulty", fontsize=12)

    png_path, pdf_path = save_figure(fig, OUTPUT_STEM)
    print("Generated:")
    print(png_path)
    print(pdf_path)
    print("Source files:")
    for source in SOURCE_FILES:
        print(source)


if __name__ == "__main__":
    main()
