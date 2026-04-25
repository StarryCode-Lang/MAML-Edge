from _paper_plot import (
    MODEL_ORDER,
    PREPROCESS_ORDER,
    SHOT_ORDER,
    read_csv_rows,
    save_figure,
    to_float,
    to_int,
)

import matplotlib.pyplot as plt
import numpy as np


SOURCE_FILES = [
    "logs/thesis_tables/paper_balanced/table0_preprocess_model_matrix_mean_std.csv",
    "logs/thesis_tables/paper_balanced/paper_balanced_tables.json",
    "logs/thesis_tables/paper_balanced/paper_balanced_report.md",
    "logs/thesis_tables/controlled/thesis_tables.json",
    "logs/thesis_tables/controlled/benchmark_rows.csv",
    "test_layer/paper_tables.py",
    "test_layer/thesis_tables.py",
    "test_layer/benchmark.py",
    "test_layer/result_aggregator.py",
]

OUTPUT_STEM = "fig5_3_main_experiment_heatmap"


def load_matrix_rows():
    rows = read_csv_rows("logs/thesis_tables/paper_balanced/table0_preprocess_model_matrix_mean_std.csv")
    parsed = []
    for row in rows:
        parsed.append({
            "preprocess": row["preprocess"],
            "model": row["model"],
            "shots": to_int(row["shots"]),
            "accuracy_mean_percent": to_float(row["accuracy_mean_percent"]),
            "accuracy_std_percent": to_float(row["accuracy_std_percent"]),
        })
    return parsed


def build_shot_matrix(rows, shot_value):
    matrix = np.zeros((len(PREPROCESS_ORDER), len(MODEL_ORDER)), dtype=float)
    labels = [["" for _ in MODEL_ORDER] for _ in PREPROCESS_ORDER]
    for row in rows:
        if row["shots"] != shot_value:
            continue
        preprocess_index = PREPROCESS_ORDER.index(row["preprocess"])
        model_index = MODEL_ORDER.index(row["model"])
        matrix[preprocess_index, model_index] = row["accuracy_mean_percent"]
        labels[preprocess_index][model_index] = (
            f"{row['accuracy_mean_percent']:.2f}\n±{row['accuracy_std_percent']:.2f}"
        )
    return matrix, labels


def main():
    rows = load_matrix_rows()
    matrices = []
    for shot_value in SHOT_ORDER:
        matrix, labels = build_shot_matrix(rows, shot_value)
        matrices.append((shot_value, matrix, labels))

    global_min = min(matrix.min() for _, matrix, _ in matrices)
    global_max = max(matrix.max() for _, matrix, _ in matrices)

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.9), constrained_layout=True)
    colorbar_image = None

    for axis, (shot_value, matrix, labels) in zip(axes, matrices):
        colorbar_image = axis.imshow(matrix, cmap="viridis", vmin=global_min, vmax=global_max, aspect="auto")
        axis.set_xticks(range(len(MODEL_ORDER)))
        axis.set_xticklabels(MODEL_ORDER)
        axis.set_yticks(range(len(PREPROCESS_ORDER)))
        axis.set_yticklabels(PREPROCESS_ORDER)
        axis.set_title(f"{shot_value}-shot", fontsize=11)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                text_color = "white" if value < (global_min + global_max) / 2.0 else "black"
                axis.text(
                    column_index,
                    row_index,
                    labels[row_index][column_index],
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                )

    axes[0].set_ylabel("Preprocess")
    for axis in axes:
        axis.set_xlabel("Model")

    colorbar = fig.colorbar(colorbar_image, ax=axes, fraction=0.024, pad=0.02)
    colorbar.set_label("Accuracy (%)")
    fig.suptitle("Main Experiment Overview by Preprocess, Model, and Shot", fontsize=12)

    png_path, pdf_path = save_figure(fig, OUTPUT_STEM)
    print("Generated:")
    print(png_path)
    print(pdf_path)
    print("Source files:")
    for source in SOURCE_FILES:
        print(source)


if __name__ == "__main__":
    main()
