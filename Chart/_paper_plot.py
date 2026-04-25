from pathlib import Path
import csv
import json
import math

import matplotlib.pyplot as plt
import matplotlib as mpl


ROOT_DIR = Path(__file__).resolve().parents[1]
CHART_DIR = Path(__file__).resolve().parent

try:
    plt.style.use("seaborn-paper")
except OSError:
    pass

mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["axes.linewidth"] = 0.8
mpl.rcParams["figure.dpi"] = 120

PREPROCESS_ORDER = ["FFT", "STFT", "WT"]
MODEL_ORDER = ["CNN", "MAML", "ProtoNet"]
SHOT_ORDER = [5, 10, 15]

PROFILE_COLORS = {
    "FFT + ProtoNet + 5-shot": "#2a6fdb",
    "STFT + MAML + 5-shot": "#c97a1f",
}

VARIANT_ORDER = [
    "baseline",
    "int8_only",
    "prune_only",
    "prune_recovery_float",
    "prune_recovery_int8",
]

VARIANT_LABELS = {
    "baseline": "Baseline",
    "int8_only": "INT8 only",
    "prune_only": "Prune only",
    "prune_recovery_float": "Prune + recovery",
    "prune_recovery_int8": "Prune + recovery + INT8",
}

VARIANT_MARKERS = {
    "baseline": "o",
    "int8_only": "s",
    "prune_only": "^",
    "prune_recovery_float": "D",
    "prune_recovery_int8": "P",
}


def repo_path(relative_path):
    return ROOT_DIR / relative_path


def chart_output_stem(stem):
    return CHART_DIR / stem


def read_csv_rows(relative_path):
    with open(repo_path(relative_path), "r", encoding="utf-8-sig", newline="") as file_pointer:
        return list(csv.DictReader(file_pointer))


def read_json(relative_path):
    with open(repo_path(relative_path), "r", encoding="utf-8") as file_pointer:
        return json.load(file_pointer)


def to_float(value):
    if value is None or value == "":
        return None
    return float(value)


def to_int(value):
    if value is None or value == "":
        return None
    return int(value)


def percent_label(value):
    if value is None:
        return "-"
    return f"{value:.2f}%"


def save_figure(fig, output_stem, dpi=1200):
    png_path = chart_output_stem(f"{output_stem}.png")
    pdf_path = chart_output_stem(f"{output_stem}.pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    return png_path, pdf_path


def shot_tick_labels():
    return [f"{shot}-shot" for shot in SHOT_ORDER]


def midpoint(values):
    if not values:
        return None
    return sum(values) / float(len(values))


def normalize_variant_rows(rows):
    ranked = []
    for row in rows:
        variant = row["variant"]
        if variant not in VARIANT_ORDER:
            continue
        ranked.append((VARIANT_ORDER.index(variant), row))
    ranked.sort(key=lambda item: item[0])
    return [row for _, row in ranked]


def safe_ylim(values, padding_ratio=0.08, lower_bound=None, upper_bound=None):
    cleaned = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    if not cleaned:
        return None, None
    minimum = min(cleaned)
    maximum = max(cleaned)
    if minimum == maximum:
        delta = max(1.0, abs(minimum) * 0.05)
        minimum -= delta
        maximum += delta
    else:
        padding = (maximum - minimum) * padding_ratio
        minimum -= padding
        maximum += padding
    if lower_bound is not None:
        minimum = max(lower_bound, minimum)
    if upper_bound is not None:
        maximum = min(upper_bound, maximum)
    return minimum, maximum
