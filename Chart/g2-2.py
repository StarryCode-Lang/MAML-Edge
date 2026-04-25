from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image

# -----------------------------
# Paper-style plotting settings
# -----------------------------
plt.style.use('seaborn-paper')
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["axes.linewidth"] = 0.8

# -----------------------------
# Paths
# -----------------------------
DATA_ROOT = Path("../data")
RAW_ROOT = DATA_ROOT / "CWRU_12k"
STFT_ROOT = DATA_ROOT / "STFT_CWRU"
WT_ROOT = DATA_ROOT / "WT_CWRU"

OUTPUT_PNG = "fig2_2_preprocessing_representations.png"

# -----------------------------
# Project-consistent params
# -----------------------------
TIME_STEPS = 1024
OVERLAP_RATIO = 0.5
STRIDE = int(TIME_STEPS * (1 - OVERLAP_RATIO))
FS = 12000

# CWRU mapping from preprocess_cwru.py
DATANAME_DICT = {
    0: [97, 105, 118, 130, 169, 185, 197, 209, 222, 234],
    1: [98, 106, 119, 131, 170, 186, 198, 210, 223, 235],
    2: [99, 107, 120, 132, 171, 187, 199, 211, 224, 236],
    3: [100, 108, 121, 133, 172, 188, 200, 212, 225, 237],
}

AXIS_SUFFIX = "_DE_time"


def build_mat_key(file_id: int) -> str:
    if file_id < 100:
        return f"X0{file_id}{AXIS_SUFFIX}"
    return f"X{file_id}{AXIS_SUFFIX}"


def load_raw_signal(domain: int, label: int) -> np.ndarray:
    file_id = DATANAME_DICT[domain][label]
    mat_path = RAW_ROOT / f"Drive_end_{domain}" / f"{file_id}.mat"
    mat_key = build_mat_key(file_id)

    if not mat_path.exists():
        raise FileNotFoundError(f"Missing raw file: {mat_path}")

    mat = sio.loadmat(mat_path)
    if mat_key not in mat:
        raise KeyError(f"Key {mat_key} not found in {mat_path}")

    return mat[mat_key].reshape(-1)


def slice_window(signal: np.ndarray, sample_index: int) -> np.ndarray:
    start = sample_index * STRIDE
    end = start + TIME_STEPS
    if end > len(signal):
        raise IndexError("Requested sample index exceeds available windows.")
    return signal[start:end]


def fft_representation(signal_window: np.ndarray):
    freq = np.fft.rfftfreq(TIME_STEPS, d=1 / FS)
    mag = np.abs(np.fft.rfft(signal_window)) / TIME_STEPS
    return freq, mag


def parse_image_index(image_path: Path):
    label_str, sample_str = image_path.stem.split("_")
    return int(label_str), int(sample_str)


def collect_domain_image_indices(image_root: Path, domain: int):
    image_dir = image_root / f"Drive_end_{domain}"
    indices_by_label = {}
    if not image_dir.exists():
        return indices_by_label

    for image_path in image_dir.glob("*.png"):
        try:
            label, sample_index = parse_image_index(image_path)
        except ValueError:
            continue
        indices_by_label.setdefault(label, {})[sample_index] = image_path
    return indices_by_label


def collect_raw_metadata():
    metadata = {}
    for domain, file_ids in DATANAME_DICT.items():
        metadata[domain] = {}
        for label, file_id in enumerate(file_ids):
            raw_signal = load_raw_signal(domain, label)
            max_windows = (len(raw_signal) - TIME_STEPS) // STRIDE + 1
            metadata[domain][label] = {
                "file_id": file_id,
                "raw_signal": raw_signal,
                "max_windows": max_windows,
            }
    return metadata


def build_candidate_inventory():
    raw_metadata = collect_raw_metadata()
    inventory = []

    for domain in range(4):
        stft_indices = collect_domain_image_indices(STFT_ROOT, domain)
        wt_indices = collect_domain_image_indices(WT_ROOT, domain)

        for label in range(10):
            raw_info = raw_metadata[domain][label]
            raw_sample_indices = set(range(raw_info["max_windows"]))
            stft_sample_indices = set(stft_indices.get(label, {}).keys())
            wt_sample_indices = set(wt_indices.get(label, {}).keys())

            common_indices = sorted(raw_sample_indices & stft_sample_indices & wt_sample_indices)
            if not common_indices:
                continue

            inventory.append({
                "domain": domain,
                "label": label,
                "file_id": raw_info["file_id"],
                "raw_signal": raw_info["raw_signal"],
                "common_indices": common_indices,
                "stft_paths": stft_indices[label],
                "wt_paths": wt_indices[label],
            })

    if not inventory:
        raise RuntimeError("No aligned sample found across raw/STFT/WT data.")

    return inventory


def select_representative_sample(inventory):
    """
    Scan all aligned candidates first, then select a representative one.
    Preference:
    1. More aligned windows under the same domain/label pair
    2. Lower domain index
    3. Lower label index
    4. Median sample index within that aligned set
    """
    ranked_groups = sorted(
        inventory,
        key=lambda item: (-len(item["common_indices"]), item["domain"], item["label"])
    )
    chosen_group = ranked_groups[0]
    chosen_index = chosen_group["common_indices"][len(chosen_group["common_indices"]) // 2]

    return {
        "domain": chosen_group["domain"],
        "label": chosen_group["label"],
        "file_id": chosen_group["file_id"],
        "sample_index": chosen_index,
        "raw_signal": chosen_group["raw_signal"],
        "stft_path": chosen_group["stft_paths"][chosen_index],
        "wt_path": chosen_group["wt_paths"][chosen_index],
        "aligned_count": len(chosen_group["common_indices"]),
        "candidate_group_count": len(inventory),
    }


def main():
    inventory = build_candidate_inventory()
    sample = select_representative_sample(inventory)

    domain = sample["domain"]
    label = sample["label"]
    file_id = sample["file_id"]
    sample_index = sample["sample_index"]

    raw_signal = sample["raw_signal"]
    window = slice_window(raw_signal, sample_index)
    t = np.arange(TIME_STEPS) / FS
    freq, fft_mag = fft_representation(window)

    stft_img = np.array(Image.open(sample["stft_path"]).convert("RGB"))
    wt_img = np.array(Image.open(sample["wt_path"]).convert("RGB"))

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 5.0), constrained_layout=True)

    # (a) Raw signal
    axes[0].plot(t, window, lw=1.0, color="black")
    axes[0].set_title("(a) Raw signal", fontsize=11)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].text(
        0.5, -0.28,
        f"Domain {domain}, file {file_id}.mat, label {label}, sample {sample_index}",
        transform=axes[0].transAxes,
        ha="center",
        fontsize=8
    )

    # (b) FFT
    axes[1].plot(freq, fft_mag, lw=1.0, color="black")
    axes[1].set_title("(b) FFT spectrum", fontsize=11)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Magnitude")
    axes[1].text(
        0.5, -0.28,
        "1D input for CNN1D encoder",
        transform=axes[1].transAxes,
        ha="center",
        fontsize=8
    )

    # (c) STFT
    axes[2].imshow(stft_img)
    axes[2].set_title("(c) STFT representation", fontsize=11)
    axes[2].axis("off")
    axes[2].text(
        0.5, -0.10,
        "Stored in STFT_CWRU, used as 2D input",
        transform=axes[2].transAxes,
        ha="center",
        fontsize=8
    )

    # (d) WT
    axes[3].imshow(wt_img)
    axes[3].set_title("(d) WT representation", fontsize=11)
    axes[3].axis("off")
    axes[3].text(
        0.5, -0.10,
        "Stored in WT_CWRU, used as 2D input",
        transform=axes[3].transAxes,
        ha="center",
        fontsize=8
    )

    fig.suptitle(
        "Preprocessing representations of the same real CWRU vibration segment",
        fontsize=12
    )

    fig.savefig(OUTPUT_PNG, dpi=1200, bbox_inches="tight")

    print("Generated:")
    print(OUTPUT_PNG)
    print(f"Scanned candidate groups: {sample['candidate_group_count']}")
    print(f"Selected domain={domain}, label={label}, sample_index={sample_index}")
    print(f"Selected raw file: {file_id}.mat")
    print(f"Aligned samples in selected group: {sample['aligned_count']}")
    print(f"STFT image: {sample['stft_path']}")
    print(f"WT image: {sample['wt_path']}")


if __name__ == "__main__":
    main()
