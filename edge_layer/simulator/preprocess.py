import numpy as np


def compute_rms(signal):
    signal = np.asarray(signal, dtype=np.float32)
    return float(np.sqrt(np.mean(np.square(signal))))


def compute_peak(signal):
    signal = np.asarray(signal, dtype=np.float32)
    return float(np.max(np.abs(signal)))


def compute_fft_summary(signal, top_k=8):
    signal = np.asarray(signal, dtype=np.float32)
    spectrum = np.fft.rfft(signal)
    magnitude = np.abs(spectrum) / max(1, signal.shape[0])
    limited = magnitude[:max(1, top_k)]
    return [float(item) for item in limited]


def compute_feature_summary(signal):
    signal = np.asarray(signal, dtype=np.float32)
    return {
        'rms': compute_rms(signal),
        'peak': compute_peak(signal),
        'mean': float(np.mean(signal)),
        'std': float(np.std(signal)),
        'fft_summary': compute_fft_summary(signal),
    }


def detect_event(features, rms_threshold=0.15, peak_threshold=0.4):
    return bool(
        features.get('rms', 0.0) >= rms_threshold or
        features.get('peak', 0.0) >= peak_threshold
    )
