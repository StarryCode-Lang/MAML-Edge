import time

from edge_layer.simulator.preprocess import compute_feature_summary, detect_event


def build_signal_payload(device_id, raw_signal, temperature=None, metadata=None):
    features = compute_feature_summary(raw_signal)
    payload = {
        'device_id': device_id,
        'timestamp': int(time.time()),
        'temperature': temperature,
        'raw_signal': [float(item) for item in raw_signal],
        'feature_summary': features,
        'event_triggered': detect_event(features),
    }
    if metadata is not None:
        payload['metadata'] = metadata
    return payload


def example_payload():
    signal = [0.01, 0.03, 0.02, 0.15, 0.22, 0.18, 0.03, 0.02]
    return build_signal_payload('esp32-demo', signal, temperature=36.5)
