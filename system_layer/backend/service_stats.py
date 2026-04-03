import threading
import time


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


class ServiceStats:
    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self._channels = {
                'direct': self._empty_channel_state(),
                'mqtt': self._empty_channel_state(),
            }
            self._alert_count = 0
            self._adaptation = {
                'request_count': 0,
                'sample_count': 0,
                'updated_labels': [],
                'last_status': None,
                'last_result': None,
            }
            self._started_at = time.time()

    def _empty_channel_state(self):
        return {
            'request_count': 0,
            'total_preprocess_latency_ms': 0.0,
            'total_inference_latency_ms': 0.0,
            'total_end_to_end_latency_ms': 0.0,
            'last_result': None,
        }

    def record_prediction(self, result, source='direct', alert_raised=False):
        channel = 'mqtt' if source == 'mqtt' else 'direct'
        with self._lock:
            state = self._channels[channel]
            state['request_count'] += 1
            state['total_preprocess_latency_ms'] += _safe_float(result.get('preprocess_latency_ms'))
            state['total_inference_latency_ms'] += _safe_float(result.get('inference_latency_ms'))
            state['total_end_to_end_latency_ms'] += _safe_float(
                result.get('end_to_end_latency_ms', result.get('latency_ms'))
            )
            state['last_result'] = {
                'device_id': result.get('device_id'),
                'timestamp': result.get('timestamp'),
                'predicted_label': result.get('predicted_label'),
                'confidence': result.get('confidence'),
            }
            if alert_raised:
                self._alert_count += 1

    def record_adaptation(self, result):
        adaptation = result.get('adaptation') or {}
        with self._lock:
            self._adaptation['request_count'] += 1
            self._adaptation['sample_count'] += int(adaptation.get('sample_count') or 0)
            self._adaptation['updated_labels'] = list(adaptation.get('updated_labels') or [])
            self._adaptation['last_status'] = result.get('status')
            self._adaptation['last_result'] = result

    def snapshot(self):
        with self._lock:
            channels = {}
            for key, state in self._channels.items():
                count = state['request_count']
                channels[key] = {
                    'request_count': count,
                    'avg_preprocess_latency_ms': (
                        state['total_preprocess_latency_ms'] / count if count else None
                    ),
                    'avg_inference_latency_ms': (
                        state['total_inference_latency_ms'] / count if count else None
                    ),
                    'avg_end_to_end_latency_ms': (
                        state['total_end_to_end_latency_ms'] / count if count else None
                    ),
                    'last_result': state['last_result'],
                }
            return {
                'started_at': self._started_at,
                'uptime_seconds': max(0.0, time.time() - self._started_at),
                'channels': channels,
                'alerts_triggered': self._alert_count,
                'adaptation': dict(self._adaptation),
            }
