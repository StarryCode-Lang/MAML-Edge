import io
import os
import time

import numpy as np
import pywt
from PIL import Image
from scipy.signal import stft

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from deploy_layer.inference_service import DeploymentInferenceService


def _normalize_image_array(image):
    image = image.astype(np.float32) / 255.0
    image = image.transpose(2, 0, 1)
    return (image - 0.5) / 0.5


def _render_stft_image(signal, image_size, window_size, overlap, dataset_name):
    overlap_samples = int(window_size * overlap)
    frequency, timeline, magnitude = stft(signal, nperseg=window_size, noverlap=overlap_samples)
    if dataset_name == 'HST':
        magnitude = np.log10(np.abs(magnitude) + 1e-10)
    else:
        magnitude = np.abs(magnitude)

    figure = plt.figure(figsize=(image_size / 100.0, image_size / 100.0), dpi=100)
    plt.pcolormesh(timeline, frequency, magnitude, shading='gouraud')
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    plt.close(figure)
    buffer.seek(0)
    image = Image.open(buffer).convert('RGB').resize((image_size, image_size))
    return _normalize_image_array(np.asarray(image))


def _render_wt_image(signal, image_size, dataset_name):
    sampling_length = len(signal)
    if dataset_name == 'CWRU':
        sampling_period = 1.0 / 12000
        total_scale = 128
        wavelet = 'cmor100-1'
    else:
        sampling_period = 4e-6
        total_scale = 16
        wavelet = 'morl'

    fc = pywt.central_frequency(wavelet)
    cparam = 2 * fc * total_scale
    scales = cparam / np.arange(total_scale, 0, -1)
    coefficients, frequencies = pywt.cwt(signal, scales, wavelet, sampling_period)
    amplitude = np.abs(coefficients)
    if dataset_name == 'HST':
        amplitude = np.log10(amplitude + 1e-4)

    timeline = np.linspace(0, sampling_period, sampling_length, endpoint=False)
    figure = plt.figure(figsize=(image_size / 100.0, image_size / 100.0), dpi=100)
    plt.contourf(timeline, frequencies, amplitude, cmap='jet')
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    plt.close(figure)
    buffer.seek(0)
    image = Image.open(buffer).convert('RGB').resize((image_size, image_size))
    return _normalize_image_array(np.asarray(image))


def _transform_signal(signal, settings):
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    if signal.shape[0] < settings.time_steps:
        padded = np.zeros(settings.time_steps, dtype=np.float32)
        padded[:signal.shape[0]] = signal
        signal = padded
    else:
        signal = signal[:settings.time_steps]

    if settings.preprocess == 'FFT':
        spectrum = np.fft.fft(signal)
        magnitude = np.abs(spectrum) / len(signal)
        return magnitude[:len(signal) // 2][np.newaxis, :].astype(np.float32)
    if settings.preprocess == 'STFT':
        return _render_stft_image(
            signal=signal,
            image_size=settings.image_size,
            window_size=settings.stft_window_size,
            overlap=settings.stft_overlap,
            dataset_name=settings.dataset_name,
        ).astype(np.float32)
    if settings.preprocess == 'WT':
        return _render_wt_image(
            signal=signal,
            image_size=settings.image_size,
            dataset_name=settings.dataset_name,
        ).astype(np.float32)
    raise ValueError('Unsupported preprocess type: {}'.format(settings.preprocess))


class RealTimePredictor:
    def __init__(self, settings):
        if settings.model_summary_path is None or not os.path.exists(settings.model_summary_path):
            raise FileNotFoundError('Compression summary not found: {}'.format(settings.model_summary_path))
        self.settings = settings
        self.service = DeploymentInferenceService(
            summary_path=settings.model_summary_path,
            runtime_backend=settings.runtime_backend,
            prefer_int8=settings.prefer_int8,
        )
        experiment_title = self.service.summary.get('experiment_title', '')
        self.service.summary['_summary_path'] = settings.model_summary_path
        if '_FFT_' in experiment_title:
            self.settings.preprocess = 'FFT'
        elif '_STFT_' in experiment_title:
            self.settings.preprocess = 'STFT'
        elif '_WT_' in experiment_title:
            self.settings.preprocess = 'WT'
        if 'CWRU' in experiment_title:
            self.settings.dataset_name = 'CWRU'
        elif 'HST' in experiment_title:
            self.settings.dataset_name = 'HST'

    def _prepare_model_input(self, signal):
        return _transform_signal(signal, self.settings)

    def predict_payload(self, payload):
        start_time = time.perf_counter()
        signal = payload.get('raw_signal') or payload.get('signal')
        if signal is None:
            raise ValueError('Payload must contain "raw_signal" or "signal".')
        preprocess_start = time.perf_counter()
        model_input = self._prepare_model_input(signal)
        preprocess_latency_ms = (time.perf_counter() - preprocess_start) * 1000.0
        prediction = self.service.predict(model_input)[0]
        end_to_end_latency_ms = (time.perf_counter() - start_time) * 1000.0
        result = {
            'device_id': payload.get('device_id', 'unknown-device'),
            'timestamp': payload.get('timestamp', int(time.time())),
            'predicted_label': prediction['predicted_label'],
            'confidence': prediction['confidence'],
            'latency_ms': end_to_end_latency_ms,
            'preprocess_latency_ms': preprocess_latency_ms,
            'inference_latency_ms': prediction['inference_latency_ms'],
            'end_to_end_latency_ms': end_to_end_latency_ms,
            'runtime_backend': prediction['runtime_backend'],
            'providers': prediction['providers'],
            'model_path': prediction['model_path'],
            'model_summary_path': self.settings.model_summary_path,
            'experiment_title': self.service.summary.get('experiment_title'),
            'deployment_backend': self.service.summary.get('deployment_backend', prediction['runtime_backend']),
            'temperature': payload.get('temperature'),
            'event_triggered': payload.get('event_triggered', False),
            'feature_summary': payload.get('feature_summary', {}),
            'metadata': payload.get('metadata', {}),
        }
        return result

    def adapt_payload(self, payload):
        if not self.service.adaptation_supported():
            return {
                'status': 'unsupported',
                'message': 'Runtime adaptation is only available for encoder-with-prototypes deployments.',
                'model_info': self.model_info(),
            }

        support_features = payload.get('support_features') or []
        blend_factor = payload.get('blend_factor')
        if support_features:
            labels = []
            embeddings = []
            for feature in support_features:
                if 'label' not in feature:
                    raise ValueError('Each support feature requires a label.')
                embedding = feature.get('embedding') or feature.get('feature') or feature.get('vector')
                if embedding is None:
                    raise ValueError('Each support feature requires an embedding/vector payload.')
                labels.append(int(feature['label']))
                embeddings.append(np.asarray(embedding, dtype=np.float32))
            embeddings = np.asarray(embeddings, dtype=np.float32)
        else:
            support_samples = payload.get('support_samples') or []
            if not support_samples and payload.get('label') is not None:
                support_samples = [payload]
            if not support_samples:
                raise ValueError('support_samples or support_features is required for adaptation.')

            labels = []
            model_inputs = []
            for sample in support_samples:
                if 'label' not in sample:
                    raise ValueError('Each support sample requires a label.')
                signal = sample.get('raw_signal') or sample.get('signal')
                if signal is None:
                    raise ValueError('Each support sample requires raw_signal or signal.')
                labels.append(int(sample['label']))
                model_inputs.append(self._prepare_model_input(signal))

            embeddings = self.service.extract_embeddings(np.stack(model_inputs, axis=0))

        adaptation = self.service.update_prototypes_from_embeddings(
            embeddings=embeddings,
            labels=labels,
            blend_factor=blend_factor,
        )
        return {
            'status': 'ok',
            'message': 'Runtime prototypes updated.',
            'adaptation': adaptation,
            'model_info': self.model_info(),
        }

    def model_info(self):
        info = self.service.model_info()
        info['summary_path'] = self.settings.model_summary_path
        return info
