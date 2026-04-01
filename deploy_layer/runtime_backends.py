import logging
import os
import time

import numpy as np
import torch


BACKEND_PROVIDER_CHAINS = {
    'onnxruntime': ['CUDAExecutionProvider', 'CPUExecutionProvider'],
    'tensorrt': ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'],
    'openvino': ['OpenVINOExecutionProvider', 'CPUExecutionProvider'],
}


class NumpyCalibrationDataReader:
    def __init__(self, batches):
        self._batches = iter(batches)

    def get_next(self):
        return next(self._batches, None)


def resolve_execution_providers(runtime_backend, ort_module):
    if runtime_backend not in BACKEND_PROVIDER_CHAINS:
        raise ValueError('Unsupported runtime backend: {}'.format(runtime_backend))
    available_providers = ort_module.get_available_providers()
    preferred_chain = BACKEND_PROVIDER_CHAINS[runtime_backend]
    selected_providers = [provider for provider in preferred_chain if provider in available_providers]
    if not selected_providers:
        raise RuntimeError(
            'Runtime backend "{}" is unavailable. Available providers: {}'.format(
                runtime_backend,
                ', '.join(available_providers) if available_providers else 'none',
            )
        )
    return selected_providers


def collect_calibration_batches(support_data, query_data, calibration_size):
    combined = torch.cat([support_data, query_data], dim=0)
    limited = combined[:max(1, calibration_size)]
    return [{'input': limited[index:index + 1].cpu().numpy()} for index in range(limited.size(0))]


def quantize_and_evaluate_onnx(args, deployment_bundle, float_model_path, artifact_dir, experiment_title):
    try:
        from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    except ImportError as exc:
        warning = 'onnxruntime quantization is unavailable: {}'.format(exc)
        logging.warning(warning)
        return None, None, warning

    calibration_input = collect_calibration_batches(
        support_data=deployment_bundle['support_data'],
        query_data=deployment_bundle['query_data'],
        calibration_size=args.calibration_size,
    )
    quant_model_path = os.path.join(artifact_dir, '{}_int8.onnx'.format(experiment_title))
    quantize_static(
        model_input=float_model_path,
        model_output=quant_model_path,
        calibration_data_reader=NumpyCalibrationDataReader(calibration_input),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
    )
    metrics = evaluate_onnx_bundle(
        onnx_path=quant_model_path,
        deployment_bundle=deployment_bundle,
        runtime_backend=getattr(args, 'runtime_backend', 'onnxruntime'),
    )
    return quant_model_path, metrics, None


def evaluate_onnx_bundle(onnx_path, deployment_bundle, runtime_backend='onnxruntime'):
    import onnxruntime as ort

    providers = resolve_execution_providers(runtime_backend, ort)
    session = ort.InferenceSession(onnx_path, providers=providers)
    query_array = deployment_bundle['query_data'].cpu().numpy()
    query_labels = deployment_bundle['query_labels'].cpu().numpy()

    outputs = []
    total_time = 0.0
    for sample in query_array:
        feed_dict = {'input': sample[np.newaxis, ...]}
        start_time = time.perf_counter()
        session_output = session.run(None, feed_dict)[0]
        total_time += (time.perf_counter() - start_time)
        outputs.append(session_output[0])

    outputs = np.asarray(outputs)
    if deployment_bundle['deployment_type'] == 'classifier':
        predicted_labels = outputs.argmax(axis=1)
        loss_value = softmax_cross_entropy(outputs, query_labels)
    else:
        prototypes = deployment_bundle['prototypes'].cpu().numpy()
        distances = ((outputs[:, np.newaxis, :] - prototypes[np.newaxis, :, :]) ** 2).sum(axis=2)
        logits = -distances
        predicted_labels = logits.argmax(axis=1)
        loss_value = softmax_cross_entropy(logits, query_labels)

    return {
        'loss': float(loss_value),
        'accuracy': float((predicted_labels == query_labels).mean()),
        'avg_latency_ms': float((total_time / max(1, len(query_array))) * 1000.0),
        'providers': providers,
        'runtime_backend': runtime_backend,
    }


def softmax_cross_entropy(logits, labels):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    row_indices = np.arange(labels.shape[0])
    return -np.log(np.clip(probabilities[row_indices, labels], 1e-12, 1.0)).mean()
