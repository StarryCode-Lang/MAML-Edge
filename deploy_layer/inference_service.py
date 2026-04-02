import json
import os
import re
import time

import numpy as np

from deploy_layer.runtime_backends import resolve_execution_providers


def _softmax(logits):
    stabilized = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(stabilized)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def _parse_fault_labels(experiment_title):
    match = re.search(r'labels([0-9,]+)', experiment_title)
    if not match:
        return None
    raw_value = match.group(1)
    if ',' in raw_value:
        return [int(item) for item in raw_value.split(',') if item]
    return [int(character) for character in raw_value]


class DeploymentInferenceService:
    def __init__(self, summary_path, runtime_backend=None, prefer_int8=True):
        import onnxruntime as ort

        with open(summary_path, 'r', encoding='utf-8') as file_pointer:
            self.summary = json.load(file_pointer)

        self.runtime_backend = runtime_backend or self.summary.get('deployment_backend', 'onnxruntime')
        float_model_path = self.summary['float_model_path']
        int8_model_path = self.summary.get('int8_model_path')
        self.model_path = int8_model_path if prefer_int8 and int8_model_path and os.path.exists(int8_model_path) else float_model_path

        self.providers = resolve_execution_providers(self.runtime_backend, ort)
        self.session = ort.InferenceSession(self.model_path, providers=self.providers)
        self.input_rank = len(self.session.get_inputs()[0].shape)
        self.prototype_path = self.summary.get('prototype_path')
        self.deployment_type = 'encoder_with_prototypes' if self.prototype_path else 'classifier'
        self.fault_labels = _parse_fault_labels(self.summary.get('experiment_title', '')) or []

        self.prototypes = None
        self.prototype_labels = None
        if self.prototype_path and os.path.exists(self.prototype_path):
            prototype_bundle = np.load(self.prototype_path)
            self.prototypes = prototype_bundle['prototypes'].astype(np.float32)
            self.prototype_labels = prototype_bundle['selected_labels'].astype(np.int64).tolist()

    def predict(self, model_input):
        input_array = np.asarray(model_input, dtype=np.float32)
        if input_array.ndim == 1:
            input_array = input_array[np.newaxis, :]
        if input_array.ndim == self.input_rank - 1:
            input_array = input_array[np.newaxis, ...]

        start_time = time.perf_counter()
        outputs = self.session.run(None, {'input': input_array})[0]
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        if self.deployment_type == 'classifier':
            probabilities = _softmax(outputs)
            predicted_indices = np.argmax(probabilities, axis=1)
            label_space = self.fault_labels if len(self.fault_labels) == probabilities.shape[1] else list(range(probabilities.shape[1]))
            predicted_labels = [int(label_space[index]) for index in predicted_indices]
            confidences = [float(probabilities[row_index, predicted_indices[row_index]]) for row_index in range(len(predicted_indices))]
        else:
            distances = ((outputs[:, np.newaxis, :] - self.prototypes[np.newaxis, :, :]) ** 2).sum(axis=2)
            logits = -distances
            probabilities = _softmax(logits)
            predicted_indices = np.argmax(probabilities, axis=1)
            label_space = self.prototype_labels if self.prototype_labels is not None else list(range(probabilities.shape[1]))
            predicted_labels = [int(label_space[index]) for index in predicted_indices]
            confidences = [float(probabilities[row_index, predicted_indices[row_index]]) for row_index in range(len(predicted_indices))]

        results = []
        for index, predicted_label in enumerate(predicted_labels):
            results.append({
                'predicted_label': predicted_label,
                'confidence': confidences[index],
                'latency_ms': latency_ms / max(1, len(predicted_labels)),
                'runtime_backend': self.runtime_backend,
                'providers': list(self.providers),
                'model_path': self.model_path,
            })
        return results

    def model_info(self):
        return {
            'experiment_title': self.summary.get('experiment_title'),
            'algorithm': self.summary.get('algorithm'),
            'deployment_type': self.deployment_type,
            'runtime_backend': self.runtime_backend,
            'providers': list(self.providers),
            'model_path': self.model_path,
            'prototype_path': self.prototype_path,
        }
