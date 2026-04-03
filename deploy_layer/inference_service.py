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


def _as_inference_input(array, input_rank):
    input_array = np.asarray(array, dtype=np.float32)
    if input_array.ndim == 1:
        input_array = input_array[np.newaxis, :]
    if input_array.ndim == input_rank - 1:
        input_array = input_array[np.newaxis, ...]
    return input_array.astype(np.float32, copy=False)


class DeploymentInferenceService:
    def __init__(self, summary_path, runtime_backend=None, prefer_int8=True):
        import onnxruntime as ort

        with open(summary_path, 'r', encoding='utf-8') as file_pointer:
            self.summary = json.load(file_pointer)

        self.summary['_summary_path'] = summary_path
        self.runtime_backend = runtime_backend or self.summary.get('deployment_backend', 'onnxruntime')
        float_model_path = self.summary['float_model_path']
        int8_model_path = self.summary.get('int8_model_path')
        self.model_path = (
            int8_model_path
            if prefer_int8 and int8_model_path and os.path.exists(int8_model_path)
            else float_model_path
        )

        self.providers = resolve_execution_providers(self.runtime_backend, ort)
        self.session = ort.InferenceSession(self.model_path, providers=self.providers)
        self.input_rank = len(self.session.get_inputs()[0].shape)
        self.prototype_path = self.summary.get('prototype_path')
        self.deployment_type = 'encoder_with_prototypes' if self.prototype_path else 'classifier'
        self.fault_labels = _parse_fault_labels(self.summary.get('experiment_title', '')) or []

        self.prototypes = None
        self.prototype_labels = None
        self.prototype_support_counts = {}
        self._default_prototype_support_count = max(
            1,
            int((self.summary.get('experiment') or {}).get('shots') or 1),
        )
        if self.prototype_path and os.path.exists(self.prototype_path):
            prototype_bundle = np.load(self.prototype_path)
            self.prototypes = prototype_bundle['prototypes'].astype(np.float32)
            self.prototype_labels = prototype_bundle['selected_labels'].astype(np.int64).tolist()
            self.prototype_support_counts = {
                int(label): self._default_prototype_support_count
                for label in self.prototype_labels
            }

    def adaptation_supported(self):
        return self.deployment_type == 'encoder_with_prototypes'

    def extract_embeddings(self, model_input):
        if not self.adaptation_supported():
            raise ValueError('Runtime prototype adaptation is only available for encoder deployments.')
        input_array = _as_inference_input(model_input, self.input_rank)
        outputs = self.session.run(None, {'input': input_array})[0]
        return np.asarray(outputs, dtype=np.float32)

    def _predict_classifier(self, outputs):
        probabilities = _softmax(outputs)
        predicted_indices = np.argmax(probabilities, axis=1)
        label_space = (
            self.fault_labels
            if len(self.fault_labels) == probabilities.shape[1]
            else list(range(probabilities.shape[1]))
        )
        predicted_labels = [int(label_space[index]) for index in predicted_indices]
        confidences = [
            float(probabilities[row_index, predicted_indices[row_index]])
            for row_index in range(len(predicted_indices))
        ]
        return predicted_labels, confidences

    def _predict_with_prototypes(self, embeddings):
        if self.prototypes is None or self.prototype_labels is None:
            raise ValueError('Prototype deployment is missing runtime prototype data.')
        distances = ((embeddings[:, np.newaxis, :] - self.prototypes[np.newaxis, :, :]) ** 2).sum(axis=2)
        logits = -distances
        probabilities = _softmax(logits)
        predicted_indices = np.argmax(probabilities, axis=1)
        predicted_labels = [int(self.prototype_labels[index]) for index in predicted_indices]
        confidences = [
            float(probabilities[row_index, predicted_indices[row_index]])
            for row_index in range(len(predicted_indices))
        ]
        return predicted_labels, confidences

    def predict(self, model_input):
        input_array = _as_inference_input(model_input, self.input_rank)

        start_time = time.perf_counter()
        outputs = self.session.run(None, {'input': input_array})[0]
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        if self.deployment_type == 'classifier':
            predicted_labels, confidences = self._predict_classifier(outputs)
        else:
            predicted_labels, confidences = self._predict_with_prototypes(outputs)

        results = []
        for index, predicted_label in enumerate(predicted_labels):
            results.append({
                'predicted_label': predicted_label,
                'confidence': confidences[index],
                'inference_latency_ms': latency_ms / max(1, len(predicted_labels)),
                'runtime_backend': self.runtime_backend,
                'providers': list(self.providers),
                'model_path': self.model_path,
            })
        return results

    def update_prototypes_from_embeddings(self, embeddings, labels, blend_factor=None):
        if not self.adaptation_supported():
            raise ValueError('Runtime prototype adaptation is only available for encoder deployments.')

        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError('Embeddings must be a 2D array with shape [num_samples, embedding_dim].')

        label_list = [int(label) for label in labels]
        if not label_list:
            raise ValueError('At least one support label is required for adaptation.')
        if embeddings.shape[0] != len(label_list):
            raise ValueError('The number of embeddings must match the number of labels.')

        grouped = {}
        for index, label in enumerate(label_list):
            grouped.setdefault(label, []).append(embeddings[index])

        existing = {}
        if self.prototypes is not None and self.prototype_labels is not None:
            existing = {
                int(label): self.prototypes[index].astype(np.float32)
                for index, label in enumerate(self.prototype_labels)
            }

        updated_labels = []
        new_labels_added = []
        support_count_by_label = {}
        for label, label_embeddings in grouped.items():
            label_matrix = np.asarray(label_embeddings, dtype=np.float32)
            label_mean = label_matrix.mean(axis=0)
            support_count = int(label_matrix.shape[0])
            support_count_by_label[label] = support_count
            if label in existing:
                previous_count = int(self.prototype_support_counts.get(label, self._default_prototype_support_count))
                if blend_factor is None:
                    total_count = previous_count + support_count
                    existing[label] = (
                        (existing[label] * previous_count) + (label_mean * support_count)
                    ) / float(total_count)
                else:
                    alpha = min(1.0, max(0.0, float(blend_factor)))
                    existing[label] = (existing[label] * (1.0 - alpha)) + (label_mean * alpha)
                    total_count = previous_count + support_count
                self.prototype_support_counts[label] = total_count
            else:
                existing[label] = label_mean
                self.prototype_support_counts[label] = support_count
                new_labels_added.append(label)
            updated_labels.append(label)

        sorted_labels = sorted(existing.keys())
        self.prototype_labels = [int(label) for label in sorted_labels]
        self.prototypes = np.stack([existing[label] for label in sorted_labels]).astype(np.float32)

        return {
            'status': 'ok',
            'strategy': 'prototype_update',
            'sample_count': int(len(label_list)),
            'updated_labels': updated_labels,
            'new_labels_added': new_labels_added,
            'support_count_by_label': support_count_by_label,
            'prototype_count': len(self.prototype_labels),
            'prototype_labels': list(self.prototype_labels),
            'blend_factor': blend_factor,
        }

    def model_info(self):
        return {
            'experiment_title': self.summary.get('experiment_title'),
            'algorithm': self.summary.get('algorithm'),
            'deployment_type': self.deployment_type,
            'runtime_backend': self.runtime_backend,
            'providers': list(self.providers),
            'model_path': self.model_path,
            'prototype_path': self.prototype_path,
            'summary_path': self.summary.get('_summary_path'),
            'deployment_backend': self.summary.get('deployment_backend', self.runtime_backend),
            'selected_labels': self.summary.get('selected_labels') or [],
            'adaptation_supported': self.adaptation_supported(),
            'prototype_labels': list(self.prototype_labels or []),
            'prototype_count': len(self.prototype_labels or []),
            'prototype_support_counts': dict(self.prototype_support_counts),
        }
