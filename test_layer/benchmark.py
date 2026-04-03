import argparse
import csv
import io
import json
import os
from pathlib import Path


LATENCY_SEMANTICS = (
    'deployment-layer inference latency from compression summary, '
    'not system-layer end-to-end latency'
)


def _parse_experiment_title(experiment_title):
    if not experiment_title:
        return {}
    import re

    match = re.match(
        (
            r'^(?P<algorithm>[^_]+)_(?P<dataset>[^_]+)_(?P<preprocess>[^_]+)_'
            r'(?P<ways>\d+)w(?P<shots>\d+)s(?:(?P<query_shots>\d+)q)?_'
            r'source(?P<train_domains>[0-9]+)_target(?P<test_domain>\d+)_labels(?P<labels>[0-9,]+)$'
        ),
        experiment_title,
    )
    if not match:
        return {}
    parsed = match.groupdict()
    labels = parsed.get('labels') or ''
    if ',' in labels:
        fault_labels = [int(item) for item in labels.split(',') if item]
    else:
        fault_labels = [int(character) for character in labels]
    return {
        'algorithm': parsed.get('algorithm', '').lower() or None,
        'dataset': parsed.get('dataset'),
        'preprocess': parsed.get('preprocess'),
        'ways': int(parsed['ways']) if parsed.get('ways') else None,
        'shots': int(parsed['shots']) if parsed.get('shots') else None,
        'query_shots': int(parsed['query_shots']) if parsed.get('query_shots') else None,
        'train_domains': [int(character) for character in parsed.get('train_domains', '')],
        'test_domain': int(parsed['test_domain']) if parsed.get('test_domain') else None,
        'fault_labels': fault_labels,
    }


def load_summary(summary_path):
    with open(summary_path, 'r', encoding='utf-8') as file_pointer:
        summary = json.load(file_pointer)
    summary.setdefault('_summary_path', summary_path)
    return summary


def get_deployment_metrics(summary):
    return summary.get('deployment_int8_metrics') or summary.get('deployment_float_metrics') or {}


def build_benchmark_row(summary):
    experiment = summary.get('experiment') or {}
    legacy = _parse_experiment_title(summary.get('experiment_title'))
    profile = summary.get('model_profile') or {}
    size_bytes = summary.get('artifact_sizes_bytes') or {}
    size_mb = summary.get('artifact_sizes_mb') or {}
    deployment_metrics = get_deployment_metrics(summary)
    selected_labels = summary.get('selected_labels') or []

    return {
        'summary_path': summary.get('_summary_path') or summary.get('float_model_path') or '',
        'experiment_title': summary.get('experiment_title'),
        'algorithm': summary.get('algorithm') or experiment.get('algorithm') or legacy.get('algorithm'),
        'dataset': experiment.get('dataset') or legacy.get('dataset'),
        'preprocess': experiment.get('preprocess') or legacy.get('preprocess'),
        'ways': experiment.get('ways') or legacy.get('ways'),
        'shots': experiment.get('shots') or legacy.get('shots'),
        'query_shots': experiment.get('query_shots') or legacy.get('query_shots'),
        'train_domains': ','.join(str(item) for item in (experiment.get('train_domains') or legacy.get('train_domains') or [])),
        'test_domain': experiment.get('test_domain') or legacy.get('test_domain'),
        'fault_labels': ','.join(str(item) for item in (experiment.get('fault_labels') or legacy.get('fault_labels') or [])),
        'selected_labels': ','.join(str(item) for item in selected_labels),
        'deployment_type': summary.get('deployment_type'),
        'deployment_backend': summary.get('deployment_backend'),
        'accuracy': deployment_metrics.get('accuracy'),
        'avg_latency_ms': deployment_metrics.get('avg_latency_ms'),
        'loss': deployment_metrics.get('loss'),
        'providers': ','.join(str(item) for item in deployment_metrics.get('providers') or []),
        'runtime_backend': deployment_metrics.get('runtime_backend') or summary.get('deployment_backend'),
        'float_model_path': summary.get('float_model_path'),
        'int8_model_path': summary.get('int8_model_path'),
        'prototype_path': summary.get('prototype_path'),
        'float_model_size_mb': size_mb.get('float_model'),
        'int8_model_size_mb': size_mb.get('int8_model'),
        'prototype_size_mb': size_mb.get('prototype_bundle'),
        'float_model_size_bytes': size_bytes.get('float_model'),
        'int8_model_size_bytes': size_bytes.get('int8_model'),
        'prototype_size_bytes': size_bytes.get('prototype_bundle'),
        'baseline_params': profile.get('baseline_params'),
        'pruned_params': profile.get('pruned_params'),
        'parameter_reduction_ratio': profile.get('parameter_reduction_ratio'),
        'prune_ratio': profile.get('prune_ratio'),
        'latency_semantics': LATENCY_SEMANTICS,
    }


def check_thresholds(summary, accuracy_threshold=0.95, latency_threshold_ms=100.0):
    deployment_metrics = get_deployment_metrics(summary)
    accuracy_value = deployment_metrics.get('accuracy')
    latency_value = deployment_metrics.get('avg_latency_ms')
    return {
        'accuracy_pass': accuracy_value is not None and accuracy_value >= accuracy_threshold,
        'latency_pass': latency_value is not None and latency_value <= latency_threshold_ms,
        'accuracy': accuracy_value,
        'avg_latency_ms': latency_value,
        'deployment_backend': summary.get('deployment_backend'),
        'deployment_type': summary.get('deployment_type'),
        'latency_semantics': LATENCY_SEMANTICS,
        'benchmark_row': build_benchmark_row(summary),
    }


def export_rows(rows, output_path=None, output_format='json'):
    output_format = output_format.lower()
    if output_format not in {'json', 'csv'}:
        raise ValueError('output_format must be json or csv.')

    if output_format == 'json':
        payload = json.dumps(rows, indent=2, ensure_ascii=False)
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(payload, encoding='utf-8')
        return payload

    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    payload = buffer.getvalue()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(payload, encoding='utf-8', newline='')
    return payload


def main():
    parser = argparse.ArgumentParser(
        description='Read deployment benchmark summaries and export thesis-ready benchmark rows.',
    )
    parser.add_argument('--summary_path', type=str, required=True,
                        help='Path to deploy_layer compression summary JSON.')
    parser.add_argument('--accuracy_threshold', type=float, default=0.95,
                        help='Accuracy target, default=0.95.')
    parser.add_argument('--latency_threshold_ms', type=float, default=100.0,
                        help='Latency target in ms, default=100.0.')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Optional path for exported JSON/CSV results.')
    parser.add_argument('--output_format', type=str, default='json', choices=['json', 'csv'],
                        help='Export format for benchmark row output.')
    parser.add_argument('--rows_only', action='store_true',
                        help='Only export the flattened benchmark row instead of the threshold check payload.')
    args = parser.parse_args()

    if not os.path.exists(args.summary_path):
        raise FileNotFoundError('Summary file not found: {}'.format(args.summary_path))

    summary = load_summary(args.summary_path)
    result = check_thresholds(
        summary,
        accuracy_threshold=args.accuracy_threshold,
        latency_threshold_ms=args.latency_threshold_ms,
    )
    payload = [result['benchmark_row']] if args.rows_only else [result]
    rendered = export_rows(payload, output_path=args.output_path, output_format=args.output_format)
    print(rendered)


if __name__ == '__main__':
    main()
