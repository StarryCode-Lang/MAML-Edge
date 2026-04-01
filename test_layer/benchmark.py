import argparse
import json
import os


def load_summary(summary_path):
    with open(summary_path, 'r', encoding='utf-8') as file_pointer:
        return json.load(file_pointer)


def check_thresholds(summary, accuracy_threshold=0.95, latency_threshold_ms=100.0):
    deployment_metrics = summary.get('deployment_int8_metrics') or summary.get('deployment_float_metrics') or {}
    accuracy_value = deployment_metrics.get('accuracy')
    latency_value = deployment_metrics.get('avg_latency_ms')
    return {
        'accuracy_pass': accuracy_value is not None and accuracy_value >= accuracy_threshold,
        'latency_pass': latency_value is not None and latency_value <= latency_threshold_ms,
        'accuracy': accuracy_value,
        'avg_latency_ms': latency_value,
    }


def main():
    parser = argparse.ArgumentParser(description='Read deployment benchmark summaries and check target thresholds.')
    parser.add_argument('--summary_path', type=str, required=True,
                        help='Path to deploy_layer compression summary JSON.')
    parser.add_argument('--accuracy_threshold', type=float, default=0.95,
                        help='Accuracy target, default=0.95.')
    parser.add_argument('--latency_threshold_ms', type=float, default=100.0,
                        help='Latency target in ms, default=100.0.')
    args = parser.parse_args()

    if not os.path.exists(args.summary_path):
        raise FileNotFoundError('Summary file not found: {}'.format(args.summary_path))

    result = check_thresholds(
        load_summary(args.summary_path),
        accuracy_threshold=args.accuracy_threshold,
        latency_threshold_ms=args.latency_threshold_ms,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
