import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from test_layer.benchmark import build_benchmark_row, build_compression_ablation_rows, export_rows, load_summary
from test_layer.thesis_config import THESIS_ALL_PREPROCESSES, THESIS_FEW_SHOT_VALUES, THESIS_MODEL_COMPARE_ALGORITHMS


def discover_summary_paths(patterns):
    seen = set()
    paths = []
    for pattern in patterns:
        for summary_path in sorted(ROOT_DIR.glob(pattern)):
            resolved = str(summary_path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(summary_path)
    return paths


def load_rows(patterns):
    loaded = []
    for summary_path in discover_summary_paths(patterns):
        summary = load_summary(str(summary_path))
        row = build_benchmark_row(summary)
        loaded.append((summary, row))
    return loaded


def _normalize_domains(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [int(item) for item in value.split(',') if item]
    return [int(item) for item in value]


def _safe_float(value):
    if value is None:
        return None
    return float(value)


def _safe_int(value):
    if value is None or value == '':
        return None
    return int(value)


def _mean_std(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return None, None, 0
    if len(values) == 1:
        return values[0], 0.0, 1
    return statistics.mean(values), statistics.stdev(values), len(values)


def _format_mean_std(mean_value, std_value, scale=1.0, digits=2):
    if mean_value is None:
        return '-'
    scaled_mean = round(mean_value * scale, digits)
    scaled_std = round((std_value or 0.0) * scale, digits)
    return '{} +- {}'.format(scaled_mean, scaled_std)


def _baseline_accuracy(row):
    return row.get('baseline_deployment_accuracy') if row.get('baseline_deployment_accuracy') is not None else row.get('accuracy')


def _baseline_latency(row):
    return row.get('baseline_avg_latency_ms') if row.get('baseline_avg_latency_ms') is not None else row.get('avg_latency_ms')


def _matches_main_split(row):
    return _normalize_domains(row.get('train_domains')) == [0, 1, 2] and _safe_int(row.get('test_domain')) == 3


def build_seed_matrix_table(rows):
    table_rows = []
    for preprocess in THESIS_ALL_PREPROCESSES:
        for algorithm in THESIS_MODEL_COMPARE_ALGORITHMS:
            for shots in THESIS_FEW_SHOT_VALUES:
                matched = [
                    row for _, row in rows
                    if _matches_main_split(row)
                    and str(row.get('preprocess')) == preprocess
                    and str(row.get('algorithm')).lower() == str(algorithm).lower()
                    and _safe_int(row.get('shots')) == shots
                ]
                if not matched:
                    continue
                accuracy_mean, accuracy_std, sample_count = _mean_std([_baseline_accuracy(row) for row in matched])
                latency_mean, latency_std, _ = _mean_std([_baseline_latency(row) for row in matched])
                table_rows.append({
                    'preprocess': preprocess,
                    'model': str(algorithm).upper() if algorithm != 'protonet' else 'ProtoNet',
                    'shots': shots,
                    'sample_count': sample_count,
                    'metric_protocol': 'deployment_baseline_mean_std',
                    'accuracy_mean': round(accuracy_mean, 6) if accuracy_mean is not None else None,
                    'accuracy_std': round(accuracy_std, 6) if accuracy_std is not None else None,
                    'accuracy_mean_percent': round(accuracy_mean * 100.0, 2) if accuracy_mean is not None else None,
                    'accuracy_std_percent': round(accuracy_std * 100.0, 2) if accuracy_std is not None else None,
                    'accuracy_mean_std': _format_mean_std(accuracy_mean, accuracy_std, scale=100.0, digits=2),
                    'latency_mean_ms': round(latency_mean, 4) if latency_mean is not None else None,
                    'latency_std_ms': round(latency_std, 4) if latency_std is not None else None,
                    'latency_mean_std_ms': _format_mean_std(latency_mean, latency_std, scale=1.0, digits=4),
                })
    return table_rows


def build_model_performance_mean_std(rows):
    seed_matrix = build_seed_matrix_table(rows)
    selected = []
    for row in seed_matrix:
        if row.get('preprocess') == 'STFT' and _safe_int(row.get('shots')) == 5:
            selected.append({
                'model': row.get('model'),
                'sample_count': row.get('sample_count'),
                'metric_protocol': row.get('metric_protocol'),
                'accuracy_mean_percent': row.get('accuracy_mean_percent'),
                'accuracy_std_percent': row.get('accuracy_std_percent'),
                'accuracy_mean_std': row.get('accuracy_mean_std'),
            })
    return selected


def build_few_shot_mean_std(rows):
    seed_matrix = build_seed_matrix_table(rows)
    selected = []
    for row in seed_matrix:
        if row.get('preprocess') == 'STFT' and row.get('model') == 'MAML':
            selected.append({
                'model': 'MAML',
                'shots': row.get('shots'),
                'sample_count': row.get('sample_count'),
                'metric_protocol': row.get('metric_protocol'),
                'accuracy_mean_percent': row.get('accuracy_mean_percent'),
                'accuracy_std_percent': row.get('accuracy_std_percent'),
                'accuracy_mean_std': row.get('accuracy_mean_std'),
            })
    selected.sort(key=lambda row: row.get('shots') or 0)
    return selected


def build_domain_robustness_table(rows):
    supported_splits = {
        ((0, 1, 2), 3),
        ((0, 1, 3), 2),
        ((0, 2, 3), 1),
        ((1, 2, 3), 0),
    }
    table_rows = []
    for _, row in rows:
        train_domains = tuple(_normalize_domains(row.get('train_domains')))
        test_domain = _safe_int(row.get('test_domain'))
        if (train_domains, test_domain) not in supported_splits:
            continue
        accuracy_value = _baseline_accuracy(row)
        latency_value = _baseline_latency(row)
        table_rows.append({
            'train_domains': ','.join(str(item) for item in train_domains),
            'test_domain': test_domain,
            'preprocess': row.get('preprocess'),
            'model': str(row.get('algorithm')).upper() if row.get('algorithm') != 'protonet' else 'ProtoNet',
            'shots': _safe_int(row.get('shots')),
            'seed': row.get('seed'),
            'metric_protocol': 'deployment_baseline',
            'accuracy': accuracy_value,
            'accuracy_percent': round(accuracy_value * 100.0, 2) if accuracy_value is not None else None,
            'latency_ms': round(latency_value, 4) if latency_value is not None else None,
            'experiment_title': row.get('experiment_title'),
        })
    table_rows.sort(
        key=lambda row: (
            row.get('train_domains'),
            row.get('test_domain'),
            row.get('preprocess'),
            row.get('model'),
            row.get('shots'),
        )
    )
    return table_rows


def build_compression_ablation_table(rows):
    table_rows = []
    for summary, row in rows:
        preprocess = row.get('preprocess')
        algorithm = str(row.get('algorithm')).lower()
        shots = _safe_int(row.get('shots'))
        if (preprocess, algorithm, shots) not in {('STFT', 'maml', 5), ('FFT', 'protonet', 5)}:
            continue
        profile_name = '{} + {} + {}-shot'.format(
            preprocess,
            'ProtoNet' if algorithm == 'protonet' else algorithm.upper(),
            shots,
        )
        for ablation_row in build_compression_ablation_rows(summary):
            table_rows.append({
                'profile': profile_name,
                'variant': ablation_row.get('variant'),
                'metric_protocol': ablation_row.get('metric_protocol'),
                'accuracy': ablation_row.get('accuracy'),
                'accuracy_percent': round(float(ablation_row.get('accuracy')) * 100.0, 2),
                'latency_ms': round(float(ablation_row.get('avg_latency_ms')), 4) if ablation_row.get('avg_latency_ms') is not None else None,
                'parameter_count': ablation_row.get('parameter_count'),
                'model_size_mb': ablation_row.get('model_size_mb'),
                'runtime_backend': ablation_row.get('runtime_backend'),
            })
    table_rows.sort(key=lambda row: (row.get('profile'), row.get('variant')))
    return table_rows


def build_system_performance_table(system_benchmark_payload):
    if not system_benchmark_payload:
        return []
    table_rows = []
    for channel in system_benchmark_payload.get('channels') or []:
        channel_name = channel.get('channel')
        request_count = channel.get('request_count')
        table_rows.extend([
            {
                'channel': channel_name,
                'stage': 'preprocess',
                'request_count': request_count,
                'latency_ms': round(float(channel.get('avg_preprocess_latency_ms')), 4) if channel.get('avg_preprocess_latency_ms') is not None else None,
            },
            {
                'channel': channel_name,
                'stage': 'inference',
                'request_count': request_count,
                'latency_ms': round(float(channel.get('avg_inference_latency_ms')), 4) if channel.get('avg_inference_latency_ms') is not None else None,
            },
            {
                'channel': channel_name,
                'stage': 'end_to_end',
                'request_count': request_count,
                'latency_ms': round(float(channel.get('avg_end_to_end_latency_ms')), 4) if channel.get('avg_end_to_end_latency_ms') is not None else None,
            },
        ])
    return table_rows


def _render_markdown_table(rows):
    if not rows:
        return 'No data available.\n'
    headers = list(rows[0].keys())
    lines = [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |',
    ]
    for row in rows:
        lines.append('| ' + ' | '.join(str(row.get(header, '')) for header in headers) + ' |')
    return '\n'.join(lines) + '\n'


def build_markdown_report(table0, table1, table2, domain_table, compression_table, system_table):
    sections = [
        '# Paper Balanced Tables',
        '',
        '## Table 0 - Seed Stability Matrix',
        '',
        _render_markdown_table(table0),
        '## Table 1 - Model Performance Mean +- Std',
        '',
        _render_markdown_table(table1),
        '## Table 2 - Few-Shot Mean +- Std',
        '',
        _render_markdown_table(table2),
        '## Table 3 - Domain Robustness',
        '',
        _render_markdown_table(domain_table),
        '## Table 4 - Compression Ablation',
        '',
        _render_markdown_table(compression_table),
        '## Table 5 - System Performance',
        '',
        _render_markdown_table(system_table),
    ]
    return '\n'.join(sections).strip() + '\n'


def main():
    parser = argparse.ArgumentParser(description='Build paper-balanced thesis tables from controlled + extension runs.')
    parser.add_argument('--base_glob', type=str, default='deploy_artifacts/*/compression_summary.json',
                        help='Glob for the original 27 controlled summaries.')
    parser.add_argument('--seed_glob', type=str, default='deploy_artifacts/paper_balanced/seed/*/*/compression_summary.json',
                        help='Glob for seed extension summaries.')
    parser.add_argument('--domain_glob', type=str, default='deploy_artifacts/paper_balanced/domain/*/*/compression_summary.json',
                        help='Glob for domain extension summaries.')
    parser.add_argument('--ablation_glob', type=str, default='deploy_artifacts/paper_balanced/ablation/*/compression_summary.json',
                        help='Glob for compression ablation summaries.')
    parser.add_argument('--system_benchmark_path', type=str, default='logs/thesis_tables/paper_balanced/system_benchmark.json',
                        help='Path to the saved system benchmark payload.')
    parser.add_argument('--output_dir', type=str, default='logs/thesis_tables/paper_balanced',
                        help='Output directory for paper-balanced tables.')
    parser.add_argument('--output_format', type=str, default='csv', choices=['json', 'csv'],
                        help='Tabular export format.')
    args = parser.parse_args()

    output_dir = ROOT_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_rows = load_rows([args.base_glob, args.seed_glob])
    domain_rows = load_rows([args.base_glob, args.domain_glob])
    ablation_rows = load_rows([args.ablation_glob])

    table0 = build_seed_matrix_table(seed_rows)
    table1 = build_model_performance_mean_std(seed_rows)
    table2 = build_few_shot_mean_std(seed_rows)
    domain_table = build_domain_robustness_table(domain_rows)
    compression_table = build_compression_ablation_table(ablation_rows)

    system_benchmark_path = ROOT_DIR / args.system_benchmark_path
    system_payload = None
    if system_benchmark_path.exists():
        system_payload = json.loads(system_benchmark_path.read_text(encoding='utf-8'))
    system_table = build_system_performance_table(system_payload)

    export_rows(table0, output_path=str(output_dir / 'table0_preprocess_model_matrix_mean_std.{}'.format(args.output_format)), output_format=args.output_format)
    export_rows(table1, output_path=str(output_dir / 'table1_model_performance_mean_std.{}'.format(args.output_format)), output_format=args.output_format)
    export_rows(table2, output_path=str(output_dir / 'table2_few_shot_mean_std.{}'.format(args.output_format)), output_format=args.output_format)
    export_rows(domain_table, output_path=str(output_dir / 'table3_domain_robustness.{}'.format(args.output_format)), output_format=args.output_format)
    export_rows(compression_table, output_path=str(output_dir / 'table4_compression_ablation.{}'.format(args.output_format)), output_format=args.output_format)
    export_rows(system_table, output_path=str(output_dir / 'table5_system_performance.{}'.format(args.output_format)), output_format=args.output_format)

    combined_payload = {
        'table0_preprocess_model_matrix_mean_std': table0,
        'table1_model_performance_mean_std': table1,
        'table2_few_shot_mean_std': table2,
        'table3_domain_robustness': domain_table,
        'table4_compression_ablation': compression_table,
        'table5_system_performance': system_table,
    }
    (output_dir / 'paper_balanced_tables.json').write_text(
        json.dumps(combined_payload, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    (output_dir / 'paper_balanced_report.md').write_text(
        build_markdown_report(table0, table1, table2, domain_table, compression_table, system_table),
        encoding='utf-8',
    )
    print(output_dir)


if __name__ == '__main__':
    main()
