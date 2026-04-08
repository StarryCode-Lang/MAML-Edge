import argparse
import json
import sys
from pathlib import Path
from urllib.request import urlopen

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from test_layer.benchmark import build_benchmark_row, build_compression_rows, export_rows, load_summary
from test_layer.thesis_config import (
    THESIS_BASE_PROFILE,
    THESIS_DEFAULT_SYSTEM_CHANNEL,
    THESIS_FEW_SHOT_VALUES,
    THESIS_ALL_PREPROCESSES,
    THESIS_MODEL_COMPARE_ALGORITHMS,
    THESIS_PRIMARY_MODEL,
    row_matches_thesis_profile,
)


def discover_summary_paths(pattern):
    return sorted(ROOT_DIR.glob(pattern))


def load_all_rows(summary_glob):
    rows = []
    for summary_path in discover_summary_paths(summary_glob):
        summary = load_summary(str(summary_path))
        row = build_benchmark_row(summary)
        rows.append((summary, row))
    return rows


def load_matching_rows(summary_glob):
    rows = []
    for summary, row in load_all_rows(summary_glob):
        if row_matches_thesis_profile(row):
            rows.append((summary, row))
    return rows


def _match_row(rows, algorithm, shots):
    for summary, row in rows:
        if str(row.get('algorithm')).lower() != str(algorithm).lower():
            continue
        if int(row.get('shots') or 0) != int(shots):
            continue
        return summary, row
    return None, None


def _require_row(rows, algorithm, shots, allow_missing=False):
    summary, row = _match_row(rows, algorithm, shots)
    if row is not None or allow_missing:
        return summary, row
    raise FileNotFoundError(
        'Missing thesis summary for algorithm={} shots={}.'.format(algorithm, shots)
    )


def build_model_performance_table(rows, allow_missing=False):
    table_rows = []
    for algorithm in THESIS_MODEL_COMPARE_ALGORITHMS:
        _, row = _require_row(rows, algorithm=algorithm, shots=5, allow_missing=allow_missing)
        if row is None:
            continue
        accuracy_value = row.get('pre_prune_accuracy') or row.get('baseline_deployment_accuracy') or row.get('accuracy')
        table_rows.append({
            'model': str(algorithm).upper() if algorithm != 'protonet' else 'ProtoNet',
            'accuracy': accuracy_value,
            'accuracy_percent': _percent_value(accuracy_value),
            'experiment_title': row.get('experiment_title'),
        })
    return table_rows


def build_preprocess_model_matrix_table(rows):
    table_rows = []
    for preprocess in THESIS_ALL_PREPROCESSES:
        for algorithm in THESIS_MODEL_COMPARE_ALGORITHMS:
            for shots in THESIS_FEW_SHOT_VALUES:
                row = None
                for _, candidate_row in rows:
                    if str(candidate_row.get('algorithm')).lower() != str(algorithm).lower():
                        continue
                    if int(candidate_row.get('shots') or 0) != shots:
                        continue
                    if str(candidate_row.get('preprocess')) != preprocess:
                        continue
                    row = candidate_row
                    break
                if row is None:
                    continue
                accuracy_value = row.get('pre_prune_accuracy') or row.get('baseline_deployment_accuracy') or row.get('accuracy')
                table_rows.append({
                    'preprocess': preprocess,
                    'model': str(algorithm).upper() if algorithm != 'protonet' else 'ProtoNet',
                    'shots': shots,
                    'accuracy': accuracy_value,
                    'accuracy_percent': _percent_value(accuracy_value),
                    'latency_ms': _round_or_none(row.get('baseline_avg_latency_ms') or row.get('avg_latency_ms'), 4),
                    'experiment_title': row.get('experiment_title'),
                })
    return table_rows


def build_few_shot_table(rows, allow_missing=False):
    table_rows = []
    for shots in THESIS_FEW_SHOT_VALUES:
        _, row = _require_row(rows, algorithm=THESIS_PRIMARY_MODEL, shots=shots, allow_missing=allow_missing)
        if row is None:
            continue
        accuracy_value = row.get('pre_prune_accuracy') or row.get('baseline_deployment_accuracy') or row.get('accuracy')
        table_rows.append({
            'model': THESIS_PRIMARY_MODEL.upper(),
            'shots': shots,
            'accuracy': accuracy_value,
            'accuracy_percent': _percent_value(accuracy_value),
            'experiment_title': row.get('experiment_title'),
        })
    return table_rows


def build_compression_table(rows, allow_missing=False):
    summary, _ = _require_row(rows, algorithm=THESIS_PRIMARY_MODEL, shots=5, allow_missing=allow_missing)
    if summary is None:
        return []
    mapped = {
        'original': 'Original MAML',
        'pruned': 'Pruned',
        'pruned_int8': 'Pruned + INT8',
    }
    table_rows = []
    for row in build_compression_rows(summary):
        accuracy_value = row.get('accuracy')
        latency_value = row.get('avg_latency_ms')
        table_rows.append({
            'model_variant': mapped.get(row.get('variant'), row.get('variant')),
            'accuracy': accuracy_value,
            'accuracy_percent': _percent_value(accuracy_value),
            'latency_ms': _round_or_none(latency_value, 4),
            'parameter_count': row.get('parameter_count'),
            'model_size_mb': row.get('model_size_mb'),
            'runtime_backend': row.get('runtime_backend'),
        })
    return table_rows


def _load_system_stats_from_url(url):
    with urlopen(url) as response:
        return json.loads(response.read().decode('utf-8'))


def _load_system_stats(path=None, url=None):
    if path:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    if url:
        return _load_system_stats_from_url(url)
    return None


def build_system_performance_table(system_stats, channel):
    if not system_stats:
        return []
    channels = (system_stats.get('channels') or {})
    selected = channels.get(channel) or {}
    request_count = selected.get('request_count')
    return [
        {
            'stage': 'preprocess',
            'channel': channel,
            'request_count': request_count,
            'latency_ms': _round_or_none(selected.get('avg_preprocess_latency_ms'), 4),
        },
        {
            'stage': 'inference',
            'channel': channel,
            'request_count': request_count,
            'latency_ms': _round_or_none(selected.get('avg_inference_latency_ms'), 4),
        },
        {
            'stage': 'end_to_end',
            'channel': channel,
            'request_count': request_count,
            'latency_ms': _round_or_none(selected.get('avg_end_to_end_latency_ms'), 4),
        },
    ]


def write_table(table_rows, output_path, output_format):
    export_rows(table_rows, output_path=str(output_path), output_format=output_format)


def _round_or_none(value, digits=4):
    if value is None:
        return None
    return round(float(value), digits)


def _percent_value(value):
    if value is None:
        return None
    return round(float(value) * 100.0, 2)


def _display_value(value, suffix=''):
    if value is None:
        return '-'
    return '{}{}'.format(value, suffix)


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


def build_markdown_report(model_table, few_shot_table, compression_table, system_table):
    display_model_table = [
        {
            'Model': row.get('model'),
            'Accuracy (%)': _display_value(row.get('accuracy_percent'), '%'),
        }
        for row in model_table
    ]
    display_few_shot_table = [
        {
            'Model': row.get('model'),
            'Shot': row.get('shots'),
            'Accuracy (%)': _display_value(row.get('accuracy_percent'), '%'),
        }
        for row in few_shot_table
    ]
    display_compression_table = [
        {
            'Variant': row.get('model_variant'),
            'Accuracy (%)': _display_value(row.get('accuracy_percent'), '%'),
            'Latency (ms)': _display_value(row.get('latency_ms')),
            'Parameter Count': _display_value(row.get('parameter_count')),
            'Model Size (MB)': _display_value(row.get('model_size_mb')),
        }
        for row in compression_table
    ]
    display_system_table = [
        {
            'Stage': row.get('stage'),
            'Channel': row.get('channel'),
            'Requests': _display_value(row.get('request_count')),
            'Latency (ms)': _display_value(row.get('latency_ms')),
        }
        for row in system_table
    ]

    sections = [
        '# Thesis Tables',
        '',
        '## Locked Profile',
        '',
        _render_markdown_table([THESIS_BASE_PROFILE]),
        '## Table 1 - Model Performance',
        '',
        _render_markdown_table(display_model_table),
        '## Table 2 - Few-Shot Performance',
        '',
        _render_markdown_table(display_few_shot_table),
        '## Table 3 - Compression Impact',
        '',
        _render_markdown_table(display_compression_table),
        '## Table 4 - System Performance',
        '',
        _render_markdown_table(display_system_table),
    ]
    return '\n'.join(sections).strip() + '\n'


def build_overnight_markdown_report(preprocess_table, model_table, few_shot_table, compression_table):
    display_preprocess_table = [
        {
            'Preprocess': row.get('preprocess'),
            'Model': row.get('model'),
            'Shot': row.get('shots'),
            'Accuracy (%)': _display_value(row.get('accuracy_percent'), '%'),
            'Latency (ms)': _display_value(row.get('latency_ms')),
        }
        for row in preprocess_table
    ]
    sections = [
        '# Overnight Run Tables',
        '',
        '## Table 0 - Preprocess x Model Matrix',
        '',
        _render_markdown_table(display_preprocess_table),
        '## Locked Thesis Tables',
        '',
        build_markdown_report(model_table, few_shot_table, compression_table, []).strip(),
    ]
    return '\n'.join(sections).strip() + '\n'


def main():
    parser = argparse.ArgumentParser(
        description='Build the four thesis tables from locked MAML-Edge experiment outputs.',
    )
    parser.add_argument(
        '--summary_glob',
        type=str,
        default='deploy_artifacts/*/compression_summary.json',
        help='Glob pattern for deployment summaries.',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='logs/thesis_tables',
        help='Directory used to export thesis-ready tables.',
    )
    parser.add_argument(
        '--output_format',
        type=str,
        default='csv',
        choices=['json', 'csv'],
        help='Output format for each table file.',
    )
    parser.add_argument(
        '--system_stats_path',
        type=str,
        default=None,
        help='Optional path to a saved /system/stats JSON payload.',
    )
    parser.add_argument(
        '--system_stats_url',
        type=str,
        default=None,
        help='Optional URL for a live /system/stats endpoint.',
    )
    parser.add_argument(
        '--system_channel',
        type=str,
        default=THESIS_DEFAULT_SYSTEM_CHANNEL,
        choices=['direct', 'mqtt'],
        help='Channel used for the system latency table.',
    )
    parser.add_argument(
        '--allow_missing',
        action='store_true',
        help='Skip missing tables instead of failing hard.',
    )
    args = parser.parse_args()

    output_dir = ROOT_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = load_all_rows(args.summary_glob)
    rows = [(summary, row) for summary, row in all_rows if row_matches_thesis_profile(row)]
    preprocess_matrix = build_preprocess_model_matrix_table(all_rows)
    model_table = build_model_performance_table(rows, allow_missing=args.allow_missing)
    few_shot_table = build_few_shot_table(rows, allow_missing=args.allow_missing)
    compression_table = build_compression_table(rows, allow_missing=args.allow_missing)
    system_stats = _load_system_stats(path=args.system_stats_path, url=args.system_stats_url)
    system_table = build_system_performance_table(system_stats, channel=args.system_channel)

    write_table(preprocess_matrix, output_dir / 'table0_preprocess_model_matrix.{}'.format(args.output_format), args.output_format)
    write_table(model_table, output_dir / 'table1_model_performance.{}'.format(args.output_format), args.output_format)
    write_table(few_shot_table, output_dir / 'table2_few_shot.{}'.format(args.output_format), args.output_format)
    write_table(compression_table, output_dir / 'table3_compression.{}'.format(args.output_format), args.output_format)
    if system_table:
        write_table(system_table, output_dir / 'table4_system_performance.{}'.format(args.output_format), args.output_format)

    combined_payload = {
        'locked_profile': THESIS_BASE_PROFILE,
        'table0_preprocess_model_matrix': preprocess_matrix,
        'table1_model_performance': model_table,
        'table2_few_shot': few_shot_table,
        'table3_compression': compression_table,
        'table4_system_performance': system_table,
    }
    (output_dir / 'thesis_tables.json').write_text(
        json.dumps(combined_payload, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    (output_dir / 'thesis_tables.md').write_text(
        build_overnight_markdown_report(preprocess_matrix, model_table, few_shot_table, compression_table)
        + (
            '\n## Table 4 - System Performance\n\n'
            + _render_markdown_table([
                {
                    'Stage': row.get('stage'),
                    'Channel': row.get('channel'),
                    'Requests': _display_value(row.get('request_count')),
                    'Latency (ms)': _display_value(row.get('latency_ms')),
                }
                for row in system_table
            ])
            if system_table else ''
        ),
        encoding='utf-8',
    )
    print(output_dir)


if __name__ == '__main__':
    main()
