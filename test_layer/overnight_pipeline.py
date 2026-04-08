import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from test_layer.benchmark import build_benchmark_row, export_rows, load_summary
from test_layer.experiment_runner import (
    build_command,
    export_manifest,
    infer_expected_summary_path_from_config,
)
from test_layer.thesis_config import (
    OVERNIGHT_PRESET_NAME,
    THESIS_DEFAULT_SYSTEM_CHANNEL,
    build_overnight_experiment_records,
    row_matches_thesis_profile,
)
from test_layer.thesis_tables import (
    build_compression_table,
    build_few_shot_table,
    build_model_performance_table,
    build_overnight_markdown_report,
    build_preprocess_model_matrix_table,
    write_table,
)


def _resolve_deploy_command(train_command, algorithm):
    return [
        train_command[0],
        'deploy_layer/deploy.py',
        '--algorithm',
        algorithm,
        *train_command[6:],
    ]


def _run_logged_command(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as log_file:
        log_file.write('\n=== {} ===\n'.format(datetime.now().isoformat(timespec='seconds')))
        log_file.write('$ {}\n'.format(subprocess.list2cmdline(command)))
        log_file.flush()
        completed = subprocess.run(
            command,
            cwd=str(ROOT_DIR),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_file.write('\n[return_code] {}\n'.format(completed.returncode))
        return completed.returncode


def _load_all_rows(summary_paths):
    rows = []
    for summary_path in summary_paths:
        summary = load_summary(str(summary_path))
        row = build_benchmark_row(summary)
        rows.append((summary, row))
    return rows


def _export_tables(output_dir, output_format, summary_paths):
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = _load_all_rows(summary_paths)
    thesis_rows = [(summary, row) for summary, row in all_rows if row_matches_thesis_profile(row)]

    preprocess_matrix = build_preprocess_model_matrix_table(all_rows)
    model_table = build_model_performance_table(thesis_rows, allow_missing=True)
    few_shot_table = build_few_shot_table(thesis_rows, allow_missing=True)
    compression_table = build_compression_table(thesis_rows, allow_missing=True)

    write_table(preprocess_matrix, output_dir / 'table0_preprocess_model_matrix.{}'.format(output_format), output_format)
    write_table(model_table, output_dir / 'table1_model_performance.{}'.format(output_format), output_format)
    write_table(few_shot_table, output_dir / 'table2_few_shot.{}'.format(output_format), output_format)
    write_table(compression_table, output_dir / 'table3_compression.{}'.format(output_format), output_format)

    combined_payload = {
        'preprocess_matrix': preprocess_matrix,
        'locked_profile_table1': model_table,
        'locked_profile_table2': few_shot_table,
        'locked_profile_table3': compression_table,
        'locked_profile_channel': THESIS_DEFAULT_SYSTEM_CHANNEL,
    }
    (output_dir / 'overnight_tables.json').write_text(
        json.dumps(combined_payload, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    (output_dir / 'overnight_tables.md').write_text(
        build_overnight_markdown_report(
            preprocess_matrix,
            model_table,
            few_shot_table,
            compression_table,
        ),
        encoding='utf-8',
    )


def main():
    parser = argparse.ArgumentParser(
        description='Run the full overnight A10 experiment pipeline and export benchmark/table artifacts.',
    )
    parser.add_argument('--preset', type=str, default=OVERNIGHT_PRESET_NAME, choices=[OVERNIGHT_PRESET_NAME])
    parser.add_argument('--max_attempts', type=int, default=2,
                        help='Maximum train attempts per experiment before marking it failed.')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip experiments whose compression summary already exists.')
    parser.add_argument('--manifest_dir', type=str, default='logs/overnight_runs/latest')
    parser.add_argument('--tables_dir', type=str, default='logs/thesis_tables')
    parser.add_argument('--output_format', type=str, default='csv', choices=['json', 'csv'])
    args = parser.parse_args()

    records = []
    benchmark_rows = []
    summary_paths = []
    manifest_dir = ROOT_DIR / args.manifest_dir
    log_dir = manifest_dir / 'logs'
    configs = build_overnight_experiment_records()

    for config in configs:
        command = build_command(config)
        expected_summary_path = infer_expected_summary_path_from_config(config)
        log_path = log_dir / '{}.log'.format(expected_summary_path.parent.name)
        record = {
            'generated_at': datetime.now().isoformat(timespec='seconds'),
            'config': config,
            'command': command,
            'command_string': subprocess.list2cmdline(command),
            'expected_summary_path': str(expected_summary_path),
            'status': 'pending',
            'attempts': 0,
            'train_return_codes': [],
            'deploy_recovery_return_code': None,
            'log_path': str(log_path),
        }

        if args.skip_existing and expected_summary_path.exists():
            summary = load_summary(str(expected_summary_path))
            benchmark_rows.append(build_benchmark_row(summary))
            summary_paths.append(expected_summary_path)
            record['status'] = 'skipped_existing'
            records.append(record)
            continue

        for attempt in range(1, args.max_attempts + 1):
            record['attempts'] = attempt
            return_code = _run_logged_command(command, log_path)
            record['train_return_codes'].append(return_code)
            if return_code == 0 and expected_summary_path.exists():
                break
        else:
            return_code = record['train_return_codes'][-1]

        if not expected_summary_path.exists():
            deploy_command = _resolve_deploy_command(command, config['algorithm'])
            record['deploy_recovery_return_code'] = _run_logged_command(deploy_command, log_path)

        if expected_summary_path.exists():
            summary = load_summary(str(expected_summary_path))
            benchmark_rows.append(build_benchmark_row(summary))
            summary_paths.append(expected_summary_path)
            record['status'] = 'completed'
        else:
            record['status'] = 'failed'
        records.append(record)

    manifest_path = export_manifest(
        {
            'generated_at': datetime.now().isoformat(timespec='seconds'),
            'preset': args.preset,
            'max_attempts': args.max_attempts,
            'skip_existing': args.skip_existing,
            'records': records,
            'benchmark_rows': benchmark_rows,
        },
        manifest_dir,
    )

    export_rows(
        benchmark_rows,
        output_path=str(manifest_dir / 'benchmark_rows.{}'.format(args.output_format)),
        output_format=args.output_format,
    )
    _export_tables(ROOT_DIR / args.tables_dir, args.output_format, summary_paths)

    print(manifest_path)

    failed = [record for record in records if record.get('status') == 'failed']
    if failed:
        raise SystemExit('Overnight pipeline finished with {} failed experiment(s).'.format(len(failed)))


if __name__ == '__main__':
    main()
