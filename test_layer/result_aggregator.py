import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from test_layer.benchmark import build_benchmark_row, export_rows, load_summary


def discover_summary_paths(pattern):
    return sorted(ROOT_DIR.glob(pattern))


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate deployment summaries into thesis-ready JSON/CSV tables.',
    )
    parser.add_argument(
        '--summary_glob',
        type=str,
        default='deploy_artifacts/*/compression_summary.json',
        help='Glob pattern, relative to repository root, used to find summaries.',
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='logs/thesis_benchmark_rows.json',
        help='Path for the aggregated output file.',
    )
    parser.add_argument(
        '--output_format',
        type=str,
        default='json',
        choices=['json', 'csv'],
        help='Output format for the aggregated result.',
    )
    parser.add_argument(
        '--sort_by',
        type=str,
        default='experiment_title',
        help='Field used to sort rows before export.',
    )
    parser.add_argument(
        '--descending',
        action='store_true',
        help='Sort rows in descending order.',
    )
    args = parser.parse_args()

    rows = []
    for summary_path in discover_summary_paths(args.summary_glob):
        summary = load_summary(str(summary_path))
        rows.append(build_benchmark_row(summary))

    rows.sort(
        key=lambda row: (row.get(args.sort_by) is None, row.get(args.sort_by)),
        reverse=args.descending,
    )
    rendered = export_rows(rows, output_path=args.output_path, output_format=args.output_format)
    print(rendered)


if __name__ == '__main__':
    main()
