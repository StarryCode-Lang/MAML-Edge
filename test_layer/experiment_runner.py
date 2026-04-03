import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from test_layer.benchmark import build_benchmark_row, load_summary


def parse_csv_list(value, cast=str):
    return [cast(item.strip()) for item in str(value).split(',') if item.strip()]


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {'true', '1', 'yes', 'y'}:
        return True
    if value in {'false', '0', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def _resolve_entry_helpers(algorithm):
    if algorithm == 'maml':
        from model_layer.train_maml import build_experiment_title, normalize_args, parse_args
    elif algorithm == 'protonet':
        from model_layer.train_protonet import build_experiment_title, normalize_args, parse_args
    elif algorithm == 'cnn':
        from model_layer.train_cnn import build_experiment_title, normalize_args, parse_args
    else:
        raise ValueError('Unsupported algorithm: {}'.format(algorithm))
    return parse_args, normalize_args, build_experiment_title


def build_command(config):
    command = [
        sys.executable,
        'train.py',
        '--mode',
        'train',
        '--algorithm',
        config['algorithm'],
        '--dataset',
        config['dataset'],
        '--preprocess',
        config['preprocess'],
        '--ways',
        str(config['ways']),
        '--shots',
        str(config['shots']),
        '--query_shots',
        str(config['query_shots']),
        '--train_domains',
        config['train_domains'],
        '--test_domain',
        str(config['test_domain']),
        '--runtime_backend',
        config['runtime_backend'],
        '--enable_compression',
        'true' if config['enable_compression'] else 'false',
        '--plot',
        'false',
        '--log',
        'true',
    ]
    if config.get('fault_labels'):
        command.extend(['--fault_labels', config['fault_labels']])
    if config.get('cuda') is not None:
        command.extend(['--cuda', 'true' if config['cuda'] else 'false'])
    if config['algorithm'] in {'maml', 'protonet'}:
        if config.get('iters') is not None:
            command.extend(['--iters', str(config['iters'])])
        if config.get('meta_batch_size') is not None:
            command.extend(['--meta_batch_size', str(config['meta_batch_size'])])
        if config.get('train_task_num') is not None:
            command.extend(['--train_task_num', str(config['train_task_num'])])
        if config.get('test_task_num') is not None:
            command.extend(['--test_task_num', str(config['test_task_num'])])
    else:
        if config.get('epochs') is not None:
            command.extend(['--epochs', str(config['epochs'])])
        if config.get('batch_size') is not None:
            command.extend(['--batch_size', str(config['batch_size'])])
        if config.get('test_task_num') is not None:
            command.extend(['--test_task_num', str(config['test_task_num'])])
    return command


def infer_expected_summary_path(command, algorithm):
    forwarded = command[5:]
    parse_args, normalize_args, build_experiment_title = _resolve_entry_helpers(algorithm)
    normalized_args = normalize_args(parse_args(forwarded))
    experiment_title = build_experiment_title(normalized_args)
    return ROOT_DIR / normalized_args.compression_output_path / experiment_title / 'compression_summary.json'


def export_manifest(records, manifest_dir):
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / 'experiment_manifest.json'
    manifest_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8')
    return manifest_path


def main():
    parser = argparse.ArgumentParser(
        description='Run or dry-run a thesis-oriented experiment matrix on top of train.py.',
    )
    parser.add_argument('--algorithms', type=str, default='maml,protonet,cnn')
    parser.add_argument('--preprocesses', type=str, default='FFT,STFT,WT')
    parser.add_argument('--shots', type=str, default='1,5')
    parser.add_argument('--ways', type=int, default=5)
    parser.add_argument('--dataset', type=str, default='CWRU')
    parser.add_argument('--train_domains', type=str, default='0,1,2')
    parser.add_argument('--test_domains', type=str, default='3')
    parser.add_argument('--fault_labels', type=str, default='')
    parser.add_argument('--query_shots', type=str, default='')
    parser.add_argument('--runtime_backend', type=str, default='onnxruntime')
    parser.add_argument('--enable_compression', type=str2bool, default=True)
    parser.add_argument('--cuda', type=str2bool, default=True)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--iters', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--meta_batch_size', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--train_task_num', type=int, default=None)
    parser.add_argument('--test_task_num', type=int, default=None)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--manifest_dir', type=str, default='logs/thesis_runs/latest')
    args = parser.parse_args()

    algorithms = parse_csv_list(args.algorithms, str)
    preprocesses = parse_csv_list(args.preprocesses, str)
    shots_list = parse_csv_list(args.shots, int)
    test_domains = parse_csv_list(args.test_domains, int)
    query_shot_values = parse_csv_list(args.query_shots, int) if args.query_shots else []

    manifest_records = []
    benchmark_rows = []

    for repeat_index in range(args.repeat):
        for algorithm in algorithms:
            for preprocess in preprocesses:
                for shot_index, shots in enumerate(shots_list):
                    query_shots = query_shot_values[shot_index] if shot_index < len(query_shot_values) else shots
                    for test_domain in test_domains:
                        config = {
                            'algorithm': algorithm,
                            'dataset': args.dataset,
                            'preprocess': preprocess,
                            'ways': args.ways,
                            'shots': shots,
                            'query_shots': query_shots,
                            'train_domains': args.train_domains,
                            'test_domain': test_domain,
                            'fault_labels': args.fault_labels,
                            'runtime_backend': args.runtime_backend,
                            'enable_compression': args.enable_compression,
                            'cuda': args.cuda,
                            'iters': args.iters,
                            'epochs': args.epochs,
                            'meta_batch_size': args.meta_batch_size,
                            'batch_size': args.batch_size,
                            'train_task_num': args.train_task_num,
                            'test_task_num': args.test_task_num,
                        }
                        command = build_command(config)
                        expected_summary_path = infer_expected_summary_path(command, algorithm)
                        record = {
                            'repeat_index': repeat_index,
                            'config': config,
                            'command': command,
                            'command_string': subprocess.list2cmdline(command),
                            'expected_summary_path': str(expected_summary_path),
                            'executed': False,
                            'return_code': None,
                        }
                        if args.execute:
                            completed = subprocess.run(command, cwd=str(ROOT_DIR), check=False)
                            record['executed'] = True
                            record['return_code'] = completed.returncode
                        if expected_summary_path.exists():
                            benchmark_rows.append(build_benchmark_row(load_summary(str(expected_summary_path))))
                            record['summary_found'] = True
                        else:
                            record['summary_found'] = False
                        manifest_records.append(record)

    manifest_dir = ROOT_DIR / args.manifest_dir
    manifest_path = export_manifest(
        {
            'generated_at': datetime.now().isoformat(timespec='seconds'),
            'execute': args.execute,
            'records': manifest_records,
            'benchmark_rows': benchmark_rows,
        },
        manifest_dir,
    )
    print(manifest_path)
    if benchmark_rows:
        benchmark_path = manifest_dir / 'benchmark_rows.json'
        benchmark_path.write_text(json.dumps(benchmark_rows, indent=2, ensure_ascii=False), encoding='utf-8')
        print(benchmark_path)


if __name__ == '__main__':
    main()
