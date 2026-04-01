import argparse
import json
import os
import sys


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch

from model_layer.experiment import get_model_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Run compression/deployment export from an existing best checkpoint',
        add_help=False,
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['maml', 'protonet', 'cnn'],
        help='Algorithm to deploy: maml, protonet, or cnn',
    )
    parser.add_argument('--best_checkpoint_path', type=str, default=None,
                        help='Optional path to a *_best.pt checkpoint. Defaults to the inferred experiment path.')
    parser.add_argument('--history_path', type=str, default=None,
                        help='Optional path to a *_history.json file. Defaults to the inferred experiment path.')
    parser.add_argument('-h', '--help', action='store_true', dest='help_requested')
    args, remaining = parser.parse_known_args(argv)
    return args, remaining


def load_best_record(history_path):
    if history_path is None or not os.path.exists(history_path):
        return None
    with open(history_path, 'r', encoding='utf-8') as file_pointer:
        history_records = json.load(file_pointer)
    if not history_records:
        return None
    from model_layer.utils import is_better_model_record

    best_record = None
    for record in history_records:
        if is_better_model_record(record, best_record):
            best_record = record
    return best_record


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    args, remaining = parse_args(argv)

    if args.help_requested and args.algorithm is None:
        top_parser = argparse.ArgumentParser(
            description='Run compression/deployment export from an existing best checkpoint',
        )
        top_parser.add_argument(
            '--algorithm',
            type=str,
            required=True,
            choices=['maml', 'protonet', 'cnn'],
            help='Algorithm to deploy: maml, protonet, or cnn',
        )
        top_parser.add_argument('--best_checkpoint_path', type=str, default=None)
        top_parser.add_argument('--history_path', type=str, default=None)
        top_parser.print_help()
        return

    if args.algorithm is None:
        raise SystemExit('deploy.py: error: the following arguments are required: --algorithm')

    if args.algorithm == 'maml':
        from model_layer import maml as algorithm_module
        from model_layer.train_maml import (
            build_experiment_title,
            normalize_args,
            parse_args as parse_algorithm_args,
            prepare_runtime_dirs,
        )
    elif args.algorithm == 'protonet':
        from model_layer import protonet as algorithm_module
        from model_layer.train_protonet import (
            build_experiment_title,
            normalize_args,
            parse_args as parse_algorithm_args,
            prepare_runtime_dirs,
        )
    else:
        from model_layer import cnn_baseline as algorithm_module
        from model_layer.train_cnn import (
            build_experiment_title,
            normalize_args,
            parse_args as parse_algorithm_args,
            prepare_runtime_dirs,
        )

    if args.help_requested:
        remaining = ['--help'] + remaining
    algorithm_args = normalize_args(parse_algorithm_args(remaining))
    algorithm_args.enable_compression = True

    experiment_title = build_experiment_title(algorithm_args)
    prepare_runtime_dirs(algorithm_args, experiment_title)

    best_checkpoint_path = args.best_checkpoint_path
    if best_checkpoint_path is None:
        best_checkpoint_path = os.path.join(
            algorithm_args.checkpoint_path,
            '{}_best.pt'.format(experiment_title),
        )
    if not os.path.exists(best_checkpoint_path):
        raise FileNotFoundError('Best checkpoint not found: {}'.format(best_checkpoint_path))

    history_path = args.history_path
    if history_path is None:
        history_path = os.path.join(
            algorithm_args.checkpoint_path,
            '{}_history.json'.format(experiment_title),
        )

    if algorithm_args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if args.algorithm == 'maml':
        train_tasks, test_dataset, test_pools = algorithm_module.create_datasets(algorithm_args)
        model, _, _, _ = algorithm_module.create_model(algorithm_args, device)
    elif args.algorithm == 'protonet':
        train_tasks, test_dataset, test_pools = algorithm_module.create_datasets(algorithm_args)
        model, _, _ = algorithm_module.create_model(algorithm_args, device)
    else:
        train_tasks, test_dataset, test_pools = algorithm_module.create_datasets(algorithm_args)
        model, _, _ = algorithm_module.create_model(algorithm_args, device)

    state_dict = torch.load(best_checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)

    training_result = {
        'algorithm': args.algorithm,
        'best_state_dict': state_dict,
        'best_record': load_best_record(history_path),
        'best_checkpoint_path': best_checkpoint_path,
        'history_path': history_path if os.path.exists(history_path) else None,
        'model_config': get_model_config(model),
    }

    from deploy_layer.compression import run_compression_pipeline

    run_compression_pipeline(
        args=algorithm_args,
        algorithm=args.algorithm,
        experiment_title=experiment_title,
        training_result=training_result,
        train_tasks=train_tasks,
        test_dataset=test_dataset,
        test_pools=test_pools,
        device=device,
    )


if __name__ == '__main__':
    main()
