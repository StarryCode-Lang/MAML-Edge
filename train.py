import argparse
import sys


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Unified train/deploy entry for fault diagnosis models',
        add_help=False,
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        choices=['train', 'deploy'],
        help='Run mode: train or deploy',
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['maml', 'protonet', 'cnn'],
        help='Target algorithm: maml, protonet, or cnn',
    )
    parser.add_argument('-h', '--help', action='store_true', dest='help_requested')
    args, remaining = parser.parse_known_args(argv)
    return args, remaining


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    args, remaining = parse_args(argv)

    if args.help_requested and args.algorithm is None:
        top_parser = argparse.ArgumentParser(description='Unified train/deploy entry for fault diagnosis models')
        top_parser.add_argument(
            '--mode',
            type=str,
            default='train',
            choices=['train', 'deploy'],
            help='Run mode: train or deploy',
        )
        top_parser.add_argument(
            '--algorithm',
            type=str,
            required=True,
            choices=['maml', 'protonet', 'cnn'],
            help='Target algorithm: maml, protonet, or cnn',
        )
        top_parser.print_help()
        return

    if args.algorithm is None:
        raise SystemExit('train.py: error: the following arguments are required: --algorithm')

    if args.mode == 'train':
        if args.algorithm == 'maml':
            from model_layer.train_maml import main as run_main
        elif args.algorithm == 'protonet':
            from model_layer.train_protonet import main as run_main
        else:
            from model_layer.train_cnn import main as run_main
    else:
        from deploy_layer.deploy import main as run_main

    forwarded_args = list(remaining)
    if args.algorithm is not None:
        forwarded_args = ['--algorithm', args.algorithm] + forwarded_args
    if args.help_requested:
        forwarded_args = ['--help'] + forwarded_args
    run_main(forwarded_args)


if __name__ == '__main__':
    main()
