import argparse
import sys


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Unified training entry for fault diagnosis models',
        add_help=False,
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['maml', 'protonet', 'cnn'],
        help='Training algorithm to run: maml, protonet, or cnn',
    )
    parser.add_argument('-h', '--help', action='store_true', dest='help_requested')
    args, remaining = parser.parse_known_args(argv)
    return args, remaining


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    args, remaining = parse_args(argv)

    if args.help_requested and args.algorithm is None:
        top_parser = argparse.ArgumentParser(description='Unified training entry for fault diagnosis models')
        top_parser.add_argument(
            '--algorithm',
            type=str,
            required=True,
            choices=['maml', 'protonet', 'cnn'],
            help='Training algorithm to run: maml, protonet, or cnn',
        )
        top_parser.print_help()
        return

    if args.algorithm is None:
        raise SystemExit('train.py: error: the following arguments are required: --algorithm')

    if args.algorithm == 'maml':
        from model_layer.train_maml import main as run_main
    elif args.algorithm == 'protonet':
        from model_layer.train_protonet import main as run_main
    else:
        from model_layer.train_cnn import main as run_main

    if args.help_requested:
        remaining = ['--help'] + remaining
    run_main(remaining)


if __name__ == '__main__':
    main()
