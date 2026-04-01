import argparse
import os

from . import cnn_baseline
from .utils import resolve_fault_labels, setup_logger


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {'true', '1', 'yes', 'y'}:
        return True
    if value in {'false', '0', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description='Traditional CNN baseline for fault diagnosis')
    parser.add_argument('--ways', type=int, default=5,
                        help='Number of classes for target adaptation, default=5')
    parser.add_argument('--shots', type=int, default=5,
                        help='Number of support examples per class for target adaptation, default=5')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of source training epochs, default=50')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Source training batch size, default=64')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Source training learning rate, default=0.001')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay for source training, default=0.0')
    parser.add_argument('--finetune_epochs', type=int, default=20,
                        help='Short target fine-tune epochs, default=20')
    parser.add_argument('--finetune_lr', type=float, default=0.0005,
                        help='Target fine-tune learning rate, default=0.0005')

    parser.add_argument('--cuda', type=str2bool, default=True,
                        help='Use CUDA if available, default=True')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed, default=42')

    parser.add_argument('--data_dir_path', type=str, default='./data',
                        help='Path to the data directory, default=./data')
    parser.add_argument('--dataset', type=str, default='CWRU',
                        help='Which dataset to use, options=[CWRU, HST]')
    parser.add_argument('--preprocess', type=str, default='FFT',
                        help='Which preprocessing technique to use, options=[WT, STFT, FFT]')
    parser.add_argument('--fault_labels', type=str, default=None,
                        help='Fault labels to keep. Defaults to project 5-class subset.')
    parser.add_argument('--train_domains', type=str, default='0,1,2',
                        help='Training domains, integers separated by commas, default=0,1,2')
    parser.add_argument('--test_domain', type=int, default=3,
                        help='Test domain, single integer, default=3')
    parser.add_argument('--eval_support_ratio', type=float, default=0.5,
                        help='Fixed support-pool ratio for target-domain evaluation, default=0.5')

    parser.add_argument('--plot', type=str2bool, default=True,
                        help='Plot the learning curve, default=True')
    parser.add_argument('--plot_path', type=str, default='./images',
                        help='Directory to save the learning curve, default=./images')
    parser.add_argument('--plot_step', type=int, default=10,
                        help='Step for plotting the learning curve, default=10')

    parser.add_argument('--log', type=str2bool, default=True,
                        help='Log the training process, default=True')
    parser.add_argument('--log_path', type=str, default='./logs',
                        help='Directory to save the logs, default=./logs')

    parser.add_argument('--checkpoint', type=str2bool, default=True,
                        help='Save model checkpoints, default=True')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints, default=./checkpoints')
    parser.add_argument('--checkpoint_step', type=int, default=10,
                        help='Epoch interval for saving checkpoints, default=10')
    parser.add_argument('--keep_all_checkpoints', type=str2bool, default=False,
                        help='Keep every intermediate checkpoint, default=False')

    parser.add_argument('--enable_compression', type=str2bool, default=False,
                        help='Run the deployment compression pipeline after training, default=False')
    parser.add_argument('--prune_ratio', type=float, default=0.4,
                        help='Structured pruning ratio, default=0.4')
    parser.add_argument('--compression_finetune_iters', type=int, default=100,
                        help='Short recovery fine-tuning steps after pruning, default=100')
    parser.add_argument('--compression_output_path', type=str, default='./deploy_artifacts',
                        help='Directory for ONNX/compression artifacts, default=./deploy_artifacts')
    parser.add_argument('--calibration_size', type=int, default=32,
                        help='Number of samples used for PTQ calibration, default=32')
    parser.add_argument('--enable_qat_recovery', type=str2bool, default=False,
                        help='Enable short-cycle QAT recovery when PTQ drop is large, default=False')
    parser.add_argument('--qat_recovery_epochs', type=int, default=5,
                        help='Epochs for optional QAT recovery, default=5')
    parser.add_argument('--qat_drop_threshold', type=float, default=0.02,
                        help='Trigger QAT recovery when PTQ accuracy drops by more than this value, default=0.02')
    parser.add_argument('--onnx_opset', type=int, default=17,
                        help='ONNX export opset version, default=17')

    return parser.parse_args()


def normalize_args(args):
    if args.dataset not in ['CWRU', 'HST']:
        raise ValueError('Dataset must be either CWRU or HST.')
    if args.preprocess not in ['WT', 'STFT', 'FFT']:
        raise ValueError('Preprocessing technique must be either WT, STFT, or FFT.')
    if not 0.0 < args.eval_support_ratio < 1.0:
        raise ValueError('eval_support_ratio must be between 0 and 1.')
    if not 0.0 <= args.prune_ratio < 1.0:
        raise ValueError('prune_ratio must be between 0 and 1.')

    args.fault_labels = resolve_fault_labels(args.dataset, args.fault_labels)
    if args.ways > len(args.fault_labels):
        raise ValueError('ways cannot exceed the number of selected fault labels.')
    args.train_domains = [int(item) for item in args.train_domains.split(',') if item.strip()]
    return args


def build_experiment_title(args):
    train_domains_str = ''.join(str(item) for item in args.train_domains)
    label_tag = ''.join(str(item) for item in args.fault_labels)
    return 'CNN_{}_{}_{}w{}s_source{}_target{}_labels{}'.format(
        args.dataset,
        args.preprocess,
        args.ways,
        args.shots,
        train_domains_str,
        args.test_domain,
        label_tag,
    )


def prepare_runtime_dirs(args, experiment_title):
    if args.plot:
        os.makedirs(args.plot_path, exist_ok=True)
    if args.log:
        os.makedirs(args.log_path, exist_ok=True)
        setup_logger(args.log_path, experiment_title)
    if args.checkpoint or args.enable_compression:
        os.makedirs(args.checkpoint_path, exist_ok=True)
    if args.enable_compression:
        os.makedirs(args.compression_output_path, exist_ok=True)


def main():
    args = normalize_args(parse_args())
    experiment_title = build_experiment_title(args)
    prepare_runtime_dirs(args, experiment_title)
    cnn_baseline.train(args, experiment_title)


if __name__ == '__main__':
    main()
