import argparse
import os

from .experiment import (
    DEFAULT_FFT_POOLED_LENGTH,
    apply_default_schedule,
    normalize_shared_args,
)
from .utils import setup_logger


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {'true', '1', 'yes', 'y'}:
        return True
    if value in {'false', '0', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Implementation of Prototypical Networks on Fault Diagnosis Datasets')
    parser.add_argument('--ways', type=int, default=5,
                        help='Number of classes per task, default=5')
    parser.add_argument('--shots', type=int, default=5,
                        help='Number of support examples per class, default=5')
    parser.add_argument('--query_shots', type=int, default=None,
                        help='Number of query examples per class, default=same as shots')

    parser.add_argument('--meta_lr', type=float, default=0.001,
                        help='Episode optimizer learning rate, default=0.001')
    parser.add_argument('--meta_batch_size', type=int, default=64,
                        help='Number of episodes per iteration, default=64')
    parser.add_argument('--iters', type=int, default=1000,
                        help='Number of training iterations, default=1000')

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
                        help='Fault labels to keep. Defaults to all 10 CWRU labels or the project HST subset.')
    parser.add_argument('--fft_channels', type=str, default='32,64,64',
                        help='Shared 1D backbone channels, default=32,64,64')
    parser.add_argument('--image_channels', type=str, default='64,64,64,64',
                        help='Shared 2D backbone channels, default=64,64,64,64')
    parser.add_argument('--fft_pooled_length', type=int, default=DEFAULT_FFT_POOLED_LENGTH,
                        help='Shared 1D adaptive pooled length, default=64')
    parser.add_argument('--train_domains', type=str, default='0,1,2',
                        help='Training domains, integers separated by commas, default=0,1,2')
    parser.add_argument('--test_domain', type=int, default=3,
                        help='Test domain, single integer, default=3')
    parser.add_argument('--train_task_num', type=int, default=200,
                        help='Number of sampled tasks per source domain, default=200')
    parser.add_argument('--test_task_num', type=int, default=100,
                        help='Number of sampled tasks for target domain, default=100')
    parser.add_argument('--eval_support_ratio', type=float, default=0.5,
                        help='Fixed support-pool ratio for target-domain evaluation, default=0.5')

    parser.add_argument('--plot', type=str2bool, default=True,
                        help='Plot the learning curve, default=True')
    parser.add_argument('--plot_path', type=str, default='./images',
                        help='Directory to save the learning curve, default=./images')
    parser.add_argument('--plot_step', type=int, default=None,
                        help='Step for plotting the learning curve, default=iters//5')

    parser.add_argument('--log', type=str2bool, default=True,
                        help='Log the training process, default=True')
    parser.add_argument('--log_path', type=str, default='./logs',
                        help='Directory to save the logs, default=./logs')

    parser.add_argument('--checkpoint', type=str2bool, default=True,
                        help='Save model checkpoints, default=True')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints, default=./checkpoints')
    parser.add_argument('--checkpoint_step', type=int, default=None,
                        help='Step for saving model checkpoints, default=iters//5')
    parser.add_argument('--keep_all_checkpoints', type=str2bool, default=False,
                        help='Keep every intermediate checkpoint, default=False')

    parser.add_argument('--enable_compression', type=str2bool, default=False,
                        help='Run the deployment compression pipeline after training, default=False')
    parser.add_argument('--prune_ratio', type=float, default=0.4,
                        help='Structured pruning ratio, default=0.4')
    parser.add_argument('--compression_finetune_iters', type=int, default=100,
                        help='Short recovery fine-tuning iterations after pruning, default=100')
    parser.add_argument('--compression_meta_batch_size', type=int, default=16,
                        help='Meta-batch size during pruning recovery, default=16')
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

    return parser.parse_args(argv)


def normalize_args(args):
    args = normalize_shared_args(args, require_query_shots=True)
    return apply_default_schedule(args, args.iters)


def build_experiment_title(args):
    train_domains_str = ''.join(str(item) for item in args.train_domains)
    label_tag = ''.join(str(item) for item in args.fault_labels)
    return 'ProtoNet_{}_{}_{}w{}s{}q_source{}_target{}_labels{}'.format(
        args.dataset,
        args.preprocess,
        args.ways,
        args.shots,
        args.query_shots,
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


def main(argv=None):
    from . import protonet

    args = normalize_args(parse_args(argv))
    experiment_title = build_experiment_title(args)
    prepare_runtime_dirs(args, experiment_title)
    protonet.train(args, experiment_title)


if __name__ == "__main__":
    main()
