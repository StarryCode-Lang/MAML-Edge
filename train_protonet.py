from utils import setup_logger

import argparse
import os

import protonet


def parse_args():
    parser = argparse.ArgumentParser(description='Implementation of Prototypical Networks on Fault Diagnosis Datasets')
    parser.add_argument('--ways', type=int, default=10,
                        help='Number of classes per task, default=10')
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

    parser.add_argument('--cuda', type=bool, default=True,
                        help='Use CUDA if available, default=True')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed, default=42')

    parser.add_argument('--data_dir_path', type=str, default='./data',
                        help='Path to the data directory, default=./data')
    parser.add_argument('--dataset', type=str, default='CWRU',
                        help='Which dataset to use, options=[CWRU, HST]')
    parser.add_argument('--preprocess', type=str, default='STFT',
                        help='Which preprocessing technique to use, options=[WT, STFT, FFT]')
    parser.add_argument('--train_domains', type=str, default='0,1,2',
                        help='Training domain(s), integers separated by commas, default=0,1,2')
    parser.add_argument('--test_domain', type=int, default=3,
                        help='Test domain, single integer, default=3')
    parser.add_argument('--train_task_num', type=int, default=200,
                        help='Number of sampled tasks per source domain, default=200')
    parser.add_argument('--test_task_num', type=int, default=100,
                        help='Number of sampled tasks for target domain, default=100')
    parser.add_argument('--eval_support_ratio', type=float, default=0.5,
                        help='Fixed support-pool ratio for target-domain evaluation, default=0.5')

    parser.add_argument('--plot', type=bool, default=True,
                        help='Plot the learning curve, default=True')
    parser.add_argument('--plot_path', type=str, default='./images',
                        help='Directory to save the learning curve, default=./images')
    parser.add_argument('--plot_step', type=int, default=200,
                        help='Step for plotting the learning curve, default=200')

    parser.add_argument('--log', type=bool, default=True,
                        help='Log the training process, default=True')
    parser.add_argument('--log_path', type=str, default='./logs',
                        help='Directory to save the logs, default=./logs')

    parser.add_argument('--checkpoint', type=bool, default=True,
                        help='Save the model checkpoints, default=True')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints',
                        help='Directory to save the model checkpoints, default=./checkpoints')
    parser.add_argument('--checkpoint_step', type=int, default=100,
                        help='Step for saving the model checkpoints, default=100')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.dataset not in ['CWRU', 'HST']:
        raise ValueError('Dataset must be either CWRU or HST.')
    if args.preprocess not in ['WT', 'STFT', 'FFT']:
        raise ValueError('Preprocessing technique must be either WT, STFT, or FFT.')
    if not 0.0 < args.eval_support_ratio < 1.0:
        raise ValueError('eval_support_ratio must be between 0 and 1.')

    if args.query_shots is None:
        args.query_shots = args.shots

    args.train_domains = args.train_domains.split(',')
    train_domains_str = ''
    for domain in args.train_domains:
        train_domains_str += str(domain)
    args.train_domains = [int(i) for i in args.train_domains]

    experiment_title = 'ProtoNet_{}_{}_{}w{}s{}q_source{}_target{}'.format(
        args.dataset,
        args.preprocess,
        args.ways,
        args.shots,
        args.query_shots,
        train_domains_str,
        args.test_domain,
    )

    if args.plot and not os.path.exists(args.plot_path):
        os.makedirs(args.plot_path)

    if args.checkpoint and not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    if args.log:
        if not os.path.exists(args.log_path):
            os.makedirs(args.log_path)
        setup_logger(args.log_path, experiment_title)

    protonet.train(args, experiment_title)
