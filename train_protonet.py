from utils import setup_logger

import argparse
import os
import protonet


def parse_args():
    parser = argparse.ArgumentParser(description='Implementation of \
                                     ProtoNet on \
                                     Fault Diagnosis Datasets')
    parser.add_argument('--ways', type=int, default=10,
                        help='Number of classes per task, default=10')
    parser.add_argument('--shots', type=int, default=5,
                        help='Number of support examples per class, default=1')

    parser.add_argument('--meta_lr', type=float, default=0.001,
                        help='Outer loop learning rate, default=0.001')
    parser.add_argument('--meta_batch_size', type=int, default=32,
                        help='Number of outer loop iterations, \
                              i.e. no. of meta-tasks for each batch, \
                              default=32')
    parser.add_argument('--meta_test_batch_size', type=int, default=None,
                        help='Number of meta-test tasks for each iteration. \
                              If None, uses meta_batch_size.')
    parser.add_argument('--iters', type=int, default=300,
                        help='Number of outer-loop iterations, default=300')

    parser.add_argument('--cuda', type=bool, default=True,
                        help='Use CUDA if available, default=True')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed, default=42')

    parser.add_argument('--data_dir_path', type=str, default='./data',
                        help='Path to the data directory, default=./data')
    parser.add_argument('--dataset', type=str, default='CWRU',
                        help='Which dataset to use, default=CWRU')
    parser.add_argument('--preprocess', type=str, default='FFT',
                        help='Which preprocessing technique to use, default=FFT')
    parser.add_argument('--train_domains', type=str, default='0,1,2',
                        help='Training domain, integer(s) separated by commas, default=0,1,2')
    parser.add_argument('--test_domain', type=int, default=3,
                        help='Test domain, single integer, default=3')
    parser.add_argument('--train_task_num', type=int, default=200,
                        help='Number of samples per domain for training, default=200')
    parser.add_argument('--test_task_num', type=int, default=100,
                        help='Number of samples per domain for testing, default=100')

    parser.add_argument('--plot', type=bool, default=True,
                        help='Plot the learning curve, default=True')
    parser.add_argument('--plot_path', type=str, default='./images',
                        help='Directory to save the learning curve, default=./images')
    parser.add_argument('--plot_step', type=int, default=50,
                        help='Step for plotting the learning curve, default=50')

    parser.add_argument('--log', type=bool, default=True,
                        help='Log the training process, default=True')
    parser.add_argument('--log_path', type=str, default='./logs',
                        help='Directory to save the logs, default=./logs')

    parser.add_argument('--checkpoint', type=bool, default=True,
                        help='Save the model checkpoints, default=True')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints',
                        help='Directory to save the model checkpoints, default=./checkpoints')
    parser.add_argument('--checkpoint_step', type=int, default=50,
                        help='Step for saving the model checkpoints, default=50')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.dataset != 'CWRU':
        raise ValueError('ProtoNet comparison is implemented on CWRU only.')
    if args.preprocess != 'FFT':
        raise ValueError('ProtoNet comparison is implemented on the FFT route only.')

    args.train_domains = args.train_domains.split(',')
    train_domains_str = ''
    for i in range(len(args.train_domains)):
        train_domains_str += str(args.train_domains[i])
    args.train_domains = [int(i) for i in args.train_domains]
    if args.meta_test_batch_size is None:
        args.meta_test_batch_size = args.meta_batch_size

    experiment_title = 'ProtoNet_{}_{}_{}w{}s_source{}_target{}'.format(args.dataset,
                                                args.preprocess,
                                                args.ways,
                                                args.shots,
                                                train_domains_str,
                                                args.test_domain)
    if args.plot:
        if not os.path.exists(args.plot_path):
            os.makedirs(args.plot_path)

    if args.checkpoint:
        if not os.path.exists(args.checkpoint_path):
            os.makedirs(args.checkpoint_path)

    if args.log:
        if not os.path.exists(args.log_path):
            os.makedirs(args.log_path)
        setup_logger(args.log_path, experiment_title)

    protonet.train(args, experiment_title)
