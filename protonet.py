from fault_datasets import CWRU_FFT
from models import ProtoNet1D
from utils import (
    print_logs,
    fast_adapt_protonet,
)

import logging
import torch
import random
import numpy as np
import learn2learn as l2l
import matplotlib.pyplot as plt

from torch import nn
from learn2learn.data.transforms import (
    FusedNWaysKShots,
    LoadData,
    RemapLabels,
    ConsecutiveLabels,
)


def train(args, experiment_title):
    logging.info('Experiment: {}'.format(experiment_title))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        device_count = torch.cuda.device_count()
        device = torch.device('cuda')
        logging.info('Training ProtoNet with {} GPU(s).'.format(device_count))
    else:
        device = torch.device('cpu')
        logging.info('Training ProtoNet with CPU.')

    train_tasks, test_tasks = create_datasets(args)
    model, opt, loss = create_model(args, device)

    train_model(args, model, opt, loss, train_tasks, test_tasks, device, experiment_title)


def create_datasets(args):
    if args.dataset != 'CWRU':
        raise ValueError('ProtoNet comparison is implemented on CWRU only.')
    if args.preprocess != 'FFT':
        raise ValueError('ProtoNet comparison is implemented on the FFT route only.')

    logging.info('Training domains: {}.'.format(args.train_domains))
    logging.info('Testing domain: {}.'.format(args.test_domain))
    train_datasets = []
    train_transforms = []
    train_tasks = []

    for i in range(len(args.train_domains)):
        train_datasets.append(CWRU_FFT(args.train_domains[i], args.data_dir_path))
        train_datasets[i] = l2l.data.MetaDataset(train_datasets[i])
        train_transforms.append([
            FusedNWaysKShots(train_datasets[i], n=args.ways, k=2*args.shots),
            LoadData(train_datasets[i]),
            RemapLabels(train_datasets[i]),
            ConsecutiveLabels(train_datasets[i]),
        ])
        train_tasks.append(l2l.data.Taskset(
            train_datasets[i],
            task_transforms=train_transforms[i],
            num_tasks=args.train_task_num,
        ))

    test_dataset = CWRU_FFT(args.test_domain, args.data_dir_path)
    test_dataset = l2l.data.MetaDataset(test_dataset)
    test_transforms = [
        FusedNWaysKShots(test_dataset, n=args.ways, k=2*args.shots),
        LoadData(test_dataset),
        RemapLabels(test_dataset),
        ConsecutiveLabels(test_dataset),
    ]
    test_tasks = l2l.data.Taskset(
        test_dataset,
        task_transforms=test_transforms,
        num_tasks=args.test_task_num,
    )

    return train_tasks, test_tasks


def create_model(args, device):
    model = ProtoNet1D()
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), args.meta_lr)
    loss = nn.CrossEntropyLoss(reduction='mean')

    return model, opt, loss


def train_model(args,
                model, opt, loss,
                train_tasks, test_tasks,
                device,
                experiment_title):
    train_acc_list = []
    train_err_list = []
    test_acc_list = []
    test_err_list = []

    for iteration in range(1, args.iters+1):
        opt.zero_grad()
        meta_train_err_sum = 0.0
        meta_train_acc_sum = 0.0
        meta_test_err_sum = 0.0
        meta_test_acc_sum = 0.0

        train_index = random.randint(0, len(args.train_domains)-1)

        for task in range(args.meta_batch_size):
            batch = train_tasks[train_index].sample()
            evaluation_error, evaluation_accuracy = fast_adapt_protonet(batch,
                                                                        model,
                                                                        loss,
                                                                        args.shots,
                                                                        args.ways,
                                                                        device)
            evaluation_error.backward()
            meta_train_err_sum += evaluation_error.item()
            meta_train_acc_sum += evaluation_accuracy.item()

        for task in range(args.meta_test_batch_size):
            batch = test_tasks.sample()
            evaluation_error, evaluation_accuracy = fast_adapt_protonet(batch,
                                                                        model,
                                                                        loss,
                                                                        args.shots,
                                                                        args.ways,
                                                                        device)
            meta_test_err_sum += evaluation_error.item()
            meta_test_acc_sum += evaluation_accuracy.item()

        meta_train_acc = meta_train_acc_sum / args.meta_batch_size
        meta_train_err = meta_train_err_sum / args.meta_batch_size
        meta_test_err = meta_test_err_sum / args.meta_test_batch_size
        meta_test_acc = meta_test_acc_sum / args.meta_test_batch_size

        train_acc_list.append(meta_train_acc)
        test_acc_list.append(meta_test_acc)
        train_err_list.append(meta_train_err)
        test_err_list.append(meta_test_err)

        if args.plot and iteration % args.plot_step == 0:
            plot_metrics(args,
                         iteration,
                         train_acc_list, test_acc_list,
                         train_err_list, test_err_list,
                         experiment_title)

        if args.checkpoint and iteration % args.checkpoint_step == 0:
            torch.save(model.state_dict(),
                       args.checkpoint_path + '/' +
                       experiment_title +
                       '_{}.pt'.format(iteration))

        if args.log:
            print_logs(iteration, meta_train_err, meta_train_acc, meta_test_err, meta_test_acc)

        for p in model.parameters():
            p.grad.data.mul_(1.0 / args.meta_batch_size)
        opt.step()


def plot_metrics(args,
                 iteration,
                 train_acc, test_acc,
                 train_loss, test_loss,
                 experiment_title):
    if (iteration % args.plot_step == 0):
        plt.figure(figsize=(12, 4))
        plt.subplot(121)
        plt.plot(train_acc, '-o', label="train acc")
        plt.plot(test_acc, '-o', label="test acc")
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.title("Accuracy Curve by Iteration")
        plt.legend()
        plt.subplot(122)
        plt.plot(train_loss, '-o', label="train loss")
        plt.plot(test_loss, '-o', label="test loss")
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title("Loss Curve by Iteration")
        plt.legend()
        plt.savefig(args.plot_path + '/' + experiment_title + '_{}.png'.format(iteration))
        plt.show()


if __name__ == '__main__':
    train()
