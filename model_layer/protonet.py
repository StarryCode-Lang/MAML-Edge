import logging
import os
import random

import learn2learn as l2l
import matplotlib.pyplot as plt
import numpy as np
import torch

from torch import nn
from learn2learn.data.transforms import ConsecutiveLabels, FusedNWaysKShots, LoadData, RemapLabels

from data_layer.fault_datasets import CWRU, CWRU_FFT, HST, HST_FFT
from .models import CNN1DEncoder, CNN2DEncoder
from .utils import (
    clone_state_dict_to_cpu,
    create_class_pools,
    deterministic_domain_index,
    deterministic_fixed_pool_episode,
    deterministic_task_sample,
    is_better_model_record,
    print_logs,
    protonet_fast_adapt,
    write_json,
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

    train_tasks, test_dataset, test_pools = create_datasets(args)
    model, opt, loss = create_model(args, device)
    training_result = train_model(
        args,
        model,
        opt,
        loss,
        train_tasks,
        test_dataset,
        test_pools,
        device,
        experiment_title,
    )

    if args.enable_compression:
        from deploy_layer.compression import run_compression_pipeline

        run_compression_pipeline(
            args=args,
            algorithm='protonet',
            experiment_title=experiment_title,
            training_result=training_result,
            train_tasks=train_tasks,
            test_dataset=test_dataset,
            test_pools=test_pools,
            device=device,
        )

    return training_result


def create_datasets(args):
    logging.info('Training domains: {}.'.format(args.train_domains))
    logging.info('Testing domain: {}.'.format(args.test_domain))
    train_tasks = []

    for domain in args.train_domains:
        dataset = build_dataset(args, domain)
        meta_dataset = l2l.data.MetaDataset(dataset)
        transforms = [
            FusedNWaysKShots(meta_dataset, n=args.ways, k=args.shots + args.query_shots),
            LoadData(meta_dataset),
            RemapLabels(meta_dataset),
            ConsecutiveLabels(meta_dataset),
        ]
        train_tasks.append(l2l.data.Taskset(
            meta_dataset,
            task_transforms=transforms,
            num_tasks=args.train_task_num,
        ))

    test_dataset = build_dataset(args, args.test_domain)
    test_pools = create_class_pools(test_dataset, support_ratio=args.eval_support_ratio)
    return train_tasks, test_dataset, test_pools


def build_dataset(args, domain):
    if args.preprocess == 'FFT':
        if args.dataset == 'HST':
            return HST_FFT(domain, args.data_dir_path, labels=args.fault_labels)
        return CWRU_FFT(domain, args.data_dir_path, label_subset=args.fault_labels)
    if args.dataset == 'HST':
        return HST(domain, args.data_dir_path, args.preprocess, label_subset=range(len(args.fault_labels)))
    return CWRU(domain, args.data_dir_path, args.preprocess, label_subset=args.fault_labels)


def create_model(args, device):
    if args.preprocess == 'FFT':
        model = CNN1DEncoder()
    else:
        model = CNN2DEncoder()
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), args.meta_lr)
    loss = nn.CrossEntropyLoss(reduction='mean')
    return model, opt, loss


def get_model_config(model, args):
    if args.preprocess == 'FFT':
        return {
            'model_type': 'CNN1DEncoder',
            'channels': list(model.channels),
            'pooled_length': model.pooled_length,
        }
    return {
        'model_type': 'CNN2DEncoder',
        'channels': list(model.channels),
        'in_channels': model.in_channels,
    }


def train_model(args, model, opt, loss, train_tasks, test_dataset, test_pools, device, experiment_title):
    train_acc_list = []
    train_err_list = []
    test_acc_list = []
    test_err_list = []
    history_records = []
    checkpoint_paths = []
    best_record = None
    best_state_dict = None

    for iteration in range(1, args.iters + 1):
        model.train()
        opt.zero_grad()
        meta_train_err_sum = 0.0
        meta_train_acc_sum = 0.0

        for episode in range(args.meta_batch_size):
            train_index = deterministic_domain_index(args.seed, iteration, episode, len(train_tasks))
            batch_seed = args.seed + iteration * 100000 + episode
            batch = deterministic_task_sample(train_tasks[train_index], batch_seed)
            train_error, train_accuracy = protonet_fast_adapt(
                batch,
                model,
                loss,
                args.ways,
                args.shots,
                args.query_shots,
                device,
            )
            train_error.backward()
            meta_train_err_sum += train_error.item()
            meta_train_acc_sum += train_accuracy.item()

        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.data.mul_(1.0 / args.meta_batch_size)
        opt.step()

        model.eval()
        meta_test_err_sum = 0.0
        meta_test_acc_sum = 0.0
        with torch.no_grad():
            for episode in range(args.test_task_num):
                batch_seed = args.seed + 50000000 + iteration * 100000 + episode
                batch = deterministic_fixed_pool_episode(
                    test_dataset,
                    test_pools[0],
                    test_pools[1],
                    args.ways,
                    args.shots,
                    args.query_shots,
                    batch_seed,
                )
                test_error, test_accuracy = protonet_fast_adapt(
                    batch,
                    model,
                    loss,
                    args.ways,
                    args.shots,
                    args.query_shots,
                    device,
                )
                meta_test_err_sum += test_error.item()
                meta_test_acc_sum += test_accuracy.item()

        meta_train_acc = meta_train_acc_sum / args.meta_batch_size
        meta_train_err = meta_train_err_sum / args.meta_batch_size
        meta_test_err = meta_test_err_sum / args.test_task_num
        meta_test_acc = meta_test_acc_sum / args.test_task_num

        train_acc_list.append(meta_train_acc)
        test_acc_list.append(meta_test_acc)
        train_err_list.append(meta_train_err)
        test_err_list.append(meta_test_err)

        current_record = {
            'iteration': iteration,
            'meta_train_acc': meta_train_acc,
            'meta_train_loss': meta_train_err,
            'meta_test_acc': meta_test_acc,
            'meta_test_loss': meta_test_err,
        }
        if is_better_model_record(current_record, best_record):
            best_record = dict(current_record)
            best_state_dict = clone_state_dict_to_cpu(model)

        if args.plot and iteration % args.plot_step == 0:
            plot_metrics(
                args,
                iteration,
                train_acc_list,
                test_acc_list,
                train_err_list,
                test_err_list,
                experiment_title,
            )

        if args.checkpoint and iteration % args.checkpoint_step == 0:
            checkpoint_path = os.path.join(args.checkpoint_path, '{}_{}.pt'.format(experiment_title, iteration))
            torch.save(model.state_dict(), checkpoint_path)
            checkpoint_paths.append(checkpoint_path)
            current_record['checkpoint_path'] = checkpoint_path

        history_records.append(current_record)

        if args.log:
            print_logs(iteration, meta_train_err, meta_train_acc, meta_test_err, meta_test_acc)

    history_path = None
    best_checkpoint_path = None
    if args.checkpoint or args.enable_compression:
        history_path = os.path.join(args.checkpoint_path, '{}_history.json'.format(experiment_title))
        write_json(history_path, history_records)

    if best_state_dict is not None and (args.checkpoint or args.enable_compression):
        best_checkpoint_path = os.path.join(args.checkpoint_path, '{}_best.pt'.format(experiment_title))
        torch.save(best_state_dict, best_checkpoint_path)

    if args.checkpoint and not args.keep_all_checkpoints:
        for checkpoint_path in checkpoint_paths:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)

    return {
        'algorithm': 'protonet',
        'history': history_records,
        'history_path': history_path,
        'best_record': best_record,
        'best_state_dict': best_state_dict,
        'best_checkpoint_path': best_checkpoint_path,
        'model_config': get_model_config(model, args),
    }


def plot_metrics(args, iteration, train_acc, test_acc, train_loss, test_loss, experiment_title):
    if iteration % args.plot_step == 0:
        plt.figure(figsize=(12, 4))
        plt.subplot(121)
        plt.plot(train_acc, '-o', label='train acc')
        plt.plot(test_acc, '-o', label='test acc')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curve by Iteration')
        plt.legend()
        plt.subplot(122)
        plt.plot(train_loss, '-o', label='train loss')
        plt.plot(test_loss, '-o', label='test loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss Curve by Iteration')
        plt.legend()
        plt.savefig(os.path.join(args.plot_path, '{}_{}.png'.format(experiment_title, iteration)))
        plt.close()
