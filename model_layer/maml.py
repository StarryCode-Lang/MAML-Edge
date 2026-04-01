import logging
import os
import random

import learn2learn as l2l
import matplotlib.pyplot as plt
import numpy as np
import torch

from torch import nn
from learn2learn.data.transforms import ConsecutiveLabels, FusedNWaysKShots, LoadData, RemapLabels

from .experiment import build_classifier_from_args, build_dataset_from_args, get_model_config
from .utils import (
    clone_state_dict_to_cpu,
    create_class_pools,
    deterministic_domain_index,
    deterministic_fixed_pool_episode,
    deterministic_task_sample,
    fast_adapt,
    is_better_model_record,
    print_logs,
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
        logging.info('Training MAML with {} GPU(s).'.format(device_count))
    else:
        device = torch.device('cpu')
        logging.info('Training MAML with CPU.')

    train_tasks, test_dataset, test_pools = create_datasets(args)
    model, maml_wrapper, opt, loss = create_model(args, device)
    training_result = train_model(
        args,
        model,
        maml_wrapper,
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
            algorithm='maml',
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
        train_dataset = build_dataset_from_args(args, domain)
        meta_dataset = l2l.data.MetaDataset(train_dataset)
        train_transforms = [
            FusedNWaysKShots(meta_dataset, n=args.ways, k=args.shots + args.query_shots),
            LoadData(meta_dataset),
            RemapLabels(meta_dataset),
            ConsecutiveLabels(meta_dataset),
        ]
        train_tasks.append(l2l.data.Taskset(
            meta_dataset,
            task_transforms=train_transforms,
            num_tasks=args.train_task_num,
        ))
    test_dataset = build_dataset_from_args(args, args.test_domain)
    test_pools = create_class_pools(test_dataset, support_ratio=args.eval_support_ratio)
    return train_tasks, test_dataset, test_pools


def create_model(args, device):
    model = build_classifier_from_args(args, output_size=len(args.fault_labels))
    model.to(device)
    maml_wrapper = l2l.algorithms.MAML(model, lr=args.fast_lr, first_order=args.first_order)
    opt = torch.optim.Adam(model.parameters(), args.meta_lr)
    loss = nn.CrossEntropyLoss(reduction='mean')
    return model, maml_wrapper, opt, loss


def train_model(args, model, maml_wrapper, opt, loss, train_tasks, test_dataset, test_pools, device, experiment_title):
    train_acc_list = []
    train_err_list = []
    test_acc_list = []
    test_err_list = []
    history_records = []
    checkpoint_paths = []
    best_record = None
    best_state_dict = None

    for iteration in range(1, args.iters + 1):
        opt.zero_grad()
        meta_train_err_sum = 0.0
        meta_train_acc_sum = 0.0
        meta_test_err_sum = 0.0
        meta_test_acc_sum = 0.0

        for episode in range(args.meta_batch_size):
            train_index = deterministic_domain_index(args.seed, iteration, episode, len(train_tasks))
            learner = maml_wrapper.clone()
            batch_seed = args.seed + iteration * 100000 + episode
            batch = deterministic_task_sample(train_tasks[train_index], batch_seed)
            evaluation_error, evaluation_accuracy = fast_adapt(
                batch,
                learner,
                loss,
                args.adapt_steps,
                args.shots,
                args.ways,
                device,
                args.query_shots,
            )
            evaluation_error.backward()
            meta_train_err_sum += evaluation_error.item()
            meta_train_acc_sum += evaluation_accuracy.item()

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
            learner = maml_wrapper.clone()
            evaluation_error, evaluation_accuracy = fast_adapt(
                batch,
                learner,
                loss,
                args.adapt_steps,
                args.shots,
                args.ways,
                device,
                args.query_shots,
            )
            meta_test_err_sum += evaluation_error.item()
            meta_test_acc_sum += evaluation_accuracy.item()

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

        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.data.mul_(1.0 / args.meta_batch_size)
        opt.step()

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
        'algorithm': 'maml',
        'history': history_records,
        'history_path': history_path,
        'best_record': best_record,
        'best_state_dict': best_state_dict,
        'best_checkpoint_path': best_checkpoint_path,
        'model_config': get_model_config(model),
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
