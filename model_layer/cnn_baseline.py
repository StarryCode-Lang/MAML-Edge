import copy
import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from data_layer.fault_datasets import CWRU, CWRU_FFT, HST, HST_FFT
from .models import CNN1D, CNN2D
from .utils import accuracy, clone_state_dict_to_cpu, create_class_pools, is_better_model_record, write_json


def train(args, experiment_title):
    logging.info('Experiment: {}'.format(experiment_title))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        device_count = torch.cuda.device_count()
        device = torch.device('cuda')
        logging.info('Training CNN baseline with {} GPU(s).'.format(device_count))
    else:
        device = torch.device('cpu')
        logging.info('Training CNN baseline with CPU.')

    source_loader, test_dataset, test_pools = create_datasets(args)
    model, optimizer, criterion = create_model(args, device)
    training_result = train_model(
        args=args,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        source_loader=source_loader,
        test_dataset=test_dataset,
        test_pools=test_pools,
        device=device,
        experiment_title=experiment_title,
    )

    if args.enable_compression:
        from deploy_layer.compression import run_compression_pipeline

        run_compression_pipeline(
            args=args,
            algorithm='cnn',
            experiment_title=experiment_title,
            training_result=training_result,
            train_tasks=source_loader,
            test_dataset=test_dataset,
            test_pools=test_pools,
            device=device,
        )

    return training_result


def build_dataset(args, domain):
    if args.preprocess == 'FFT':
        if args.dataset == 'HST':
            return HST_FFT(domain, args.data_dir_path, labels=args.fault_labels)
        return CWRU_FFT(domain, args.data_dir_path, label_subset=args.fault_labels)
    if args.dataset == 'HST':
        return HST(domain, args.data_dir_path, args.preprocess, label_subset=range(len(args.fault_labels)))
    return CWRU(domain, args.data_dir_path, args.preprocess, label_subset=args.fault_labels)


def create_datasets(args):
    source_datasets = [build_dataset(args, domain) for domain in args.train_domains]
    source_dataset = ConcatDataset(source_datasets)
    source_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_dataset = build_dataset(args, args.test_domain)
    test_pools = create_class_pools(test_dataset, support_ratio=args.eval_support_ratio)
    return source_loader, test_dataset, test_pools


def create_model(args, device):
    output_size = len(args.fault_labels)
    if args.preprocess == 'FFT':
        model = CNN1D(output_size=output_size)
    else:
        model = CNN2D(output_size=output_size)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    return model, optimizer, criterion


def get_model_config(model, args):
    if args.preprocess == 'FFT':
        return {
            'model_type': 'CNN1D',
            'output_size': model.fc.out_features,
            'channels': list(model.encoder.channels),
            'pooled_length': model.encoder.pooled_length,
        }
    return {
        'model_type': 'CNN2D',
        'output_size': model.fc.out_features,
        'channels': list(model.encoder.channels),
        'in_channels': model.encoder.in_channels,
    }


def train_model(args, model, optimizer, criterion, source_loader, test_dataset, test_pools, device, experiment_title):
    train_acc_list = []
    train_loss_list = []
    target_acc_list = []
    target_loss_list = []
    history_records = []
    checkpoint_paths = []
    best_record = None
    best_state_dict = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0

        for batch_data, batch_labels in source_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            logits = model(batch_data)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            batch_size = batch_labels.size(0)
            running_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == batch_labels).sum().item()
            total += batch_size

        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)
        target_metrics = evaluate_target_deployment(args, model, test_dataset, test_pools, device)

        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        target_acc_list.append(target_metrics['accuracy'])
        target_loss_list.append(target_metrics['loss'])

        current_record = {
            'iteration': epoch,
            'meta_train_acc': train_acc,
            'meta_train_loss': train_loss,
            'meta_test_acc': target_metrics['accuracy'],
            'meta_test_loss': target_metrics['loss'],
        }
        if is_better_model_record(current_record, best_record):
            best_record = dict(current_record)
            best_state_dict = clone_state_dict_to_cpu(model)

        if args.plot and epoch % args.plot_step == 0:
            plot_metrics(
                args,
                epoch,
                train_acc_list,
                target_acc_list,
                train_loss_list,
                target_loss_list,
                experiment_title,
            )

        if args.checkpoint and epoch % args.checkpoint_step == 0:
            checkpoint_path = os.path.join(args.checkpoint_path, '{}_{}.pt'.format(experiment_title, epoch))
            torch.save(model.state_dict(), checkpoint_path)
            checkpoint_paths.append(checkpoint_path)
            current_record['checkpoint_path'] = checkpoint_path

        history_records.append(current_record)
        if args.log:
            logging.info('Epoch %s:', epoch)
            logging.info('Source Train Loss: %.6f', train_loss)
            logging.info('Source Train Accuracy: %.6f', train_acc)
            logging.info('Target Fine-tune Loss: %.6f', target_metrics['loss'])
            logging.info('Target Fine-tune Accuracy: %.6f\n', target_metrics['accuracy'])

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
        'algorithm': 'cnn',
        'history': history_records,
        'history_path': history_path,
        'best_record': best_record,
        'best_state_dict': best_state_dict,
        'best_checkpoint_path': best_checkpoint_path,
        'model_config': get_model_config(model, args),
    }


def build_fixed_deployment_split(dataset, support_pools, query_pools, ways, shots, seed):
    rng = np.random.RandomState(seed)
    available_labels = [
        label for label in sorted(support_pools.keys())
        if len(support_pools[label]) >= shots and len(query_pools[label]) > 0
    ]
    selected_labels = available_labels[:ways]
    support_samples = []
    support_labels = []
    query_samples = []
    query_labels = []

    for new_label, original_label in enumerate(selected_labels):
        sampled_support_indices = rng.choice(support_pools[original_label], size=shots, replace=False)
        for sample_index in sampled_support_indices:
            sample, _ = dataset[int(sample_index)]
            support_samples.append(sample)
            support_labels.append(new_label)

        for sample_index in query_pools[original_label]:
            sample, _ = dataset[int(sample_index)]
            query_samples.append(sample)
            query_labels.append(new_label)

    return (
        torch.stack(support_samples),
        torch.tensor(support_labels, dtype=torch.int64),
        torch.stack(query_samples),
        torch.tensor(query_labels, dtype=torch.int64),
    )


def fine_tune_classifier(model, support_data, support_labels, lr, epochs, device):
    tuned_model = copy.deepcopy(model).to(device)
    tuned_model.train()
    optimizer = torch.optim.Adam(tuned_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    support_data = support_data.to(device)
    support_labels = support_labels.to(device)

    for _ in range(epochs):
        optimizer.zero_grad()
        logits = tuned_model(support_data)
        loss = criterion(logits, support_labels)
        loss.backward()
        optimizer.step()

    return tuned_model.eval()


def evaluate_target_deployment(args, model, test_dataset, test_pools, device):
    support_data, support_labels, query_data, query_labels = build_fixed_deployment_split(
        test_dataset,
        test_pools[0],
        test_pools[1],
        args.ways,
        args.shots,
        args.seed,
    )
    tuned_model = fine_tune_classifier(model, support_data, support_labels, args.finetune_lr, args.finetune_epochs, device)
    query_data = query_data.to(device)
    query_labels = query_labels.to(device)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    with torch.no_grad():
        logits = tuned_model(query_data)
        loss_value = criterion(logits, query_labels).item()
        accuracy_value = accuracy(logits, query_labels).item()
    return {'loss': loss_value, 'accuracy': accuracy_value}


def plot_metrics(args, iteration, train_acc, test_acc, train_loss, test_loss, experiment_title):
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(train_acc, '-o', label='source train acc')
    plt.plot(test_acc, '-o', label='target fine-tune acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve by Epoch')
    plt.legend()
    plt.subplot(122)
    plt.plot(train_loss, '-o', label='source train loss')
    plt.plot(test_loss, '-o', label='target fine-tune loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve by Epoch')
    plt.legend()
    plt.savefig(os.path.join(args.plot_path, '{}_{}.png'.format(experiment_title, iteration)))
    plt.close()
