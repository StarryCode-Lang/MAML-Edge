import copy
import logging
import os
import time

import learn2learn as l2l
import numpy as np
import torch

from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model_layer.models import CNN1D, CNN1DEncoder, CNN2D, CNN2DEncoder
from model_layer.utils import (
    accuracy,
    clone_state_dict_to_cpu,
    deterministic_domain_index,
    deterministic_fixed_pool_episode,
    deterministic_task_sample,
    fast_adapt,
    pairwise_distances_logits,
    protonet_fast_adapt,
    write_json,
)


def run_compression_pipeline(args, algorithm, experiment_title, training_result, train_tasks, test_dataset, test_pools, device):
    artifact_dir = os.path.join(args.compression_output_path, experiment_title)
    os.makedirs(artifact_dir, exist_ok=True)

    base_model = build_model_from_config(training_result['model_config'])
    base_model.load_state_dict(training_result['best_state_dict'])
    base_model.to(device)

    meta_eval_before = evaluate_meta_model(
        algorithm=algorithm,
        args=args,
        model=base_model,
        test_dataset=test_dataset,
        test_pools=test_pools,
        device=device,
    )

    pruned_model, prune_metadata = structured_prune_model(base_model.cpu(), args.prune_ratio)
    pruned_model.to(device)

    recovered_model = recover_pruned_model(
        algorithm=algorithm,
        args=args,
        model=pruned_model,
        train_tasks=train_tasks,
        device=device,
    )

    meta_eval_after = evaluate_meta_model(
        algorithm=algorithm,
        args=args,
        model=recovered_model,
        test_dataset=test_dataset,
        test_pools=test_pools,
        device=device,
    )

    deployment_bundle = build_deployment_bundle(
        algorithm=algorithm,
        args=args,
        model=recovered_model,
        test_dataset=test_dataset,
        test_pools=test_pools,
        device=device,
    )

    float_model_path = os.path.join(artifact_dir, '{}_float.onnx'.format(experiment_title))
    export_deployment_bundle_to_onnx(
        deployment_bundle=deployment_bundle,
        onnx_path=float_model_path,
        opset_version=args.onnx_opset,
    )

    quant_model_path, quant_metrics, quant_warning = quantize_and_evaluate_onnx(
        args=args,
        deployment_bundle=deployment_bundle,
        float_model_path=float_model_path,
        artifact_dir=artifact_dir,
        experiment_title=experiment_title,
    )

    qat_metrics = None
    if should_run_qat(args, deployment_bundle, quant_metrics):
        qat_bundle = run_qat_recovery(deployment_bundle, args)
        qat_float_path = os.path.join(artifact_dir, '{}_qat_float.onnx'.format(experiment_title))
        export_deployment_bundle_to_onnx(
            deployment_bundle=qat_bundle,
            onnx_path=qat_float_path,
            opset_version=args.onnx_opset,
        )
        quant_model_path, quant_metrics, quant_warning = quantize_and_evaluate_onnx(
            args=args,
            deployment_bundle=qat_bundle,
            float_model_path=qat_float_path,
            artifact_dir=artifact_dir,
            experiment_title='{}_qat'.format(experiment_title),
        )
        qat_metrics = evaluate_deployment_bundle(qat_bundle, device=device)
        deployment_bundle = qat_bundle

    summary = {
        'experiment_title': experiment_title,
        'algorithm': algorithm,
        'best_training_record': training_result['best_record'],
        'meta_eval_before_prune': meta_eval_before,
        'meta_eval_after_recovery': meta_eval_after,
        'deployment_float_metrics': evaluate_deployment_bundle(deployment_bundle, device=device),
        'deployment_int8_metrics': quant_metrics,
        'qat_float_metrics': qat_metrics,
        'prune_metadata': prune_metadata,
        'float_model_path': float_model_path,
        'int8_model_path': quant_model_path,
        'prototype_path': deployment_bundle.get('prototype_path'),
        'quantization_warning': quant_warning,
        'model_profile': {
            'baseline_params': count_parameters(base_model),
            'pruned_params': count_parameters(recovered_model),
        },
    }
    write_json(os.path.join(artifact_dir, 'compression_summary.json'), summary)
    logging.info('Compression summary saved to %s.', os.path.join(artifact_dir, 'compression_summary.json'))
    return summary


def build_model_from_config(model_config):
    model_type = model_config['model_type']
    if model_type == 'CNN1D':
        return CNN1D(
            output_size=model_config['output_size'],
            channels=model_config.get('channels'),
            pooled_length=model_config.get('pooled_length', 64),
        )
    if model_type == 'CNN2D':
        return CNN2D(
            output_size=model_config['output_size'],
            in_channels=model_config.get('in_channels', 3),
            channels=model_config.get('channels'),
        )
    if model_type == 'CNN1DEncoder':
        return CNN1DEncoder(
            channels=model_config.get('channels'),
            pooled_length=model_config.get('pooled_length', 64),
        )
    if model_type == 'CNN2DEncoder':
        return CNN2DEncoder(
            in_channels=model_config.get('in_channels', 3),
            channels=model_config.get('channels'),
        )
    raise ValueError('Unsupported model type: {}'.format(model_type))


def count_parameters(model):
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _channel_keep_indices(conv_module, prune_ratio):
    output_channels = conv_module.out_channels
    keep_channels = max(1, int(round(output_channels * (1.0 - prune_ratio))))
    keep_channels = min(output_channels, keep_channels)
    scores = conv_module.weight.detach().abs().mean(dim=tuple(range(1, conv_module.weight.dim())))
    keep_indices = torch.topk(scores, k=keep_channels, largest=True).indices
    return keep_indices.sort().values


def _copy_batch_norm(new_bn, old_bn, keep_indices):
    new_bn.weight.data.copy_(old_bn.weight.data[keep_indices])
    new_bn.bias.data.copy_(old_bn.bias.data[keep_indices])
    new_bn.running_mean.data.copy_(old_bn.running_mean.data[keep_indices])
    new_bn.running_var.data.copy_(old_bn.running_var.data[keep_indices])


def _copy_conv1d(new_conv, old_conv, out_indices, in_indices=None):
    weight = old_conv.weight.data.clone()
    if in_indices is not None:
        weight = weight[:, in_indices, :]
    weight = weight[out_indices, :, :]
    new_conv.weight.data.copy_(weight)
    if old_conv.bias is not None:
        new_conv.bias.data.copy_(old_conv.bias.data[out_indices])


def _copy_conv2d(new_conv, old_conv, out_indices, in_indices=None):
    weight = old_conv.weight.data.clone()
    if in_indices is not None:
        weight = weight[:, in_indices, :, :]
    weight = weight[out_indices, :, :, :]
    new_conv.weight.data.copy_(weight)
    if old_conv.bias is not None:
        new_conv.bias.data.copy_(old_conv.bias.data[out_indices])


def _copy_linear_input_features(new_linear, old_linear, feature_indices):
    new_linear.weight.data.copy_(old_linear.weight.data[:, feature_indices])
    if old_linear.bias is not None:
        new_linear.bias.data.copy_(old_linear.bias.data)


def prune_cnn1d_encoder(model, prune_ratio):
    keep_indices_1 = _channel_keep_indices(model.layer1[0], prune_ratio)
    keep_indices_2 = _channel_keep_indices(model.layer2[0], prune_ratio)
    keep_indices_3 = _channel_keep_indices(model.layer3[0], prune_ratio)

    pruned_model = CNN1DEncoder(
        channels=[len(keep_indices_1), len(keep_indices_2), len(keep_indices_3)],
        pooled_length=model.pooled_length,
    )
    _copy_conv1d(pruned_model.layer1[0], model.layer1[0], keep_indices_1)
    _copy_batch_norm(pruned_model.layer1[1], model.layer1[1], keep_indices_1)

    _copy_conv1d(pruned_model.layer2[0], model.layer2[0], keep_indices_2, in_indices=keep_indices_1)
    _copy_batch_norm(pruned_model.layer2[1], model.layer2[1], keep_indices_2)

    _copy_conv1d(pruned_model.layer3[0], model.layer3[0], keep_indices_3, in_indices=keep_indices_2)
    _copy_batch_norm(pruned_model.layer3[1], model.layer3[1], keep_indices_3)
    return pruned_model, {'kept_channels': [len(keep_indices_1), len(keep_indices_2), len(keep_indices_3)], 'last_indices': keep_indices_3}


def prune_cnn1d_classifier(model, prune_ratio):
    pruned_encoder, metadata = prune_cnn1d_encoder(model.encoder, prune_ratio)
    pruned_model = CNN1D(
        output_size=model.fc.out_features,
        channels=pruned_encoder.channels,
        pooled_length=pruned_encoder.pooled_length,
    )
    pruned_model.encoder.load_state_dict(pruned_encoder.state_dict())

    feature_indices = []
    pooled_length = model.encoder.pooled_length
    for channel_index in metadata['last_indices'].tolist():
        start_index = channel_index * pooled_length
        feature_indices.extend(range(start_index, start_index + pooled_length))
    feature_indices = torch.tensor(feature_indices, dtype=torch.long)
    _copy_linear_input_features(pruned_model.fc, model.fc, feature_indices)
    return pruned_model, {'kept_channels': metadata['kept_channels']}


def prune_cnn2d_encoder(model, prune_ratio):
    original_blocks = model.features
    keep_indices = [_channel_keep_indices(block[0], prune_ratio) for block in original_blocks]
    pruned_model = CNN2DEncoder(
        in_channels=model.in_channels,
        channels=[len(indices) for indices in keep_indices],
    )

    previous_indices = None
    for block_index, (new_block, old_block, current_indices) in enumerate(zip(pruned_model.features, original_blocks, keep_indices)):
        _copy_conv2d(new_block[0], old_block[0], current_indices, in_indices=previous_indices)
        _copy_batch_norm(new_block[1], old_block[1], current_indices)
        previous_indices = current_indices
    return pruned_model, {'kept_channels': [len(indices) for indices in keep_indices], 'last_indices': keep_indices[-1]}


def prune_cnn2d_classifier(model, prune_ratio):
    pruned_encoder, metadata = prune_cnn2d_encoder(model.encoder, prune_ratio)
    pruned_model = CNN2D(
        output_size=model.fc.out_features,
        in_channels=model.encoder.in_channels,
        channels=pruned_encoder.channels,
    )
    pruned_model.encoder.load_state_dict(pruned_encoder.state_dict())
    _copy_linear_input_features(pruned_model.fc, model.fc, metadata['last_indices'])
    return pruned_model, {'kept_channels': metadata['kept_channels']}


def structured_prune_model(model, prune_ratio):
    if isinstance(model, CNN1D):
        return prune_cnn1d_classifier(model, prune_ratio)
    if isinstance(model, CNN1DEncoder):
        return prune_cnn1d_encoder(model, prune_ratio)
    if isinstance(model, CNN2D):
        return prune_cnn2d_classifier(model, prune_ratio)
    if isinstance(model, CNN2DEncoder):
        return prune_cnn2d_encoder(model, prune_ratio)
    raise ValueError('Unsupported model class for pruning: {}'.format(type(model).__name__))


def recover_pruned_model(algorithm, args, model, train_tasks, device):
    if algorithm == 'maml':
        return recover_pruned_maml_model(args, model, train_tasks, device)
    if algorithm == 'protonet':
        return recover_pruned_protonet_model(args, model, train_tasks, device)
    if algorithm == 'cnn':
        return recover_pruned_cnn_model(args, model, train_tasks, device)
    raise ValueError('Unsupported algorithm: {}'.format(algorithm))


def recover_pruned_maml_model(args, model, train_tasks, device):
    model = model.to(device)
    model.train()
    maml_wrapper = l2l.algorithms.MAML(model, lr=args.fast_lr, first_order=args.first_order)
    optimizer = torch.optim.Adam(model.parameters(), args.meta_lr)
    loss = nn.CrossEntropyLoss(reduction='mean')

    for iteration in range(1, args.compression_finetune_iters + 1):
        optimizer.zero_grad()
        for episode in range(args.compression_meta_batch_size):
            train_index = deterministic_domain_index(args.seed + 700000, iteration, episode, len(train_tasks))
            learner = maml_wrapper.clone()
            batch_seed = args.seed + 91000000 + iteration * 100000 + episode
            batch = deterministic_task_sample(train_tasks[train_index], batch_seed)
            train_error, _ = fast_adapt(
                batch,
                learner,
                loss,
                args.adapt_steps,
                args.shots,
                args.ways,
                device,
                args.query_shots,
            )
            train_error.backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.data.mul_(1.0 / args.compression_meta_batch_size)
        optimizer.step()
    return model.eval()


def recover_pruned_protonet_model(args, model, train_tasks, device):
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), args.meta_lr)
    loss = nn.CrossEntropyLoss(reduction='mean')

    for iteration in range(1, args.compression_finetune_iters + 1):
        optimizer.zero_grad()
        for episode in range(args.compression_meta_batch_size):
            train_index = deterministic_domain_index(args.seed + 700000, iteration, episode, len(train_tasks))
            batch_seed = args.seed + 91000000 + iteration * 100000 + episode
            batch = deterministic_task_sample(train_tasks[train_index], batch_seed)
            train_error, _ = protonet_fast_adapt(
                batch,
                model,
                loss,
                args.ways,
                args.shots,
                args.query_shots,
                device,
            )
            train_error.backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.data.mul_(1.0 / args.compression_meta_batch_size)
        optimizer.step()
    return model.eval()


def recover_pruned_cnn_model(args, model, train_loader, device):
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=getattr(args, 'lr', 1e-3))
    criterion = nn.CrossEntropyLoss(reduction='mean')
    train_iterator = iter(train_loader)

    for _ in range(args.compression_finetune_iters):
        try:
            batch_data, batch_labels = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch_data, batch_labels = next(train_iterator)
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        batch_loss = criterion(model(batch_data), batch_labels)
        batch_loss.backward()
        optimizer.step()
    return model.eval()


def evaluate_meta_model(algorithm, args, model, test_dataset, test_pools, device):
    model = model.to(device)
    loss = nn.CrossEntropyLoss(reduction='mean')
    error_sum = 0.0
    accuracy_sum = 0.0

    if algorithm == 'maml':
        model.train()
        maml_wrapper = l2l.algorithms.MAML(model, lr=args.fast_lr, first_order=args.first_order)
        for episode in range(args.test_task_num):
            batch_seed = args.seed + 120000000 + episode
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
            episode_error, episode_accuracy = fast_adapt(
                batch,
                learner,
                loss,
                args.adapt_steps,
                args.shots,
                args.ways,
                device,
                args.query_shots,
            )
            error_sum += episode_error.item()
            accuracy_sum += episode_accuracy.item()
    elif algorithm == 'protonet':
        with torch.no_grad():
            for episode in range(args.test_task_num):
                batch_seed = args.seed + 120000000 + episode
                batch = deterministic_fixed_pool_episode(
                    test_dataset,
                    test_pools[0],
                    test_pools[1],
                    args.ways,
                    args.shots,
                    args.query_shots,
                    batch_seed,
                )
                episode_error, episode_accuracy = protonet_fast_adapt(
                    batch,
                    model,
                    loss,
                    args.ways,
                    args.shots,
                    args.query_shots,
                    device,
                )
                error_sum += episode_error.item()
                accuracy_sum += episode_accuracy.item()
    elif algorithm == 'cnn':
        deployment_bundle = build_deployment_bundle(algorithm, args, model, test_dataset, test_pools, device)
        return evaluate_deployment_bundle(deployment_bundle, device)
    else:
        raise ValueError('Unsupported algorithm: {}'.format(algorithm))

    return {
        'loss': error_sum / args.test_task_num,
        'accuracy': accuracy_sum / args.test_task_num,
    }


def build_deployment_bundle(algorithm, args, model, test_dataset, test_pools, device):
    support_data, support_labels, query_data, query_labels, selected_labels = build_fixed_deployment_split(
        dataset=test_dataset,
        support_pools=test_pools[0],
        query_pools=test_pools[1],
        ways=args.ways,
        shots=args.shots,
        seed=args.seed,
    )
    if algorithm == 'maml':
        deployment_model = adapt_maml_for_deployment(args, model, support_data, support_labels, device)
        return {
            'deployment_type': 'classifier',
            'algorithm': algorithm,
            'model': deployment_model.cpu().eval(),
            'model_config': infer_deployment_model_config(deployment_model),
            'support_data': support_data.cpu(),
            'support_labels': support_labels.cpu(),
            'query_data': query_data.cpu(),
            'query_labels': query_labels.cpu(),
            'selected_labels': selected_labels,
        }

    if algorithm == 'protonet':
        prototypes = build_prototypes(model, support_data, args.ways, args.shots, device)
        prototype_path = os.path.join(args.compression_output_path, 'tmp_prototypes.npz')
        return {
            'deployment_type': 'encoder_with_prototypes',
            'algorithm': algorithm,
            'model': copy.deepcopy(model).cpu().eval(),
            'model_config': infer_deployment_model_config(model),
            'support_data': support_data.cpu(),
            'support_labels': support_labels.cpu(),
            'query_data': query_data.cpu(),
            'query_labels': query_labels.cpu(),
            'selected_labels': selected_labels,
            'prototypes': prototypes.cpu(),
            'prototype_path': prototype_path,
        }

    if algorithm == 'cnn':
        deployment_model = fine_tune_classifier_for_deployment(
            model=model,
            support_data=support_data,
            support_labels=support_labels,
            lr=getattr(args, 'finetune_lr', 5e-4),
            epochs=getattr(args, 'finetune_epochs', 20),
            device=device,
        )
        return {
            'deployment_type': 'classifier',
            'algorithm': algorithm,
            'model': deployment_model.cpu().eval(),
            'model_config': infer_deployment_model_config(deployment_model),
            'support_data': support_data.cpu(),
            'support_labels': support_labels.cpu(),
            'query_data': query_data.cpu(),
            'query_labels': query_labels.cpu(),
            'selected_labels': selected_labels,
        }

    raise ValueError('Unsupported algorithm: {}'.format(algorithm))


def infer_deployment_model_config(model):
    if isinstance(model, CNN1D):
        return {
            'model_type': 'CNN1D',
            'output_size': model.fc.out_features,
            'channels': list(model.encoder.channels),
            'pooled_length': model.encoder.pooled_length,
        }
    if isinstance(model, CNN2D):
        return {
            'model_type': 'CNN2D',
            'output_size': model.fc.out_features,
            'channels': list(model.encoder.channels),
            'in_channels': model.encoder.in_channels,
        }
    if isinstance(model, CNN1DEncoder):
        return {
            'model_type': 'CNN1DEncoder',
            'channels': list(model.channels),
            'pooled_length': model.pooled_length,
        }
    if isinstance(model, CNN2DEncoder):
        return {
            'model_type': 'CNN2DEncoder',
            'channels': list(model.channels),
            'in_channels': model.in_channels,
        }
    raise ValueError('Unsupported deployment model type: {}'.format(type(model).__name__))


def build_fixed_deployment_split(dataset, support_pools, query_pools, ways, shots, seed):
    rng = np.random.RandomState(seed)
    available_labels = [
        label for label in sorted(support_pools.keys())
        if len(support_pools[label]) >= shots and len(query_pools[label]) > 0
    ]
    if len(available_labels) < ways:
        raise ValueError('Not enough labels for deployment split.')

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
        selected_labels,
    )


def adapt_maml_for_deployment(args, model, support_data, support_labels, device):
    base_model = copy.deepcopy(model).to(device).train()
    learner = l2l.algorithms.MAML(base_model, lr=args.fast_lr, first_order=args.first_order).clone()
    loss = nn.CrossEntropyLoss(reduction='mean')
    support_data = support_data.to(device)
    support_labels = support_labels.to(device)
    for _ in range(args.adapt_steps):
        adaptation_loss = loss(learner(support_data), support_labels)
        learner.adapt(adaptation_loss)
    deployment_model = build_model_from_config(infer_deployment_model_config(learner.module))
    deployment_model.load_state_dict(clone_state_dict_to_cpu(learner.module))
    deployment_model = recalibrate_batch_norm(
        model=deployment_model,
        calibration_data=support_data.detach().cpu(),
        device=device,
        passes=max(8, args.adapt_steps * 2),
    )
    return deployment_model.cpu().eval()


def fine_tune_classifier_for_deployment(model, support_data, support_labels, lr, epochs, device):
    tuned_model = copy.deepcopy(model).to(device)
    tuned_model.train()
    optimizer = torch.optim.Adam(tuned_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    support_data = support_data.to(device)
    support_labels = support_labels.to(device)

    for _ in range(epochs):
        optimizer.zero_grad()
        support_loss = criterion(tuned_model(support_data), support_labels)
        support_loss.backward()
        optimizer.step()

    return tuned_model.cpu().eval()


def build_prototypes(model, support_data, ways, shots, device):
    encoder = copy.deepcopy(model).to(device).eval()
    with torch.no_grad():
        embeddings = encoder(support_data.to(device))
        prototypes = embeddings.reshape(ways, shots, -1).mean(dim=1)
    return prototypes.cpu()


def recalibrate_batch_norm(model, calibration_data, device, passes=8):
    batch_norm_layers = [
        module for module in model.modules()
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d))
    ]
    if not batch_norm_layers:
        return model

    model = model.to(device)
    original_momenta = {}
    for module in batch_norm_layers:
        original_momenta[module] = module.momentum
        module.reset_running_stats()
        module.momentum = None

    model.train()
    calibration_data = calibration_data.to(device)
    with torch.no_grad():
        for _ in range(passes):
            _ = model(calibration_data)

    for module in batch_norm_layers:
        module.momentum = original_momenta[module]

    model.eval()
    return model


def evaluate_deployment_bundle(deployment_bundle, device):
    deployment_type = deployment_bundle['deployment_type']
    if deployment_type == 'classifier':
        return evaluate_classifier(deployment_bundle['model'], deployment_bundle['query_data'], deployment_bundle['query_labels'], device)
    if deployment_type == 'encoder_with_prototypes':
        return evaluate_encoder_with_prototypes(
            deployment_bundle['model'],
            deployment_bundle['prototypes'],
            deployment_bundle['query_data'],
            deployment_bundle['query_labels'],
            device,
        )
    raise ValueError('Unsupported deployment type: {}'.format(deployment_type))


def evaluate_classifier(model, query_data, query_labels, device):
    model = copy.deepcopy(model).to(device).eval()
    query_data = query_data.to(device)
    query_labels = query_labels.to(device)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    with torch.no_grad():
        logits = model(query_data)
        loss_value = criterion(logits, query_labels).item()
        accuracy_value = accuracy(logits, query_labels).item()
    return {
        'loss': loss_value,
        'accuracy': accuracy_value,
    }


def evaluate_encoder_with_prototypes(model, prototypes, query_data, query_labels, device):
    model = copy.deepcopy(model).to(device).eval()
    prototypes = prototypes.to(device)
    query_data = query_data.to(device)
    query_labels = query_labels.to(device)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    with torch.no_grad():
        query_embeddings = model(query_data)
        logits = pairwise_distances_logits(query_embeddings, prototypes)
        loss_value = criterion(logits, query_labels).item()
        accuracy_value = accuracy(logits, query_labels).item()
    return {
        'loss': loss_value,
        'accuracy': accuracy_value,
    }


def export_deployment_bundle_to_onnx(deployment_bundle, onnx_path, opset_version):
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    model = deployment_bundle['model'].cpu().eval()
    example_input = deployment_bundle['support_data'][:1].cpu()
    torch.onnx.export(
        model,
        example_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=opset_version,
    )
    if deployment_bundle['deployment_type'] == 'encoder_with_prototypes':
        prototype_path = os.path.splitext(onnx_path)[0] + '_prototypes.npz'
        np.savez(
            prototype_path,
            prototypes=deployment_bundle['prototypes'].cpu().numpy(),
            selected_labels=np.asarray(deployment_bundle['selected_labels']),
        )
        deployment_bundle['prototype_path'] = prototype_path


class NumpyCalibrationDataReader:
    def __init__(self, batches):
        self._batches = iter(batches)

    def get_next(self):
        return next(self._batches, None)


def quantize_and_evaluate_onnx(args, deployment_bundle, float_model_path, artifact_dir, experiment_title):
    try:
        from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    except ImportError as exc:
        warning = 'onnxruntime quantization is unavailable: {}'.format(exc)
        logging.warning(warning)
        return None, None, warning

    calibration_input = collect_calibration_batches(
        support_data=deployment_bundle['support_data'],
        query_data=deployment_bundle['query_data'],
        calibration_size=args.calibration_size,
    )
    quant_model_path = os.path.join(artifact_dir, '{}_int8.onnx'.format(experiment_title))
    quantize_static(
        model_input=float_model_path,
        model_output=quant_model_path,
        calibration_data_reader=NumpyCalibrationDataReader(calibration_input),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
    )
    metrics = evaluate_onnx_bundle(quant_model_path, deployment_bundle)
    return quant_model_path, metrics, None


def collect_calibration_batches(support_data, query_data, calibration_size):
    combined = torch.cat([support_data, query_data], dim=0)
    limited = combined[:max(1, calibration_size)]
    return [{'input': limited[index:index + 1].cpu().numpy()} for index in range(limited.size(0))]


def evaluate_onnx_bundle(onnx_path, deployment_bundle):
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    query_array = deployment_bundle['query_data'].cpu().numpy()
    query_labels = deployment_bundle['query_labels'].cpu().numpy()

    outputs = []
    total_time = 0.0
    for sample in query_array:
        feed_dict = {'input': sample[np.newaxis, ...]}
        start_time = time.perf_counter()
        session_output = session.run(None, feed_dict)[0]
        total_time += (time.perf_counter() - start_time)
        outputs.append(session_output[0])

    outputs = np.asarray(outputs)
    if deployment_bundle['deployment_type'] == 'classifier':
        predicted_labels = outputs.argmax(axis=1)
        loss_value = softmax_cross_entropy(outputs, query_labels)
    else:
        prototypes = deployment_bundle['prototypes'].cpu().numpy()
        distances = ((outputs[:, np.newaxis, :] - prototypes[np.newaxis, :, :]) ** 2).sum(axis=2)
        logits = -distances
        predicted_labels = logits.argmax(axis=1)
        loss_value = softmax_cross_entropy(logits, query_labels)

    return {
        'loss': float(loss_value),
        'accuracy': float((predicted_labels == query_labels).mean()),
        'avg_latency_ms': float((total_time / max(1, len(query_array))) * 1000.0),
    }


def softmax_cross_entropy(logits, labels):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    row_indices = np.arange(labels.shape[0])
    return -np.log(np.clip(probabilities[row_indices, labels], 1e-12, 1.0)).mean()


def should_run_qat(args, deployment_bundle, quant_metrics):
    if not args.enable_qat_recovery:
        return False
    if quant_metrics is None:
        return False
    if deployment_bundle['deployment_type'] != 'classifier':
        return False
    float_metrics = evaluate_deployment_bundle(deployment_bundle, device=torch.device('cpu'))
    return (float_metrics['accuracy'] - quant_metrics['accuracy']) > args.qat_drop_threshold


def run_qat_recovery(deployment_bundle, args):
    from torch.ao.quantization import get_default_qat_qconfig_mapping
    from torch.ao.quantization.quantize_fx import prepare_qat_fx

    model = copy.deepcopy(deployment_bundle['model']).cpu().train()
    example_inputs = (deployment_bundle['support_data'][:1].cpu(),)
    qconfig_mapping = get_default_qat_qconfig_mapping('fbgemm')
    prepared_model = prepare_qat_fx(model, qconfig_mapping, example_inputs)
    optimizer = torch.optim.Adam(prepared_model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    support_loader = DataLoader(
        TensorDataset(deployment_bundle['support_data'].cpu(), deployment_bundle['support_labels'].cpu()),
        batch_size=min(8, deployment_bundle['support_data'].size(0)),
        shuffle=True,
    )

    for _ in range(args.qat_recovery_epochs):
        for batch_data, batch_labels in support_loader:
            optimizer.zero_grad()
            batch_loss = criterion(prepared_model(batch_data), batch_labels)
            batch_loss.backward()
            optimizer.step()

    recovered_model = build_model_from_config(deployment_bundle['model_config'])
    recovered_model.load_state_dict(prepared_model.state_dict(), strict=False)
    qat_bundle = dict(deployment_bundle)
    qat_bundle['model'] = recovered_model.eval()
    return qat_bundle
