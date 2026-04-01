import json
import logging
import os
import random

import h5py
import pywt
import torch

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import stft
from PIL import Image


DEFAULT_FAULT_LABELS = {
    'CWRU': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'HST': [0, 2, 3, 5, 6],
}


def resolve_fault_labels(dataset_name, fault_labels=None):
    if fault_labels is None or fault_labels == '':
        return list(DEFAULT_FAULT_LABELS[dataset_name])
    if isinstance(fault_labels, str):
        values = [item.strip() for item in fault_labels.split(',') if item.strip()]
        return [int(item) for item in values]
    return [int(item) for item in fault_labels]


def clone_state_dict_to_cpu(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def is_better_model_record(candidate_record, best_record):
    if best_record is None:
        return True
    candidate_score = (
        candidate_record['meta_test_acc'],
        -candidate_record['meta_test_loss'],
        candidate_record['meta_train_acc'],
        -candidate_record['iteration'],
    )
    best_score = (
        best_record['meta_test_acc'],
        -best_record['meta_test_loss'],
        best_record['meta_train_acc'],
        -best_record['iteration'],
    )
    return candidate_score > best_score


def write_json(path, payload):
    with open(path, 'w', encoding='utf-8') as file_pointer:
        json.dump(payload, file_pointer, indent=2, ensure_ascii=False)


def get_dataset_labels(dataset):
    if hasattr(dataset, 'labels'):
        return np.asarray(dataset.labels)
    if hasattr(dataset, 'img_list'):
        return np.asarray([int(img_name.split('_')[0]) for img_name in dataset.img_list])
    raise ValueError('Unable to infer labels from dataset type.')


def split_support_query(data, labels, ways, shots, query_shots):
    sort = torch.sort(labels)
    data = data[sort.indices]
    labels = labels[sort.indices]

    support_indices = np.zeros(data.size(0), dtype=bool)
    selection = np.arange(ways) * (shots + query_shots)
    for offset in range(shots):
        support_indices[selection + offset] = True

    support_indices = torch.from_numpy(support_indices)
    query_indices = ~support_indices
    support_data = data[support_indices]
    support_labels = labels[support_indices]
    query_data = data[query_indices]
    query_labels = labels[query_indices]
    return support_data, support_labels, query_data, query_labels


def create_class_pools(dataset, support_ratio=0.5):
    labels = get_dataset_labels(dataset)
    unique_labels = sorted(set(labels.tolist()))
    support_pools = {}
    query_pools = {}

    for label in unique_labels:
        class_indices = np.where(labels == label)[0]
        split_index = max(1, int(len(class_indices) * support_ratio))
        split_index = min(split_index, len(class_indices) - 1)
        if split_index <= 0:
            raise ValueError('Each class must contain at least two samples for strict evaluation.')
        support_pools[label] = class_indices[:split_index].tolist()
        query_pools[label] = class_indices[split_index:].tolist()

    return support_pools, query_pools


def sample_fixed_pool_episode(dataset, support_pools, query_pools, ways, shots, query_shots):
    available_labels = [label for label in sorted(support_pools.keys())
                        if len(support_pools[label]) >= shots and len(query_pools[label]) >= query_shots]
    if len(available_labels) < ways:
        raise ValueError('Not enough classes with sufficient support/query samples for evaluation.')

    selected_labels = np.random.choice(available_labels, size=ways, replace=False)
    batch_samples = []
    batch_labels = []

    for new_label, original_label in enumerate(selected_labels):
        support_indices = np.random.choice(support_pools[original_label], size=shots, replace=False)
        query_indices = np.random.choice(query_pools[original_label], size=query_shots, replace=False)
        for index in np.concatenate([support_indices, query_indices]):
            sample, _ = dataset[int(index)]
            batch_samples.append(sample)
            batch_labels.append(new_label)

    return torch.stack(batch_samples), torch.tensor(batch_labels, dtype=torch.int64)


def deterministic_domain_index(seed, iteration, episode, domain_count):
    rng = np.random.RandomState(seed + iteration * 1000 + episode)
    return int(rng.randint(domain_count))


def deterministic_task_sample(taskset, seed):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    batch = taskset.sample()
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.random.set_rng_state(torch_state)
    return batch


def deterministic_fixed_pool_episode(dataset, support_pools, query_pools, ways, shots, query_shots, seed):
    rng = np.random.RandomState(seed)
    available_labels = [label for label in sorted(support_pools.keys())
                        if len(support_pools[label]) >= shots and len(query_pools[label]) >= query_shots]
    if len(available_labels) < ways:
        raise ValueError('Not enough classes with sufficient support/query samples for evaluation.')

    selected_labels = rng.choice(available_labels, size=ways, replace=False)
    batch_samples = []
    batch_labels = []

    for new_label, original_label in enumerate(selected_labels):
        support_indices = rng.choice(support_pools[original_label], size=shots, replace=False)
        query_indices = rng.choice(query_pools[original_label], size=query_shots, replace=False)
        for index in np.concatenate([support_indices, query_indices]):
            sample, _ = dataset[int(index)]
            batch_samples.append(sample)
            batch_labels.append(new_label)

    return torch.stack(batch_samples), torch.tensor(batch_labels, dtype=torch.int64)


def deterministic_fixed_pool_episode_split(dataset, support_pools, query_pools, ways, shots, query_shots, seed):
    rng = np.random.RandomState(seed)
    available_labels = [label for label in sorted(support_pools.keys())
                        if len(support_pools[label]) >= shots and len(query_pools[label]) >= query_shots]
    if len(available_labels) < ways:
        raise ValueError('Not enough classes with sufficient support/query samples for evaluation.')

    selected_labels = rng.choice(available_labels, size=ways, replace=False)
    support_samples = []
    support_labels = []
    query_samples = []
    query_labels = []

    for new_label, original_label in enumerate(selected_labels):
        support_indices = rng.choice(support_pools[original_label], size=shots, replace=False)
        query_indices = rng.choice(query_pools[original_label], size=query_shots, replace=False)
        for index in support_indices:
            sample, _ = dataset[int(index)]
            support_samples.append(sample)
            support_labels.append(new_label)
        for index in query_indices:
            sample, _ = dataset[int(index)]
            query_samples.append(sample)
            query_labels.append(new_label)

    return (
        torch.stack(support_samples),
        torch.tensor(support_labels, dtype=torch.int64),
        torch.stack(query_samples),
        torch.tensor(query_labels, dtype=torch.int64),
        selected_labels.tolist(),
    )


def setup_logger(log_path, experiment_title):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(message)s",
                                   datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    fh = logging.FileHandler(os.path.join(log_path, 
                                          experiment_title + '.log'))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device, query_shots=None):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)
    if query_shots is None:
        query_shots = shots

    adaptation_data, adaptation_labels, evaluation_data, evaluation_labels = split_support_query(
        data, labels, ways, shots, query_shots
    )

    # Adapt the model
    for step in range(adaptation_steps):
        train_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(train_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy


def protonet_fast_adapt(batch, model, loss, ways, shots, query_shots, device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    support_data, _, query_data, query_labels = split_support_query(data, labels, ways, shots, query_shots)
    support_embeddings = model(support_data)
    query_embeddings = model(query_data)
    support = support_embeddings.reshape(ways, shots, -1).mean(dim=1)
    query = query_embeddings
    query_labels = query_labels.long()

    logits = pairwise_distances_logits(query, support)
    valid_error = loss(logits, query_labels)
    valid_accuracy = accuracy(logits, query_labels)
    return valid_error, valid_accuracy


def pairwise_distances_logits(a, b):
    n = a.shape[0]
    m = b.shape[0]
    logits = -((a.unsqueeze(1).expand(n, m, -1) -
                b.unsqueeze(0).expand(n, m, -1))**2).sum(dim=2)
    return logits

def print_logs(iteration, meta_train_error, meta_train_accuracy, meta_test_error, meta_test_accuracy):
    logging.info('Iteration {}:'.format(iteration))
    logging.info('Meta Train Results:')
    logging.info('Meta Train Error: {}.'.format(meta_train_error))
    logging.info('Meta Train Accuracy: {}.'.format(meta_train_accuracy))
    logging.info('Meta Test Results:')
    logging.info('Meta Test Error: {}.'.format(meta_test_error))
    logging.info('Meta Test Accuracy: {}.\n'.format(meta_test_accuracy))

def normalize(data):
    return (data-min(data)) / (max(data)-min(data))



def make_time_frequency_image_STFT(dataset_name, 
                              dataset, 
                              img_size, 
                              window_size, 
                              overlap, 
                              img_path):

    overlap_samples = int(window_size * overlap)
    
    frequency, time, magnitude = stft(dataset, nperseg=window_size, noverlap=overlap_samples)
    
    if dataset_name == 'HST':
        magnitude = np.log10(np.abs(magnitude) + 1e-10)
    else:
        magnitude = np.abs(magnitude)
    # Image Plotting
    plt.pcolormesh(time, frequency, magnitude, shading='gouraud')
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gcf().set_size_inches(img_size/100, img_size/100)
    plt.savefig(img_path, dpi=100)
    plt.clf()
    plt.close()


def make_time_frequency_image_WT(dataset_name,
                                 data,
                                 img_size,
                                 img_path):
    # Data Length
    sampling_length = len(data)
    # Wavelet Transform Parameters Setting
    if dataset_name == 'CWRU':
        sampling_period  = 1.0 / 12000
        total_scale = 128
        wavelet = 'cmor100-1'
    elif dataset_name == 'HST':
        sampling_period = 4e-6
        total_scale = 16
        wavelet = 'morl'
    else:
        raise ValueError("Invalid dataset name")
    fc = pywt.central_frequency(wavelet)
    cparam = 2 * fc * total_scale
    scales = cparam / np.arange(total_scale, 0, -1)
    # Conduct Wavelet Transform
    coefficients, frequencies = pywt.cwt(data, scales, wavelet, sampling_period)
    amplitude = abs(coefficients)
    if dataset_name == 'HST':
        amplitude = np.log10(amplitude + 1e-4)
    # Image Plotting
    t = np.linspace(0, sampling_period, sampling_length, endpoint=False)
    plt.contourf(t, frequencies, amplitude, cmap='jet')
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator()) 
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gcf().set_size_inches(img_size/100, img_size/100)
    plt.savefig(img_path, dpi=100)
    plt.clf()
    plt.close()


def generate_time_frequency_image_dataset(dataset_name, 
                                          algorithm,
                                          dataset, 
                                          labels, 
                                          img_size, 
                                          window_size, 
                                          overlap, 
                                          img_dir):
    for index in range(len(labels)):
        count = 0
        for i, data in enumerate(dataset[labels[index]]):
            os.makedirs(img_dir, exist_ok=True)
            img_path = img_dir + str(index) + "_" + str(count)
            if algorithm == 'STFT':
                make_time_frequency_image_STFT(dataset_name, 
                                               data, 
                                               img_size, 
                                               window_size, 
                                               overlap, 
                                               img_path)
            elif algorithm == 'WT': 
                make_time_frequency_image_WT(dataset_name, 
                                             data, 
                                             img_size, 
                                             img_path)
            else:
                raise ValueError("Invalid algorithm name")
            count += 1
    image_list = os.listdir(img_dir)
    for image_name in image_list:
        image_path = os.path.join(img_dir, image_name)
        img = Image.open(image_path)
        img = img.convert('RGB')
        img.save(image_path)


def loadmat_v73(data_path, realaxis, channel):
    with h5py.File(data_path, 'r') as f:
        mat_data = f[f[realaxis]['Y']['Data'][channel][0]]
        return mat_data[:].reshape(-1)
    

def extract_dict_data(dataset):
    x = np.concatenate([dataset[key] for key in dataset.keys()])
    y = []
    for i, key in enumerate(dataset.keys()):
        number = len(dataset[key])
        y.append(np.tile(i, number))
    y = np.concatenate(y)
    return x, y



if __name__ == '__main__':
    pass
