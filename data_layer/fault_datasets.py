"""
PyTorch dataset classes for CWRU and HST datasets.
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

from .preprocess_cwru import load_CWRU_dataset
from .preprocess_hst import load_HST_dataset
from model_layer.utils import extract_dict_data


def _normalize_label_subset(label_subset, default_labels):
    if label_subset is None:
        return list(default_labels)
    return [int(label) for label in label_subset]

class CWRU(Dataset):

    def __init__(self, 
                 domain,
                 dir_path, 
                 preprocess,
                 label_subset=None,
                 transform=None):

        super(CWRU, self).__init__()
        if domain not in [0, 1, 2, 3]:
            raise ValueError('Argument "domain" must be 0, 1, 2 or 3.')
        self.domain = domain
        self.dir_path = dir_path
        self.label_subset = _normalize_label_subset(label_subset, range(10))
        self.label_mapping = {
            original_label: remapped_label for remapped_label, original_label in enumerate(self.label_subset)
        }
        if preprocess != 'FFT':
            self.img_dir = dir_path + "/{}_CWRU/Drive_end_".format(preprocess) + str(domain) + "/"
        else:
            self.img_dir = dir_path + "/CWRU/Drive_end_" + str(domain) + "/"
        self.img_list = [
            image_name for image_name in os.listdir(self.img_dir)
            if int(image_name.split('_')[0]) in self.label_mapping
        ]

        if transform is None:
            self.transform = transforms.Compose(
            [
                transforms.ToTensor(), 
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img_name = self.img_list[index]
        original_label = int(img_name.split('_')[0])
        label = torch.tensor(self.label_mapping[original_label], dtype=torch.int64)
        img_path = self.img_dir + img_name
        img = Image.open(img_path)
        img = self.transform(img)
        return img, label

class CWRU_FFT(Dataset):

    def __init__(self, 
                 domain,
                 dir_path,
                 label_subset=None,
                 fft=True):
        super(CWRU_FFT, self).__init__()
        self.root = dir_path

        if domain not in [0, 1, 2, 3]:
            raise ValueError('Argument "domain" must be 0, 1, 2 or 3.')
        self.domain = domain
        self.label_subset = _normalize_label_subset(label_subset, range(10))
        self.dataset = load_CWRU_dataset(domain, dir_path, labels=self.label_subset, raw=True, fft=fft)
        self.data, self.labels = extract_dict_data(self.dataset)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]

        sample = torch.from_numpy(sample).float()
        label = torch.tensor(label)
        return sample, label


class HST(Dataset):

    def __init__(self, 
                 domain,
                 dir_path, 
                 preprocess,
                 label_subset=None,
                 transform=None):

        super(HST, self).__init__()
        if domain not in [0, 1, 2]:
            raise ValueError('Argument "domain" must be 0, 1, or 2.')
        self.domain = domain
        self.dir_path = dir_path
    
        if preprocess == 'STFT' or preprocess == 'WT':
            self.img_dir = dir_path + "/{}_HST/".format(preprocess) + str(domain) + "/"
        elif preprocess == 'FFT':
            self.img_dir = dir_path + "/HST/".format(preprocess) + str(domain) + "/"
        else:
            raise ValueError('Invalid preprocess name.')
        self.label_subset = _normalize_label_subset(label_subset, range(5))
        self.label_mapping = {
            original_label: remapped_label for remapped_label, original_label in enumerate(self.label_subset)
        }
        self.img_list = [
            image_name for image_name in os.listdir(self.img_dir)
            if int(image_name.split('_')[0]) in self.label_mapping
        ]

        if transform is None:
            self.transform = transforms.Compose(
            [
                transforms.ToTensor(), 
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img_name = self.img_list[index]
        original_label = int(img_name.split('_')[0])
        label = torch.tensor(self.label_mapping[original_label], dtype=torch.int64)
        img_path = self.img_dir + img_name
        img = Image.open(img_path)
        img = self.transform(img)
        return img, label

class HST_FFT(Dataset):

    def __init__(self, 
                 domain,
                 dir_path,
                 labels=None,
                 fft=True):
        super(HST_FFT, self).__init__()
        self.root = dir_path

        if domain not in [0, 1, 2]:
            raise ValueError('Argument "domain" must be 0, 1 or 2')
        self.domain = domain
        self.dataset = load_HST_dataset(domain, dir_path, labels=labels, raw=True, fft=fft)
        self.data, self.labels = extract_dict_data(self.dataset)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]

        sample = torch.from_numpy(sample).float()
        label = torch.tensor(label)
        return sample, label


if __name__ == '__main__':
    data = CWRU(1, './data', 'STFT')
    data.__getitem__(0)
    data = HST(1, './data', 'STFT')
    print(data.__getitem__(0))

