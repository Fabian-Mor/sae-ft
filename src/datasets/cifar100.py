import os
import numpy as np
import torch
import time
from torchvision.datasets import CIFAR100 as PyTorchCIFAR100
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class CIFAR100:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):

        self.train_dataset = PyTorchCIFAR100(
            root=location, download=True, train=True, transform=preprocess
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, num_workers=num_workers
        )

        self.test_dataset = PyTorchCIFAR100(
            root=location, download=True, train=False, transform=preprocess
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        self.classnames = self.test_dataset.classes


class CIFAR100CDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data  # shape (N, 32, 32, 3)
        self.targets = targets  # shape (N,)
        self.transform = transform

    def __getitem__(self, index):
        img = self.data[index]
        label = int(self.targets[index])  # ✅ fixed from self.labels to self.targets
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.targets)


class CIFAR100C:
    def __init__(self,
                 preprocess,
                 corruption='gaussian_noise',
                 severity=5,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16):
        assert corruption in [
            'gaussian_noise', 'shot_noise', 'impulse_noise',
            'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
            'snow', 'frost', 'fog', 'brightness', 'contrast',
            'elastic_transform', 'pixelate', 'jpeg_compression'
        ], f"Invalid corruption type: {corruption}"

        # Load .npy files
        data_path = os.path.join(location, f'CIFAR-100-C/{corruption}.npy')
        labels_path = os.path.join(location, 'CIFAR-100-C/labels.npy')

        data = np.load(data_path)  # shape (50000, 32, 32, 3)
        labels = np.load(labels_path)  # shape (50000,)

        # Pick severity slice
        start = (severity - 1) * 10000
        end = severity * 10000
        data = data[start:end]
        labels = labels[start:end]

        # Create dataset and loader
        dataset = CIFAR100CDataset(data, labels, transform=preprocess)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # Match interface of CIFAR100 class
        self.dataset = dataset
        self.loader = loader
        self.train_dataset = dataset
        self.test_dataset = dataset
        self.train_loader = loader
        self.test_loader = loader

        self.classnames = [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
            'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'cactus', 'camel', 'can', 'castle', 'caterpillar',
            'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile',
            'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
            'house', 'kangaroo', 'computer_keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard',
            'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree',
            'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain',
            'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
            'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
            'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television',
            'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree',
            'wolf', 'woman', 'worm'
        ]