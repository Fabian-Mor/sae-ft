import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VisionDataset
from PIL import Image


class dSpritesDataset(Dataset):
    def __init__(self, images, targets, transform=None):
        self.images = images
        self.targets = torch.tensor(targets, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image_np = (self.images[idx] * 255).astype(np.uint8)
        image_pil = Image.fromarray(image_np).convert('RGB')

        if self.transform:
            image_pil = self.transform(image_pil)

        target = self.targets[idx]
        return image_pil, target


class dSprites:
    def __init__(self,
                 preprocess,
                 task='shape',
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 train_split_ratio=0.8,
                 seed=42,
                 classnames=None):
        valid_tasks = ['shape', 'scale', 'orientation', 'pos_x', 'pos_y']
        assert task in valid_tasks, f"Invalid task. Choose from {valid_tasks}"

        filepath = os.path.join(location, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')

        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Dataset not found at {filepath}. Please place the .npz file in the '~/data' directory."
            )

        with np.load(filepath, allow_pickle=True) as data:
            images = data['imgs']
            latents_classes = data['latents_classes']

        task_to_idx = {
            'shape': 1, 'scale': 2, 'orientation': 3, 'pos_x': 4, 'pos_y': 5
        }

        task_to_classnames = {
            'shape': ['square', 'ellipse', 'heart'],
            'scale': [str(i) for i in range(6)],
            'orientation': [str(i) for i in range(40)],
            'pos_x': [str(i) for i in range(32)],
            'pos_y': [str(i) for i in range(32)]
        }

        targets = latents_classes[:, task_to_idx[task]]
        self.classnames = task_to_classnames[task]

        num_samples = len(images)
        indices = np.arange(num_samples)

        rng = np.random.RandomState(seed)
        rng.shuffle(indices)

        split_idx = int(num_samples * train_split_ratio)
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        train_images, test_images = images[train_indices], images[test_indices]
        train_targets, test_targets = targets[train_indices], targets[test_indices]

        self.train_dataset = dSpritesDataset(
            images=train_images, targets=train_targets, transform=preprocess
        )
        self.test_dataset = dSpritesDataset(
            images=test_images, targets=test_targets, transform=preprocess
        )

        use_cuda = torch.cuda.is_available()
        kwargs = {"num_workers": num_workers, "pin_memory": True} if use_cuda else {}

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, **kwargs
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, **kwargs
        )