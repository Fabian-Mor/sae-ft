import os
import torch
from PIL import Image
from collections import defaultdict
import random


class _RawEuroSAT(torch.utils.data.Dataset):
    def __init__(self, root, split, transform=None, seed=42):
        self.root = root
        self.split = split
        self.transform = transform
        self.data_dir = os.path.join(root, 'EuroSAT_RGB')

        self.classes = sorted([
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d))
        ])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Collect all samples
        all_samples = defaultdict(list)
        for cls in self.classes:
            cls_dir = os.path.join(self.data_dir, cls)
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith(('.jpg', '.png', '.tif')):
                    all_samples[cls].append(
                        (os.path.join(cls_dir, fname), self.class_to_idx[cls])
                    )

        # Split: 60% train, 20% val, 20% test (per class, stratified)
        rng = random.Random(seed)
        self.samples = []
        for cls in self.classes:
            imgs = all_samples[cls]
            rng.shuffle(imgs)
            n = len(imgs)
            n_train = int(0.6 * n)
            n_val = int(0.2 * n)

            if split == 'train':
                self.samples.extend(imgs[:n_train])
            elif split == 'val':
                self.samples.extend(imgs[n_train:n_train + n_val])
            elif split == 'test':
                self.samples.extend(imgs[n_train + n_val:])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class EuroSAT:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=4,
                 classnames=None,
                 seed=42):
        self.preprocess = preprocess
        self.location = location
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.populate_train()
        self.populate_val()
        self.populate_test()

        self.classnames = self.train_dataset.classes

    def populate_train(self):
        self.train_dataset = _RawEuroSAT(
            root=self.location,
            split='train',
            transform=self.preprocess,
            seed=self.seed
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def populate_val(self):
        self.val_dataset = _RawEuroSAT(
            root=self.location,
            split='val',
            transform=self.preprocess,
            seed=self.seed
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def populate_test(self):
        self.test_dataset = _RawEuroSAT(
            root=self.location,
            split='test',
            transform=self.preprocess,
            seed=self.seed
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )