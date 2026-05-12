import os
import torch
from PIL import Image


class _RawDTD(torch.utils.data.Dataset):
    def __init__(self, root, split, split_idx=1, transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.data_dir = os.path.join(root, 'dtd')

        # Classes from directory names
        images_dir = os.path.join(self.data_dir, 'images')
        self.classes = sorted([
            d for d in os.listdir(images_dir)
            if os.path.isdir(os.path.join(images_dir, d))
        ])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Use official split files (train1.txt, val1.txt, test1.txt, etc.)
        split_file = os.path.join(self.data_dir, 'labels', f'{split}{split_idx}.txt')
        self.samples = []
        with open(split_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Lines are like "banded/banded_0001.jpg"
                    cls_name = line.split('/')[0]
                    img_path = os.path.join(images_dir, line)
                    self.samples.append((img_path, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class DTD:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=4,
                 classnames=None,
                 split_idx=1):
        self.preprocess = preprocess
        self.location = location
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_idx = split_idx

        self.populate_train()
        self.populate_val()
        self.populate_test()

        self.classnames = self.train_dataset.classes

    def populate_train(self):
        self.train_dataset = _RawDTD(
            root=self.location,
            split='train',
            split_idx=self.split_idx,
            transform=self.preprocess
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def populate_val(self):
        self.val_dataset = _RawDTD(
            root=self.location,
            split='val',
            split_idx=self.split_idx,
            transform=self.preprocess
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def populate_test(self):
        self.test_dataset = _RawDTD(
            root=self.location,
            split='test',
            split_idx=self.split_idx,
            transform=self.preprocess
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )