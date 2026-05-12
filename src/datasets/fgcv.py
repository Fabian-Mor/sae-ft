import os
import torch
from PIL import Image

class _RawFGVCAircraft(torch.utils.data.Dataset):
    def __init__(self, root, split, transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.data_dir = os.path.join(root, 'fgvc-aircraft-2013b', 'data')

        variants_file = os.path.join(self.data_dir, 'variants.txt')
        with open(variants_file, 'r') as f:
            self.classes = [line.strip() for line in f if line.strip()]

        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        split_file = os.path.join(self.data_dir, f'images_variant_{split}.txt')
        self.samples = []
        with open(split_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    image_id, variant = parts
                    img_path = os.path.join(self.data_dir, 'images', f'{image_id}.jpg')
                    self.samples.append((img_path, self.class_to_idx[variant]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target

class FGVCAircraft:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=4,
                 classnames=None):
        self.preprocess = preprocess
        self.location = location
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.populate_train()
        self.populate_val()
        self.populate_test()

        self.classnames = [name.replace('/', ' ') for name in self.train_dataset.classes]

    def populate_train(self):
        self.train_dataset = _RawFGVCAircraft(
            root=self.location,
            split='train',
            transform=self.preprocess
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def populate_val(self):
        self.val_dataset = _RawFGVCAircraft(
            root=self.location,
            split='val',
            transform=self.preprocess
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def populate_test(self):
        self.test_dataset = _RawFGVCAircraft(
            root=self.location,
            split='test',
            transform=self.preprocess
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )