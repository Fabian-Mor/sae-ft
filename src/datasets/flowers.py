import os
import sys

# --- SHADOWING FIX START ---
current_file_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_file_dir)

sys_path_backup = sys.path[:]

sys.path = [p for p in sys.path if p != src_dir and p != current_file_dir]

#import datasets
#from datasets import load_dataset

sys.path = sys_path_backup
# --- SHADOWING FIX END ---

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import io
from PIL import Image
import torchvision.transforms as transforms


class HFImageDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img_data = item['image']

        if isinstance(img_data, dict):
            image = Image.open(io.BytesIO(img_data['bytes'])).convert('RGB')
        else:
            image = img_data.convert('RGB')

        label = item['label']

        if self.transform:
            image = self.transform(image)

        return image, label


class HFCustomDataset(Dataset):
    def __init__(self, hf_dataset, transform=None, num_classes=102):
        self.dataset = hf_dataset
        self.transform = transform
        self.num_classes = num_classes
        self.indices_by_label = {i: [] for i in range(num_classes)}

        labels = self.dataset['label']
        for idx, label in enumerate(labels):
            self.indices_by_label[label].append(idx)

        self.class_list = sorted(list(self.indices_by_label.keys()))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        batch_img = []
        for c in self.class_list:
            indices = self.indices_by_label[c]
            if len(indices) > 0:
                rand_idx = np.random.choice(indices)
                item = self.dataset[int(rand_idx)]
                img_data = item['image']

                if isinstance(img_data, dict):
                    image = Image.open(io.BytesIO(img_data['bytes'])).convert('RGB')
                else:
                    image = img_data.convert('RGB')

                if self.transform:
                    image = self.transform(image)
                batch_img.append(image)
            else:
                raise ValueError(f"No images found for class {c}")

        batch_img = torch.stack(batch_img, dim=0)
        return batch_img


class Flowers102:
    test_subset = None

    def __init__(self,
                 preprocess,
                 batch_size=128,
                 num_workers=4,
                 subset='test',
                 custom=False,
                 **kwargs):
        self.batch_size = batch_size
        self.num_workers = num_workers

        dataset = None #load_dataset("nkirschi/oxford-flowers")

        # This list corresponds to the official Oxford 102 ID ordering (0-101)
        # It corrects the Label Mismatch that caused 0% accuracy.
        self.classnames = [
            'pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea', 'english marigold',
            'tiger lily', 'moon orchid', 'bird of paradise', 'monkshood', 'globe thistle',
            'snapdragon', "colt's foot", 'king protea', 'spear thistle', 'yellow iris',
            'globe-flower', 'purple coneflower', 'peruvian lily', 'balloon flower', 'giant white arum lily',
            'fire lily', 'pincushion flower', 'fritillary', 'red ginger', 'grape hyacinth',
            'corn poppy', 'prince of wales feathers', 'stemless gentian', 'artichoke', 'sweet william',
            'carnation', 'garden phlox', 'love in the mist', 'mexican aster', 'alpine sea holly',
            'ruby-lipped cattleya', 'cape flower', 'great masterwort', 'siam tulip', 'lenten rose',
            'barbeton daisy', 'daffodil', 'sword lily', 'poinsettia', 'bolero deep blue',
            'wallflower', 'marigold', 'buttercup', 'oxeye daisy', 'common dandelion',
            'petunia', 'wild pansy', 'primula', 'sunflower', 'pelargonium',
            'bishop of llandaff', 'gaura', 'geranium', 'orange dahlia', 'pink-yellow dahlia',
            'cautleya spicata', 'japanese anemone', 'black-eyed susan', 'silverbush', 'californian poppy',
            'osteospermum', 'spring crocus', 'bearded iris', 'windflower', 'tree poppy',
            'gazania', 'azalea', 'water lily', 'rose', 'thorn apple',
            'morning glory', 'passion flower', 'lotus', 'toad lily', 'anthurium',
            'frangipani', 'clematis', 'hibiscus', 'columbine', 'desert-rose',
            'tree mallow', 'magnolia', 'cyclamen', 'watercress', 'canna lily',
            'hippeastrum', 'bee balm', 'ball moss', 'foxglove', 'bougainvillea',
            'camellia', 'mallow', 'mexican petunia', 'bromelia', 'blanket flower',
            'trumpet creeper', 'blackberry lily'
        ]

        self.train_dataset = HFImageDataset(
            dataset['train'],
            transform=preprocess
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

        if custom:
            self.train_dataset_custom = HFCustomDataset(
                dataset['train'],
                transform=preprocess
            )
            self.train_loader_custom = DataLoader(
                self.train_dataset_custom,
                batch_size=1,
                shuffle=True,
                num_workers=self.num_workers
            )

        hf_subset_key = 'validation' if self.test_subset == 'val' else 'test'

        print(f"Loading Test Data from HuggingFace split: {hf_subset_key}")

        self.test_dataset = HFImageDataset(
            dataset[hf_subset_key],
            transform=preprocess
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )


class Flowers102Val(Flowers102):
    def __init__(self, *args, **kwargs):
        self.test_subset = 'val'
        super().__init__(*args, **kwargs)


class Flowers102Test(Flowers102):
    def __init__(self, *args, **kwargs):
        self.test_subset = 'test'
        super().__init__(*args, **kwargs)