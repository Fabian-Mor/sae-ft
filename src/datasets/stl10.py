from torchvision.datasets import STL10
import torch
import os

class STL10Dataset:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):

        self.train_dataset = STL10(
            root=location, split='train', download=True, transform=preprocess
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
        )

        self.test_dataset = STL10(
            root=location, split='test', download=True, transform=preprocess
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
        )

        self.classnames = self.test_dataset.classes