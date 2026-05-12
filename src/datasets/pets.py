import os
import torch
from torchvision.datasets import OxfordIIITPet

class Pets:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):

        self.train_dataset = OxfordIIITPet(
            root=location,
            split='trainval',
            target_types='category',
            transform=preprocess,
            download=True
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, num_workers=num_workers
        )

        self.test_dataset = OxfordIIITPet(
            root=location,
            split='test',
            target_types='category',
            transform=preprocess,
            download=True
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        self.classnames = [
            "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair", "Egyptian_Mau",
            "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue", "Siamese", "Sphynx",
            "American_Bulldog", "American_Pit_Bull_Terrier", "Basset_Hound", "Beagle", "Boxer",
            "Chihuahua", "English_Cocker_Spaniel", "English_Setter", "German_Shorthaired",
            "Great_Pyrenees", "Havanese", "Japanese_Chin", "Keeshond", "Leonberger",
            "Miniature_Pinscher", "Newfoundland", "Pomeranian", "Pug", "Saint_Bernard",
            "Samoyed", "Scottish_Terrier", "Shiba_Inu", "Staffordshire_Bull_Terrier",
            "Wheaten_Terrier", "Yorkshire_Terrier"
        ]