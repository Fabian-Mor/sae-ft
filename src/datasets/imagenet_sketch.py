import os
from .imagenet import ImageNet


class ImageNetSketch(ImageNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, location=os.path.expanduser('~/data'))

    def populate_train(self):
        pass

    def get_test_path(self):
        return os.path.join(self.location, 'sketch')
