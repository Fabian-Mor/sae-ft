from .cifar10 import *
from .cifar100 import *
from .fmow import FMOWID, FMOWOOD, FMOW
from .imagenet import *
from .imagenetv2 import ImageNetV2
from .imagenet_a import ImageNetAValClasses, ImageNetA
from .imagenet_r import ImageNetRValClasses, ImageNetR
from .imagenet_sketch import ImageNetSketch
from .imagenet_vid_robust import ImageNetVidRobustValClasses, ImageNetVidRobust
from .iwildcam import IWildCamID, IWildCamOOD, IWildCamIDNonEmpty, IWildCamOODNonEmpty, IWildCam, IWildCamNonEmpty, IWildCamUnlabeled
from .objectnet import ObjectNetValClasses, ObjectNet
from .ytbb_robust import YTBBRobustValClasses, YTBBRobust
from .dtd import DTD
from .dsprites import dSprites, dSpritesDataset
from .stl10 import STL10Dataset
from .cal101 import Caltech101
from .flowers import Flowers102, Flowers102Test, Flowers102Val
from .stanfordcars import StanfordCars, StanfordCarsTest
from .fgcv import FGVCAircraft
from .euro_sat import EuroSAT