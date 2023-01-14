import paddle
from .mini_imagenet import MiniImageNet
from .cub import CUB
from .cifarfs import CIFARFS
from .fc100 import FC100
dataset_dict = {'MiniImageNet': MiniImageNet, 'CUB': CUB, 'CIFAR-FS': CIFARFS, 'FC100': FC100}
