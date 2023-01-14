import paddle
import os.path as osp
import PIL
from PIL import Image
import pickle
import numpy as np
from .base import BaseDataset
from paddle.vision import transforms


def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)
    return label2inds


def load_data(file):
    try:
        with open(file, 'rb') as fo:
            data = pickle.load(fo)
        return data
    except:
        with open(file, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()
        return data


class CIFARFS(BaseDataset):

    def __init__(self, setname, unsupervised, args, augment='none'):
        self.DATA_PATH = osp.join(args.data_root, 'CIFAR-FS')
        super().__init__(setname, unsupervised, args, augment)

    @property
    def eval_setting(self):
        return [(5, 1), (5, 5), (5, 20), (5, 50)]

    def get_data(self, setname):
        data_train = load_data(osp.join(self.DATA_PATH,
            'CIFAR_FS_{}.pickle'.format(setname)))
        self.data = data_train['data']
        self.label = data_train['labels']
        _, self.label = np.unique(np.array(self.label), return_inverse=True)
        self.label = self.label.tolist()
        return self.data, self.label

    @property
    def image_size(self):
        return 32
