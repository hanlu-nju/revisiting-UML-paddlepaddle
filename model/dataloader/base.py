import os
import os.path as osp

import numpy as np
import paddle
from PIL import Image
from paddle.io import Dataset
from paddle.vision import transforms

from model.dataloader.transforms import *
from paddle.vision.transforms import BaseTransform

_imagenet_pca = {
    'eigval': paddle.to_tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': paddle.to_tensor([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}


class RandomApply(BaseTransform):
    """Apply randomly a list of transformations with a given probability.

    Args:
        transforms (sequence or torch.nn.Module): list of transformations
        p (float): probability
    """

    def _apply_image(self, image):
        if self.p < np.random.rand():
            return image
        for t in self.transforms:
            image = t(image)
        return image

    def __init__(self, trans, p=0.5):
        super().__init__()
        self.transforms = trans
        self.p = p

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    p={self.p}"
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class ToPILImage(BaseTransform):
    def __init__(self, mode=None, keys=None):
        super(ToPILImage, self).__init__(keys)
        self.mode = mode

    def _apply_image(self, pic):
        """
        Args:
            pic (Tensor|np.ndarray): Image to be converted to PIL Image.
        Returns:
            PIL: Converted image.
        """
        if not (isinstance(pic, paddle.Tensor) or isinstance(pic, np.ndarray)):
            raise TypeError('pic should be Tensor or ndarray. Got {}.'.format(
                type(pic)))

        elif isinstance(pic, paddle.Tensor):
            if pic.ndimension() not in {2, 3}:
                raise ValueError(
                    'pic should be 2/3 dimensional. Got {} dimensions.'.format(
                        pic.ndimension()))

            elif pic.ndimension() == 2:
                # if 2D image, add channel dimension (CHW)
                pic = pic.unsqueeze(0)

        elif isinstance(pic, np.ndarray):
            if pic.ndim not in {2, 3}:
                raise ValueError(
                    'pic should be 2/3 dimensional. Got {} dimensions.'.format(
                        pic.ndim))

            elif pic.ndim == 2:
                # if 2D image, add channel dimension (HWC)
                pic = np.expand_dims(pic, 2)

        npimg = pic
        if isinstance(pic, paddle.Tensor) and "float" in str(pic.numpy(
        ).dtype) and self.mode != 'F':
            pic = pic.mul(255).byte()
        if isinstance(pic, paddle.Tensor):
            npimg = np.transpose(pic.numpy(), (1, 2, 0))

        if not isinstance(npimg, np.ndarray):
            raise TypeError(
                'Input pic must be a paddle.Tensor or NumPy ndarray, ' +
                'not {}'.format(type(npimg)))

        if npimg.shape[2] == 1:
            expected_mode = None
            npimg = npimg[:, :, 0]
            if npimg.dtype == np.uint8:
                expected_mode = 'L'
            elif npimg.dtype == np.int16:
                expected_mode = 'I;16'
            elif npimg.dtype == np.int32:
                expected_mode = 'I'
            elif npimg.dtype == np.float32:
                expected_mode = 'F'
            if self.mode is not None and self.mode != expected_mode:
                raise ValueError(
                    "Incorrect mode ({}) supplied for input type {}. Should be {}"
                        .format(self.mode, np.dtype, expected_mode))
            self.mode = expected_mode

        elif npimg.shape[2] == 2:
            permitted_2_channel_modes = ['LA']
            if self.mode is not None and self.mode not in permitted_2_channel_modes:
                raise ValueError("Only modes {} are supported for 2D inputs".
                                 format(permitted_2_channel_modes))

            if self.mode is None and npimg.dtype == np.uint8:
                self.mode = 'LA'

        elif npimg.shape[2] == 4:
            permitted_4_channel_modes = ['RGBA', 'CMYK', 'RGBX']
            if self.mode is not None and self.mode not in permitted_4_channel_modes:
                raise ValueError("Only modes {} are supported for 4D inputs".
                                 format(permitted_4_channel_modes))

            if self.mode is None and npimg.dtype == np.uint8:
                self.mode = 'RGBA'
        else:
            permitted_3_channel_modes = ['RGB', 'YCbCr', 'HSV']
            if self.mode is not None and self.mode not in permitted_3_channel_modes:
                raise ValueError("Only modes {} are supported for 3D inputs".
                                 format(permitted_3_channel_modes))
            if self.mode is None and npimg.dtype == np.uint8:
                self.mode = 'RGB'

        if self.mode is None:
            raise TypeError('Input type {} is not supported'.format(
                npimg.dtype))

        return Image.fromarray(npimg, mode=self.mode)


def search_dir(dirs):
    data_dir = None

    for d in dirs:
        print(f'searching {os.path.abspath(d)}')
        if os.path.exists(d) and os.path.isdir(d):
            data_dir = d
            break
    if data_dir is None:
        raise FileNotFoundError('Data directory not found')
    print('data directory : %s' % data_dir)
    return data_dir


ROOT_DIRS = [
    # '/data/Few-Shot/data/',
    # '/home/hanlu/data/',  # add your possible data directory here
    # '/data/hanlu/',
    # '/home/yehj/Few-Shot/data/',
    # '/home/amax/data/',
    # '/home/lamda3/data/',
    # '/data/yangy/',
]


def identity(x):
    return x


class GaussianBlur(object):
    """blur a single image on CPU"""

    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = paddle.nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                       stride=1, padding=0, bias=False, groups=3)
        self.blur_v = paddle.nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                       stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = paddle.nn.Sequential(
            paddle.nn.Pad2d(radias, mode='reflect'),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = paddle.to_tensor(x).reshape((1, -1)).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.reshape((3, 1, self.k, 1)))
        self.blur_v.weight.data.copy_(x.reshape((3, 1, 1, self.k)))

        with paddle.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img


class RandomTranslateWithReflect:
    '''
    Translate image randomly

    Translate vertically and horizontally by n pixels where
    n is integer drawn uniformly independently for each axis
    from [-max_translation, max_translation].

    Fill the uncovered blank area with reflect padding.
    '''

    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, old_image):
        xtranslation, ytranslation = np.random.randint(-self.max_translation,
                                                       self.max_translation + 1,
                                                       size=2)
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size

        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = old_image.transpose(Image.ROTATE_180)

        new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))

        new_image.paste(old_image, (xpad, ypad))

        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop((xpad - xtranslation,
                                    ypad - ytranslation,
                                    xpad + xsize - xtranslation,
                                    ypad + ysize - ytranslation))
        return new_image


class BaseDataset(Dataset):

    def __init__(self, setname, unsupervised, args, augment='none'):
        super().__init__()
        im_size = args.orig_imsize
        self.args = args
        self.unsupervised = unsupervised
        self.setname = setname
        flags = ['train', 'val', 'test']
        # assert (setname in flags or setname == 'all')
        if hasattr(args, 'shot') and hasattr(args, 'query'):
            self.repeat = args.shot + args.query
        else:
            self.repeat = 1
        self.augment = augment
        self.wnids = []
        self.use_im_cache = (im_size != -1)  # not using cache
        if setname == 'all':
            self.data = []
            self.label = []
            label_shift = 0
            for flag in flags:
                d, l = self.get_data(flag)
                label = np.array(l) + label_shift
                self.label.append(label)
                label_shift = np.max(label) + 1
                self.data.append(d)
            self.data = np.concatenate(self.data)
            self.label = np.concatenate(self.label)
        else:
            self.data, self.label = self.get_data(self.setname)
        self.wnids = sorted(set(self.label))

        self.num_class = len(set(self.label))

        image_size = args.train_size
        self.transform = self.get_transform(args, augment, image_size, setname)
        self.image_shape = self.__getitem__(0)[0][0].shape

    def get_data(self, setname):
        im_size = self.args.orig_imsize
        csv_path = osp.join(self.split_path, setname + '.csv')
        cache_path = osp.join(self.cache_path, "{}.{}.{}.pt".format(self.__class__.__name__, setname, im_size))
        if self.use_im_cache:
            if not osp.exists(cache_path):
                print('* Cache miss... Preprocessing {}...'.format(setname))
                resize_ = identity if im_size < 0 else transforms.Resize(im_size)
                data, label = self.parse_csv(csv_path)
                data = [resize_(Image.open(path).convert('RGB')) for path in data]
                label = label
                print('* Dump cache from {}'.format(cache_path))
                paddle.save({'data': self.data, 'label': self.label}, cache_path)
            else:
                print('* Load cache from {}'.format(cache_path))
                cache = paddle.load(cache_path)
                data = cache['data']
                label = cache['label']
        else:
            data, label = self.parse_csv(csv_path)
        return data, label

    def get_transform(self, args, augment, image_size, setname):
        if setname == 'train':
            print(augment)
            print(augment == 'AMDIM')
            if augment == 'AMDIM':
                transforms_list = self.AMDIM_transforms(image_size)
            elif augment == 'SimCLR':
                transforms_list = self.SimCLR_transforms(image_size)
            elif augment == 'AutoAug':
                transforms_list = self.AutoAug_transforms(image_size)
            elif augment == 'RandAug':
                transforms_list = self.RandAug_transforms(image_size)
            elif augment == 'augment':
                transforms_list = [
                    transforms.RandomResizedCrop(image_size),
                    transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            elif augment == 'test':
                transforms_list = self.test_transforms(image_size)
            else:
                raise ValueError(
                    f'Non-supported Augmentation Type: {augment}. Please Revise Data Pre-Processing Scripts.')
        else:
            test_size = args.test_size
            transforms_list = self.test_transforms(test_size)
        transform = transforms.Compose(
            transforms_list + [
                transforms.Normalize(mean=np.array([0.485, 0.456, 0.406]),
                                     std=np.array([0.229, 0.224, 0.225]))
            ])
        return transform

    def test_transforms(self, image_size):
        if image_size == 32:
            resize = image_size
        else:
            resize = int((92 / 84) * image_size)
        transforms_list = [
            transforms.Resize(resize),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
        return transforms_list

    def RandAug_transforms(self, image_size):
        from .RandAugment import rand_augment_transform
        rgb_mean = (0.485, 0.456, 0.406)
        ra_params = dict(
            translate_const=int(224 * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
        )
        transforms_list = [
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            RandomApply([
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
            ], p=0.8),
            RandomApply([GaussianBlur(22)], p=0.5),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10),
                                   ra_params),
            RandomApply([transforms.Grayscale()], p=0.2),
            transforms.ToTensor(),
        ]
        return transforms_list

    def AutoAug_transforms(self, image_size):
        from .autoaug import RandAugment
        transforms_list = [
            RandAugment(2, 12),
            ERandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.8, 0.8, 0.8),
            transforms.ToTensor(),
            Lighting(0.1, _imagenet_pca['eigval'], _imagenet_pca['eigvec']),
        ]
        return transforms_list

    def SimCLR_transforms(self, image_size):
        s = 1
        color_jitter = transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        transforms_list = [
            transforms.RandomResizedCrop(size=image_size),
            transforms.RandomHorizontalFlip(),  # with 0.5 probability
            RandomApply([color_jitter], p=0.8),
            RandomApply([transforms.Grayscale()], p=0.2),
            transforms.ToTensor(),
        ]
        return transforms_list

    def AMDIM_transforms(self, image_size):
        self.flip_lr = transforms.RandomHorizontalFlip(prob=0.5)
        # col_jitter = transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8)
        # img_jitter = transforms.RandomApply([RandomTranslateWithReflect(4)], p=0.8)
        # rnd_gray = transforms.RandomGrayscale(p=0.25)
        transforms_list = [
            transforms.RandomResizedCrop(image_size),
            # transforms.RandomHorizontalFlip(p=0.5),
            # flip_lr,
            RandomApply([RandomTranslateWithReflect(4)], p=0.8),
            RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
            RandomApply([transforms.Grayscale()], p=0.25),
            transforms.ToTensor(),
            # transforms.Normalize(np.array([0.485, 0.456, 0.406]),
            #                      np.array([0.229, 0.224, 0.225]))
        ]
        return transforms_list

    def set_transformation(self, trans: str):
        self.transform = transforms.Compose(
            getattr(self, f'{trans}_transforms')(self.image_size) + [
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])

    @property
    def split_path(self):
        raise NotImplementedError

    @property
    def cache_path(self):
        raise NotImplementedError

    @property
    def eval_setting(self):
        raise NotImplementedError

    def parse_csv(self, csv_path):
        raise NotImplementedError

    @property
    def image_size(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    # @profile
    def __getitem__(self, i):
        data, label = self.data[i], self.label[i]
        if isinstance(data, str):
            data = Image.open(data).convert('RGB')
        elif isinstance(data, np.ndarray):
            data = Image.fromarray(data)
        elif isinstance(data, paddle.Tensor):
            data = Image.fromarray(data.numpy())
        if self.unsupervised:
            image_list = []
            # inp = self.flip_lr(data)
            if self.augment == 'AMDIM':
                data = self.flip_lr(data)
            for _ in range(self.repeat):
                image_list.append(self.transform(data))
                # inp = self.flip_lr(Image.open(data).convert('RGB'))
            return image_list, label
        else:
            image = self.transform(data)
        return image, label
