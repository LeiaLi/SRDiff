import matplotlib

from tasks.srdiff import SRDiffTrainer
from utils.dataset import SRDataSet

matplotlib.use('Agg')

from PIL import Image
from torchvision import transforms
import random
from utils.matlab_resize import imresize
from utils.hparams import hparams
import numpy as np


class Df2kDataSet(SRDataSet):
    def __init__(self, prefix='train'):
        super().__init__('train' if prefix == 'train' else 'test')
        self.patch_size = hparams['patch_size']
        self.patch_size_lr = hparams['patch_size'] // hparams['sr_scale']
        if prefix == 'valid':
            self.len = hparams['eval_batch_size'] * hparams['valid_steps']

        self.data_aug_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20, resample=Image.BICUBIC),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        ])

    def __getitem__(self, index):
        item = self._get_item(index)
        hparams = self.hparams
        sr_scale = hparams['sr_scale']

        img_hr = np.uint8(item['img'])
        img_lr = np.uint8(item['img_lr'])

        # TODO: clip for SRFlow
        h, w, c = img_hr.shape
        h = h - h % (sr_scale * 2)
        w = w - w % (sr_scale * 2)
        h_l = h // sr_scale
        w_l = w // sr_scale
        img_hr = img_hr[:h, :w]
        img_lr = img_lr[:h_l, :w_l]
        # random crop
        if self.prefix == 'train':
            if self.data_augmentation and random.random() < 0.5:
                img_hr, img_lr = self.data_augment(img_hr, img_lr)
            i = random.randint(0, h - self.patch_size) // sr_scale * sr_scale
            i_lr = i // sr_scale
            j = random.randint(0, w - self.patch_size) // sr_scale * sr_scale
            j_lr = j // sr_scale
            img_hr = img_hr[i:i + self.patch_size, j:j + self.patch_size]
            img_lr = img_lr[i_lr:i_lr + self.patch_size_lr, j_lr:j_lr + self.patch_size_lr]
        img_lr_up = imresize(img_lr / 256, hparams['sr_scale'])  # np.float [H, W, C]
        img_hr, img_lr, img_lr_up = [self.to_tensor_norm(x).float() for x in [img_hr, img_lr, img_lr_up]]
        return {
            'img_hr': img_hr, 'img_lr': img_lr,
            'img_lr_up': img_lr_up, 'item_name': item['item_name'],
            'loc': np.array(item['loc']), 'loc_bdr': np.array(item['loc_bdr'])
        }

    def __len__(self):
        return self.len

    def data_augment(self, img_hr, img_lr):
        sr_scale = self.hparams['sr_scale']
        img_hr = Image.fromarray(img_hr)
        img_hr = self.data_aug_transforms(img_hr)
        img_hr = np.asarray(img_hr)  # np.uint8 [H, W, C]
        img_lr = imresize(img_hr, 1 / sr_scale)
        return img_hr, img_lr


class SRDiffDf2k(SRDiffTrainer):
    def __init__(self):
        super().__init__()
        self.dataset_cls = Df2kDataSet
