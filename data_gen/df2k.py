import os

os.environ["OMP_NUM_THREADS"] = "1"

from multiprocessing import Pool
import glob
import os
from os import path as osp
from PIL import Image
from numpy import asarray
from tqdm import tqdm

from utils.hparams import hparams, set_hparams
from utils.indexed_datasets import IndexedDatasetBuilder
from utils.matlab_resize import imresize


def worker(args):
    i, path, patch_size, crop_size, thresh_size, sr_scale = args
    img_name, extension = osp.splitext(osp.basename(path))
    img = Image.open(path).convert('RGB')
    img = asarray(img)
    h, w, c = img.shape
    h = h - h % sr_scale
    w = w - w % sr_scale
    img = img[:h, :w]
    h, w, c = img.shape
    img_lr = imresize(img, 1 / sr_scale)
    ret = []
    x = 0
    while x < h - thresh_size:
        y = 0
        while y < w - thresh_size:
            x_l_left = x // sr_scale
            x_l_right = (x + crop_size[0]) // sr_scale
            y_l_left = y // sr_scale
            y_l_right = (y + crop_size[1]) // sr_scale
            cropped_img = img[x:x + crop_size[0], y:y + crop_size[1], ...]
            cropped_img_lr = img_lr[x_l_left:x_l_right, y_l_left:y_l_right]
            ret.append({
                'item_name': img_name,
                'loc': [x // crop_size[0], y // crop_size[1]],
                'loc_bdr': [(h + crop_size[0] - 1) // crop_size[0], (w + crop_size[1] - 1) // crop_size[1]],
                'path': path, 'img': cropped_img,
                'img_lr': cropped_img_lr,
            })
            y += crop_size[1]
        x += crop_size[0]

    return i, ret


def build_bin_dataset(paths, binary_data_dir, prefix, patch_size, crop_size, thresh_size):
    if isinstance(crop_size, int):
        crop_size = [crop_size, crop_size]
    sr_scale = hparams['sr_scale']
    assert crop_size[0] % sr_scale == 0
    assert crop_size[1] % sr_scale == 0
    assert patch_size % sr_scale == 0
    assert thresh_size % sr_scale == 0

    builder = IndexedDatasetBuilder(f'{binary_data_dir}/{prefix}')

    def get_worker_args():
        for i, path in enumerate(paths):
            yield i, path, patch_size, crop_size, thresh_size, sr_scale

    with Pool(processes=10) as pool:
        for ret in tqdm(pool.imap_unordered(worker, list(get_worker_args())), total=len(paths)):
            if prefix == 'test':
                builder.add_item(ret[1][0], id=ret[0])
            else:
                for r in ret[1]:
                    builder.add_item(r)
    builder.finalize()


if __name__ == '__main__':
    set_hparams()
    binary_data_dir = hparams['binary_data_dir']
    os.makedirs(binary_data_dir, exist_ok=True)
    train_img_list = []
    train_img_list += sorted(glob.glob('data/raw/Flickr2K/Flickr2K_HR/*.png'))
    train_img_list += sorted(glob.glob('data/raw/DIV2K/DIV2K_train_HR/*.png'))
    test_img_list = sorted(glob.glob('data/raw/DIV2K/DIV2K_valid_HR/*.png'))

    crop_size = hparams['crop_size']
    patch_size = hparams['patch_size']
    thresh_size = hparams['thresh_size']
    test_crop_size = hparams['test_crop_size']
    test_thresh_size = hparams['test_thresh_size']
    build_bin_dataset(test_img_list, binary_data_dir, 'test', patch_size, test_crop_size, test_thresh_size)
    build_bin_dataset(train_img_list, binary_data_dir, 'train', patch_size, crop_size, thresh_size)
