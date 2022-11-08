# https://github.com/DeokyunKim/Progressive-Face-Super-Resolution/blob/master/dataloader.py
import os
import traceback

from tqdm import tqdm

from utils.hparams import hparams, set_hparams
from utils.indexed_datasets import IndexedDatasetBuilder
from PIL import Image
from numpy import asarray


def build_bin_dataset(imgs, prefix):
    binary_data_dir = hparams['binary_data_dir']
    raw_data_dir = hparams['raw_data_dir']
    os.makedirs(binary_data_dir, exist_ok=True)
    builder = IndexedDatasetBuilder(f'{binary_data_dir}/{prefix}')
    for img in tqdm(imgs):
        try:
            full_path = f'{raw_data_dir}/Img/img_align_celeba/{img}'
            image = Image.open(full_path).convert('RGB')
            data = asarray(image)
            builder.add_item({'item_name': img, 'path': full_path, 'img': data})
        except KeyboardInterrupt:
            raise
        except:
            traceback.print_exc()
            print("| binarize img error: ", img)
    builder.finalize()


if __name__ == '__main__':
    set_hparams()
    raw_data_dir = hparams['raw_data_dir']
    binary_data_dir = hparams['binary_data_dir']
    eval_partition_path = f'{raw_data_dir}/Eval/list_eval_partition.txt'

    train_img_list = []
    val_img_list = []
    test_img_list = []
    with open(eval_partition_path, mode='r') as f:
        while True:
            line = f.readline().split()
            if not line: break
            if line[1] == '0':
                train_img_list.append(line[0])
            elif line[1] == '1':
                val_img_list.append(line[0])
            else:
                test_img_list.append(line[0])
    build_bin_dataset(train_img_list, 'train')
    build_bin_dataset(val_img_list, 'valid')
    build_bin_dataset(test_img_list, 'test')
