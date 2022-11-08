import torchvision.transforms as transforms
from PIL import Image

from tasks.srdiff import SRDiffTrainer
from utils.dataset import SRDataSet
from utils.hparams import hparams
from utils.indexed_datasets import IndexedDataset


class CelebDataSet(SRDataSet):
    def __init__(self, prefix='train'):
        super().__init__(prefix)
        preprocess_transforms = []
        if prefix == 'train' and self.data_augmentation:
            preprocess_transforms += [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20, resample=Image.BICUBIC),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            ]
        self.pre_process_transforms = transforms.Compose(preprocess_transforms + [
            transforms.CenterCrop((178, 178)),
            transforms.Resize((160, 160)),
        ])
        if self.prefix == 'test':
            self.len = 5000
            if hparams['test_save_png']:
                self.test_ids = hparams['test_ids']
                self.len = len(self.test_ids)

    def _get_item(self, index):
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}')
        if self.prefix == 'test' and hparams['test_save_png']:
            return self.indexed_ds[self.test_ids[index]]
        else:
            return self.indexed_ds[index]

    def pre_process(self, img_hr):
        """
        Args:
            img_hr: PIL, [h, w, c]
        Returns: PIL, [h, w, c]
        """
        img_hr = self.pre_process_transforms(img_hr)
        return img_hr


class SRDiffCeleb(SRDiffTrainer):
    def __init__(self):
        super().__init__()
        self.dataset_cls = CelebDataSet
