#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   dataset.py
@Time    :   2021/05/20 19:25:19
@Author  :   Dio
@Version :   1.0
@Contact :   sbzol.chen@gmail.com
@License :   None
@Desc    :   None
'''

# here put the import lib
import os
import random
from glob import glob

from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


class BasicDataset2D(Dataset):
    """
    Dataset for  2D image
    """
    def __init__(self, img_root, mask_root, img_transform=None, mask_transform=None):

        self.img_root = img_root
        self.mask_root = mask_root

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        self.img_list = glob(os.path.join(img_root, '*'))
        random.shuffle(self.img_list)
        self.mask_root = mask_root

        self.img_transform = img_transform
        self.mask_transform = mask_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_path = self.img_list[index]
        data_id = os.path.split(img_path)[-1]

        mask_path = os.path.join(self.mask_root, data_id)

        img = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)

        if self.img_transform is not None:
            img = self.img_transform(img)

        # if self.mask_transform is not None:
        #     mask = self.mask_transform(mask)

        return img, mask

    def __len__(self) -> int:
        return len(self.img_list)

    def _check_exists(self):
        return os.path.exists(self.img_root) and os.path.exists(self.mask_root)


class K_Fold_Dataset2D(Dataset):
    """
    K Fold Dataset for 2D image
    """
    def __init__(self, img_root, mask_root, k_i=0, k=5, data_type='train', img_transform=None, rand=False):

        self.img_root = img_root
        self.mask_root = mask_root

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        self.all_img_list = glob(os.path.join(img_root, '*'))
        self.img_list = None

        if rand:
            random.seed(1)
            random.shuffle(self.all_img_list)

        length = len(self.all_img_list)
        remainder = length % k
        every_f_len = (length - remainder) // k

        if data_type == 'val':
            if k_i != k - 1:
                self.img_list = self.all_img_list[every_f_len * k_i:every_f_len * (k_i + 1)]
            else:
                self.img_list = self.all_img_list[every_f_len * k_i:every_f_len * (k_i + 1) + remainder]

        elif data_type == 'train':
            self.img_list = self.all_img_list[:every_f_len * k_i] + self.all_img_list[every_f_len * (k_i + 1):]

        self.mask_root = mask_root

        self.img_transform = img_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_path = self.img_list[index]
        data_id = os.path.split(img_path)[-1]

        mask_path = os.path.join(self.mask_root, data_id)

        img = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)

        if self.img_transform is not None:
            img = self.img_transform(img)

        return img, mask

    def __len__(self) -> int:
        return len(self.img_list)

    def _check_exists(self):
        return os.path.exists(self.img_root) and os.path.exists(self.mask_root)


if __name__ == '__main__':
    # test code
    from config import cfg
    from torch.utils.data import random_split

    dir_img = cfg.DATA.imgs_2d_train
    dir_mask = cfg.DATA.mask_2d_train
    trans = transforms.Compose([transforms.ToTensor(), transforms.Resize((512, 512))])
    dataset = K_Fold_Dataset2D(dir_img, dir_mask, 0, 5, 'train', trans, False)
    dataset2 = K_Fold_Dataset2D(dir_img, dir_mask, 0, 5, 'val', trans, False)
    dataset2 = K_Fold_Dataset2D(dir_img, dir_mask, 0, 5, 'val', trans, False)
    print('..............')
    # a, b = dataset[0]
    # print(a.shape)
    # print(b.shape)
