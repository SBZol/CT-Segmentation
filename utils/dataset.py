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

        self.img_list = sorted(glob(img_root + "*.npy"))
        self.mask_list = sorted(glob(mask_root + "*.npy"))

        self.img_transform = img_transform
        self.mask_transform = mask_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_file, = self.img_list[index]  # filename
        mask_file = self.mask_list[index]
        img = np.load(img_file)  # mhd file
        mask = np.load(mask_file)
        img = np.float32(img)
        mask = np.float32(mask)

        if self.img_transform is not None:
            img = self.img_transform(img)

        # if self.mask_transform is not None:
        #     mask = self.mask_transform(mask)

        return img, mask

    def __len__(self) -> int:
        return len(self.img_list)

    def _check_exists(self):
        return os.path.exists(self.img_root) and os.path.exists(self.mask_root)


if __name__ == '__main__':
    dir_img = "/public/home/leedan/kidney_segmentation/kidney_train/kidney_img2D/"
    dir_mask = "/public/home/leedan/kidney_segmentation/kidney_train/kidney_mask2D/"
    trans = transforms.Compose([transforms.ToTensor(), transforms.Resize((512, 512))])
    Ds = BasicDataset2D(dir_img, dir_mask, trans, trans)
    a, b = Ds[0]

    print(a.shape)
    print(b.shape)
