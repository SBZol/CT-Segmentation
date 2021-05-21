#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   data_prepare.py
@Time    :   2021/05/20 16:55:30
@Author  :   Dio
@Version :   1.0
@Contact :   sbzol.chen@gmail.com
@License :   None
@Desc    :   None
'''

# here put the import lib
import os
from glob import glob
from shutil import copyfile

import numpy as np
import SimpleITK as sitk

from config import cfg


def data_prepare(ori_imgs_path, imgs_3d_path, mask_3d_path):
    """prepare data for training
    """

    if not os.path.exists(imgs_3d_path):
        os.mkdir(imgs_3d_path)

    imgs_list = os.listdir(ori_imgs_path)
    mask_list = os.listdir(mask_3d_path)

    for n in imgs_list:
        idx = n.split('_')[0]
        idx_t = idx + '.nii'

        if idx_t in mask_list:
            output_path = os.path.join(imgs_3d_path, idx + '.nii.gz')
            img = glob(os.path.join(ori_imgs_path, n, '*.nii.gz'))[0]
            copyfile(img, output_path)


def transform_to_2D(img_path, mask_path, img_output, mask_output):
    """transform 3D image to 2D slices

    Args:
        img_path (str): path of 3D image
        mask_path (str): path of 3D mask
        img_output (str): ouput path of 2d image slices
        mask_output (str): ouput path of 2d mask slices
    """
    if not os.path.exists(img_output):
        os.mkdir(img_output)

    if not os.path.exists(mask_output):
        os.mkdir(mask_output)

    img_list = sorted(glob(os.path.join(img_path, '*.nii.gz')))
    mask_list = sorted(glob(os.path.join(mask_path, '*.nii')))

    f_count = 0

    for idx in range(len(img_list)):
        img = sitk.ReadImage(img_list[idx])
        mask = sitk.ReadImage(mask_list[idx])
        img_arr = sitk.GetArrayFromImage(img)
        mask_arr = sitk.GetArrayFromImage(mask)

        try:
            for slice in range(img_arr.shape[0]):
                if slice < img_arr.shape[0] and slice < mask_arr.shape[0]:
                    fileid = str(idx) + "-" + str(slice)
                    np.save(os.path.join(img_output, 'image' + fileid + '.npy'), img_arr[slice])
                    np.save(os.path.join(mask_output, 'mask' + fileid + '.npy'), mask_arr[slice])
                    f_count += 1
        except Exception as e:
            print(e)
            print(str(idx))


if __name__ == '__main__':

    ori_imgs_path = cfg.DATA.ori_imgs_path
    mask_3d_path = cfg.DATA.mask_3d_path
    imgs_3d_path = cfg.DATA.imgs_3d_path
    imgs_2d_path = cfg.DATA.imgs_2d_path
    mask_2d_path = cfg.DATA.mask_2d_path

    data_prepare(ori_imgs_path, imgs_3d_path, mask_3d_path)
    transform_to_2D(imgs_3d_path, mask_3d_path, imgs_2d_path, mask_2d_path)
