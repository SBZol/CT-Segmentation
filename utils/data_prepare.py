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

import numpy as np
import SimpleITK as sitk


def transform_to_2D(ori_data_path, img_output, mask_output):
    """transform 3D image to 2D slices

    Args:
        ori_data_path (str): path of original data
        img_output (str): ouput path of 2d image slices
        mask_output (str): ouput path of 2d mask slices
    """
    if not os.path.exists(img_output):
        os.mkdir(img_output)

    if not os.path.exists(mask_output):
        os.mkdir(mask_output)

    data_list = glob(os.path.join(ori_data_path, '*'))

    for path in data_list:
        data_id = os.path.split(path)[-1]

        data_list = sorted(glob(os.path.join(path, '*.nii.gz')))
        img_path = data_list[0]
        mask_path = data_list[1]

        img = sitk.ReadImage(img_path)
        mask = sitk.ReadImage(mask_path)

        img_arr = sitk.GetArrayFromImage(img)
        mask_arr = sitk.GetArrayFromImage(mask)

        try:
            for slice in range(img_arr.shape[0]):
                if slice < img_arr.shape[0] and slice < mask_arr.shape[0]:
                    fileid = data_id + "-" + str(slice)
                    np.save(os.path.join(img_output, fileid + '.npy'), img_arr[slice])
                    np.save(os.path.join(mask_output, fileid + '.npy'), mask_arr[slice])

        except Exception as e:
            print(e)
            print(str(path))


if __name__ == '__main__':
    # test code
    from config import cfg

    ori_data_path = cfg.DATA.ori_data_path
    imgs_2d_path = cfg.DATA.imgs_2d_path
    mask_2d_path = cfg.DATA.mask_2d_path

    transform_to_2D(ori_data_path, imgs_2d_path, mask_2d_path)
