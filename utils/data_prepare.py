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


class data_processer:
    def __init__(self, ori_data_path):
        """process data

        Args:
            ori_data_path (str): path of original data
        """
        self.ori_data_path = ori_data_path
        self.data_list = glob(os.path.join(self.ori_data_path, '*'))

    def transform_to_2D(self, img_output, mask_output):
        """transform 3D image to 2D slices

        Args:
            img_output (str): ouput path of 2d image slices
            mask_output (str): ouput path of 2d mask slices
        """
        if not os.path.exists(img_output):
            os.mkdir(img_output)

        if not os.path.exists(mask_output):
            os.mkdir(mask_output)

        for path in self.data_list:
            data_id = os.path.split(path)[-1]

            file_list = sorted(glob(os.path.join(path, '*.nii.gz')))
            img_path = file_list[0]
            mask_path = file_list[1]

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

    def check_mask(self, claases):
        classes_count = []
        for i in range(classes):
            classes_count.append(.0)
        for path in self.data_list:

            file_list = sorted(glob(os.path.join(path, '*.nii.gz')))
            mask_path = file_list[1]

            mask = sitk.ReadImage(mask_path)
            mask_arr = sitk.GetArrayFromImage(mask)

            uni = np.unique(mask_arr)

            if max(uni) > claases - 1 or min(uni) < 0:
                print('---error  data----------------------------' + mask_path + str(uni))
            else:
                for n in uni:
                    classes_count[n] += np.sum(mask_arr == n)
        print(classes_count)


if __name__ == '__main__':
    # test code
    from config import cfg

    ori_data_path = cfg.DATA.ori_data_path
    imgs_2d_train = cfg.DATA.imgs_2d_train
    mask_2d_train = cfg.DATA.mask_2d_train
    classes = cfg.NET.classes

    processor = data_processer(ori_data_path)
    processor.check_mask(6)
    # processor.transform_to_2D(imgs_2d_train, mask_2d_train)
