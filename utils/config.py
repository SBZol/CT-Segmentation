#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   config.py
@Time    :   2021/05/20 15:05:12
@Author  :   Dio
@Version :   1.0
@Contact :   sbzol.chen@gmail.com
@License :   None
@Desc    :   None
'''

# here put the import lib
import os
from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.DATA = edict()
__C.DATA.root_path = os.path.join('/public', 'home', 'leedan', 'Kidney')
__C.DATA.ori_train_data = os.path.join(__C.DATA.root_path, 'Kidney_CT')
__C.DATA.ori_test_data = os.path.join(__C.DATA.root_path, 'Kidney_CT_test')
__C.DATA.imgs_2d_train = os.path.join(__C.DATA.root_path, 'processed_data', 'imgs_2d_train')
__C.DATA.mask_2d_train = os.path.join(__C.DATA.root_path, 'processed_data', 'mask_2d_train')
__C.DATA.mask_2d_test = os.path.join(__C.DATA.root_path, 'processed_data', 'imgs_2d_test')
__C.DATA.mask_2d_test = os.path.join(__C.DATA.root_path, 'processed_data', 'mask_2d_test')

__C.NET = edict()
__C.NET.input_channel = 1
__C.NET.classes = 3

__C.TRAINING = edict()
__C.TRAINING.create_data = False
__C.TRAINING.epochs = 10
__C.TRAINING.batch_size = 32
__C.TRAINING.batch_size_val = 4
__C.TRAINING.lr = 0.001
__C.TRAINING.val_percent = 0.1
__C.TRAINING.weight_path = os.path.join(__C.DATA.root_path, 'models')  # the path of model weight we want to save

__C.TEST = edict()
__C.TEST.model_weight_path = os.path.join(__C.DATA.root_path, 'models', '')
__C.TEST.test_results_path = os.path.join(__C.DATA.root_path, '', '')
