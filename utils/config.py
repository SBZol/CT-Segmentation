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
__C.DATA.ori_data_path = os.path.join('C:\\', 'cz', 'Project', 'Data', 'Enhance_CT')
__C.DATA.imgs_2d_train = os.path.join('C:\\', 'cz', 'Project', 'Data', 'training_data', 'imgs_2d')
__C.DATA.mask_2d_train = os.path.join('C:\\', 'cz', 'Project', 'Data', 'training_data', 'mask_2d')
__C.DATA.mask_2d_test = os.path.join('C:\\', 'cz', 'Project', 'Data', 'training_data', 'mask_2d')
__C.DATA.mask_2d_test = os.path.join('C:\\', 'cz', 'Project', 'Data', 'training_data', 'mask_2d')

__C.NET = edict()
__C.NET.input_channel = 1
__C.NET.classes = 6

__C.TRAINING = edict()
__C.TRAINING.create_data = False
__C.TRAINING.checkpoint_path = os.path.join('C:\\', 'cz', 'Project', 'wb_data', 'dio_data')
__C.TRAINING.epochs = 10
__C.TRAINING.batch_size = 1
__C.TRAINING.batch_size_val = 4
__C.TRAINING.lr = 0.001
__C.TRAINING.val_percent = 0.1

__C.TEST = edict()
__C.TEST.model_weight_path = os.path.join('C:\\', 'cz', 'Project', 'wb_data', 'CP_epoch3.pth')
