#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   u_net.py
@Time    :   2021/05/23 10:09:45
@Author  :   Dio
@Version :   1.0
@Contact :   sbzol.chen@gmail.com
@License :   None
@Desc    :   None
'''

# here put the import lib
import segmentation_models_pytorch as smp


def get_unet(input_channel, classes):

    unet = smp.Unet(
        encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pretrained weights for encoder initialization
        in_channels=input_channel,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
        classes=classes,  # model output channels (number of classes in your dataset)
    )

    return unet
