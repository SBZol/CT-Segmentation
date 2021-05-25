#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test.py
@Time    :   2021/05/22 17:27:24
@Author  :   Dio
@Version :   1.0
@Contact :   sbzol.chen@gmail.com
@License :   None
@Desc    :   None
'''

# here put the import lib
import os
import json

import torch
from visdom import Visdom
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

from utils.dataset import BasicDataset2D
from utils.config import cfg
from networks.u_net import get_unet


class test_2d:
    def __init__(self, net, device, config):
        self.net = net
        self.device = device

        self.imgs_path = config.DATA.imgs_2d_test
        self.mask_path = config.DATA.mask_2d_test
        self.test_results_path = cfg.TEST.test_results_path

        self.classes = config.NET.classes
        self.input_channel = config.NET.input_channel
        self.epoch = config.TRAINING.epoch
        self.train_curve = []
        self.test_ref_list = []

    def test(self):
        trans = transforms.Compose([transforms.ToTensor(), transforms.Resize((512, 512))])
        test_data = BasicDataset2D(self.imgs_path, self.mask_path, trans, trans)

        test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

        viz = Visdom()
        viz.line([0.], [0.], win='Dice/test', opts=dict(title='Dice/test'))
        self.net.eval()

        batch_count = 0

        with torch.no_grad():
            for batch in test_loader:
                print('testing...', batch_count)
                batch_count += 1

                # get batchs
                imgs, mask, fig_id = batch
                imgs = imgs.to(device=device, dtype=torch.float32)
                mask = imgs.to(device=device, dtype=torch.float32)

                pred = unet(imgs)
                pred = torch.nn.functional.softmax(pred, dim=1)
                pred = pred.squeeze(0)
                pred = torch.argmax(pred, dim=0).float()

                #
                tk_dice, st_dice = self.evaluate(pred, mask)
                self.test_ref_list.append({"fig_id": fig_id, "kidney_dice": tk_dice, "stone_dice": st_dice})
                self.write(self.test_results_path, self.test_ref_list)

    def write(self, path, data):
        if not os.path.exits(path):
            os.mkdir(path)
        else:
            with open(path, 'w') as f:
                json.dump(self.test_ref_list, f)

    def evaluate(self, pre, gt):
        try:
            tk_pd = np.greater(pre, 0)
            tk_gt = np.greater(gt, 0)
            tk_dice = 2 * np.logical_and(tk_pd, tk_gt).sum() / (tk_pd.sum() + tk_gt.sum())
        except ZeroDivisionError:
            return 0.0, 0.0
        try:
            st_pd = np.greater(pre, 1)
            st_gt = np.greater(gt, 1)
            st_dice = 2 * np.logical_and(st_pd, st_gt).sum() / (st_pd.sum() + st_gt.sum())
        except ZeroDivisionError:
            return tk_dice, 0.0
        return tk_dice, st_dice


if __name__ == '__main__':

    unet = get_unet(cfg.NET.input_channel, cfg.NET.classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_weight_path = cfg.TEST.model_weight_path
    unet.load_state_dict(torch.load(model_weight_path, map_location=torch.device('cpu')))
    test_2d(unet, device, cfg)
