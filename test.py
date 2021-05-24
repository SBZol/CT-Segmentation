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
import torch
from visdom import Visdom
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils.dataset import BasicDataset2D
from utils.loss_function import dice_coeff
from utils.config import cfg
from networks.u_net import get_unet


class test_2d:
    def __init__(self, net, device, config):
        self.net = net
        self.device = device

        self.imgs_path = config.DATA.imgs_2d_test
        self.mask_path = config.DATA.mask_2d_test

        self.classes = config.NET.classes
        self.input_channel = config.NET.input_channel
        self.epoch = config.TRAINING.epoch
        self.train_curve = []

    def test(self):
        trans = transforms.Compose([transforms.ToTensor(), transforms.Resize((512, 512))])
        test_data = BasicDataset2D(self.imgs_path, self.mask_path, trans, trans)
        n_test = len(test_data)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

        viz = Visdom()
        viz.line([0.], [0.], win='Dice/test', opts=dict(title='Dice/test'))
        self.net.eval()
        tot = 0

        # for epoch in range(epochs):
        with tqdm(total=n_test, desc='Test', unit='batch') as pbar:

            for batch in test_loader:
                imgs, true_mask = batch
                imgs = imgs.to(device=self.device, dtype=torch.float32)
                # mask_type = torch.float32 if config['classes'] == 1 else torch.long
                true_mask = true_mask.to(device=self.device, dtype=torch.float32)
                true_mask = true_mask.squeeze(0)
                with torch.no_grad():
                    pre_mask = self.net(imgs)

                pred = torch.nn.functional.softmax(pre_mask, dim=1)
                pred = pred.squeeze(0)
                pred = torch.argmax(pred, dim=0).float()
                #                 pred = (pred > 0.5).float()
                dice = dice_coeff(pred, true_mask).item()
                viz.line([dice], win='Dice/test')
                tot += dice
                # if dice!=1:
                #     imgs=imgs.squeeze()
                #     print(imgs.size())
                #     # show_prediction(imgs,pred,true_mask)
                #     print(dice)
                pbar.set_postfix(**{'dice (batch)': dice})
                pbar.update(imgs.shape[0])

            dice = tot / n_test
            print('Test Dice Coeff: {}'.format(dice))
            train_x = range(len(self.train_curve))
            train_y = self.train_curve
            plt.plot(train_x, train_y, label='Dice')
            plt.xlabel('Num')
            plt.ylabel('dice')
            plt.savefig('test_dice.png')
            #     if (np.sum(true_mask) > 0) == (np.sum(pre_mask) > 0):
            #         n += 1
            # acc = n / n_test * 100
            # print('the accuracy of test data is:%.2f%%' % acc)


if __name__ == '__main__':

    unet = get_unet(cfg.NET.input_channel, cfg.NET.classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_weight_path = cfg.TEST.model_weight_path
    unet.load_state_dict(torch.load(model_weight_path, map_location=torch.device('cpu')))
    test_2d(unet, device, cfg)
