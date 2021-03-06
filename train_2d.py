#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train_2d.py
@Time    :   2021/05/21 13:46:30
@Author  :   Dio
@Version :   1.0
@Contact :   sbzol.chen@gmail.com
@License :   None
@Desc    :   None
'''

# here put the import lib
import os
import logging

import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

from utils.dataset import BasicDataset2D
from utils.config import cfg


class train_2d:
    def __init__(self, net, device, config, save_cp=True):
        self.net = net
        self.device = device

        self.imgs_path = config.imgs_path
        self.mask_path = config.mask_path

        self.classes = config.classes
        self.input_channel = config.input_channel
        self.epoch = config.epoch
        self.batch_size = config.batch_size
        self.batch_size_val = config.batch_size_val
        self.lr = config.lr
        self.val_percent = config.val_percent

        self.save_cp = save_cp

        self.train_loss = list()
        self.epochs_loss = list()

    def train(self):
        trans = transforms.Compose([transforms.ToTensor, transforms.Resize((512, 512))])
        dataset = BasicDataset2D(self.imgs_path, self.mask_path, trans, trans)

        n_val = int(len(dataset) * self.val_percent)
        n_train = len(dataset) - n_val
        train, val = random_split(dataset, [n_train, n_val])

        train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        # val_loader = DataLoader(val,
        #                         batch_size=self.batch_size_val,
        #                         shuffle=False,
        #                         num_workers=4,
        #                         pin_memory=True,
        #                         drop_last=True)
        global_step = 0
        logging.info(f'''Starting training:
            Epochs:          {self.epochs}
            Batch size:      {self.batch_size}
            Learning rate:   {self.lr}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {self.save_cp}
            Device:          {self.device.type}
        ''')

        optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-8, momentum=0.9)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if self.classes > 1 else 'max', patience=2)

        if self.classes > 1:
            criterion = torch.nn.CrossEntropyLoss()
        else:
            criterion = torch.nn.BCEWithLogitsLoss()

        for epoch in range(self.epochs):
            self.net.train()

            epoch_loss = 0
            with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{self.epochs}', unit='img') as pbar:
                for batch in train_loader:
                    imgs, true_masks = batch

                    assert imgs.shape[1] == self.input_channel, \
                        f'Network has been defined with input channels, ' \
                        f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'

                    imgs = imgs.to(device=self.device, dtype=torch.float32)
                    mask_type = torch.float32 if self.classes == 1 else torch.long
                    true_masks = true_masks.to(device=self.device, dtype=mask_type)
                    masks_pred = self.net(imgs)
                    # masks_pred = torch.argmax(masks_pred, dim=0)
                    loss = criterion(masks_pred, true_masks)
                    epoch_loss += loss.item()

                    self.train_loss.append(loss.item())

                    pbar.set_postfix(**{'loss (batch)': loss.item()})

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(self.net.parameters(), 0.1)
                    optimizer.step()

                    pbar.update(imgs.shape[0])
                    global_step += 1

                    # if global_step % (n_train // (10 * batch_size)) == 0:
                    #     val_score = eval_net_2(net, val_loader, device)
                    #
                    #     scheduler.step(val_score)
                    #
                    #     print('Validation mean IoU: {}'.format(val_score))
                    #     if config['classes'] > 1:
                    #         print('Validation cross entropy: {}'.format(val_score))
                    #     else:
                    #         print('Validation Dice Coeff: {}'.format(val_score))

            print(f'epoch loss:{epoch_loss/n_train}')

            if self.save_cp:
                try:
                    os.mkdir(self.dir_checkpoint)
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
                torch.save(self.net.state_dict(), self.dir_checkpoint + f'CP_epoch{epoch + 1}.pth')

            # loss??????
            if epoch == 1:
                train_x = range(len(self.train_loss))
                train_y = self.train_loss
                plt.plot(train_x, train_y, label='Dice')
                plt.xlabel('iteration')
                plt.ylabel('loss value')
                plt.savefig('loss.png')
            else:
                pass
            # epoch loss
            self.epochs_loss.append(epoch_loss)


if __name__ == '__main__':
    unet = smp.Unet(
        encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pretrained weights for encoder initialization
        in_channels=cfg.NET.input_channel,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
        classes=cfg.NET.classes,  # model output channels (number of classes in your dataset)
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    trainning_obj = train_2d(unet, device, cfg, save_cp=True)
