#!/usr/bin/env python
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   cross_validation.py
@Time    :   2021/05/25 10:05:43
@Author  :   Dio
@Version :   1.0
@Contact :   sbzol.chen@gmail.com
@License :   None
@Desc    :   None
'''

# here put the import lib

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# def train_k(model, X_train, y_train, X_val, y_val, BATCH_SIZE, learning_rate, TOTAL_EPOCHS):

#     train_loader = DataLoader(TensorDataset(X_train,y_train), BATCH_SIZE, shuffle = True)
#     val_loader = DataLoader(TensorDataset(X_val, y_val), BATCH_SIZE, shuffle = True)

#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(params = model.parameters(), lr = learning_rate)

#     losses = []
#     val_losses = []
#     train_acc = []
#     val_acc = []

#     for epoch in range(TOTAL_EPOCHS):
#         model.train()
#         correct = 0   # 记录正确的个数，每个epoch训练完成之后打印accuracy
#         for i, (images, labels) in enumerate(train_loader):
#             images = images.float()
#             labels = torch.squeeze(labels.type(torch.LongTensor))
#             optimizer.zero_grad()        # 清零
#             outputs = model(images)
#             # 计算损失函数
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             losses.append(loss.item())
#             # 计算正确率
#             y_hat = model(images)
#             pred = y_hat.max(1, keepdim = True)[1]
#             correct += pred.eq(labels.view_as(pred)).sum().item()

#             if (i+1) % 10 == 0:
#             # 每10个batches打印一次loss
#                 print ('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f'%(epoch + 1, TOTAL_EPOCHS,
#                                                                    i + 1, len(X_train)//BATCH_SIZE,
#                                                                    loss.item()))
#         accuracy = 100.*correct/len(X_train)
#         print('Epoch: {}, Loss: {:.5f}, Training set accuracy: {}/{} ({:.3f}%)'.format(
#             epoch + 1, loss.item(), correct, len(X_train), accuracy))
#         train_acc.append(accuracy)

#         # 每个epoch计算测试集accuracy
#         model.eval()
#         val_loss = 0
#         correct = 0
#         with torch.no_grad():
#             for i, (images, labels) in enumerate(val_loader):
#                 images = images.float()
#                 labels = torch.squeeze(labels.type(torch.LongTensor))
#                 optimizer.zero_grad()
#                 y_hat = model(images)
#                 loss = criterion(y_hat, labels).item()      # batch average loss
#                 val_loss += loss * len(labels)             # sum up batch loss
#                 pred = y_hat.max(1, keepdim = True)[1]      # get the index of the max log-probability
#                 correct += pred.eq(labels.view_as(pred)).sum().item()

#         val_losses.append(val_loss/len(X_val))
#         accuracy = 100.*correct/len(X_val)
#         print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
#             val_loss, correct, len(X_val), accuracy))
#         val_acc.append(accuracy)

#     return losses, val_losses, train_acc, val_acc
