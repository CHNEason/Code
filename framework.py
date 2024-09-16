from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable as V

import cv2
import numpy as np


class MyFrame():
    def __init__(self, net, loss, lr=2e-4, evalmode=False):
        self.net = net().cuda()
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        self.loss = loss()
        self.old_lr = lr
        if evalmode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()

    def set_input(self, img_batch, mask_batch=None, img_id=None):
        self.img = img_batch
        self.mask = mask_batch
        self.img_id = img_id

    def forward(self, volatile=False):
        self.img = V(self.img.cuda(), volatile=volatile)
        if self.mask is not None:
            self.mask = V(self.mask.cuda(), volatile=volatile)

    def optimize(self):
        self.forward()
        self.net.train()
        self.optimizer.zero_grad()
        pred = self.net.forward(self.img)
        loss = self.loss(self.mask, pred)
        loss.backward()
        self.optimizer.step()
        return loss.data, pred
        
    def save(self, path, mode='Op'):
        if mode == 'NoOp':
            torch.save(self.net.state_dict(), path)
        else:
            torch.save((self.net.state_dict(), self.optimizer.state_dict()), path)
        
    def load(self, path, mode='Op'):
        if mode == 'NoOp':
            state_dict = torch.load(path)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace('module.', '')  # remove `module.`
                new_state_dict[name] = v
            self.net.load_state_dict(new_state_dict)
        else:
            state_dict = torch.load(path)
            new_state_dict = OrderedDict()
            for k, v in state_dict[0].items():
                name = k.replace('module.', '')  # remove `module.`
                new_state_dict[name] = v
            # self.net.load_state_dict(state_dict)
            self.net.load_state_dict(new_state_dict)
            self.optimizer.load_state_dict(state_dict[1])

    def update_lr(self, new_lr, mylog, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        print('update learning rate: %f -> %f' % (self.old_lr, new_lr), file=mylog)
        print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        self.old_lr = new_lr
