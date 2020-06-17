#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 16:49:42 2020

@author: weishaowei
"""

# pytorch training example using distributed training diagram

import torch
from torch import nn, optim
import os
from torchvision import datasets, transforms
# from torch.utils import tensorboard
import torch.distributed as dist
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser('Distributed Training Example')
parser.add_argument('--local_rank', type=int, default=0)


class MnistCNN(nn.Module):
    
    def __init__(self, num_class=10):
        super(MnistCNN, self).__init__()
        self.num_class = num_class
        self.feature = nn.Sequential(
                        # 14x14x16
                        nn.Conv2d(1, 16, 3, padding=1),
                        nn.BatchNorm2d(16),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(2, 2),
                        # 7x7x32
                        nn.Conv2d(16, 32, 3, padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(2, 2)
                        )
        
        self.classifer = nn.Sequential(
                        nn.Linear(7*7*32, 256),
                        nn.ReLU(inplace=True),
                        nn.Linear(256, 10)
                        )
        
    def forward(self, xs):
        xs = self.feature(xs)
        xs = xs.view(-1, 7*7*32)
        xs = self.classifer(xs)
        return xs
    

def make_dataset():
    trans = transforms.ToTensor()
    train_dset = datasets.MNIST(root='./mnist', train=True, transform=trans, download=True)
    test_dset = datasets.MNIST(root='./mnist', train=False, transform=trans, download=True)
    train_dataloader = torch.utils.data.DataLoader(train_dset,
                                                   batch_size=256,
                                                   shuffle=True,
                                                   drop_last=True,
                                                   pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(test_dset,
                                                  batch_size=512,
                                                  pin_memory=True)
    return train_dataloader, test_dataloader


def main():
    # print(os.path.abspath('./'))
    net = MnistCNN(10)
    train_dataloader, test_dataloader = make_dataset()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    
    if torch.cuda.is_available():
        net = net.cuda()
        criterion = criterion.cuda()
    
    for epoch in range(20):
        pbar = tqdm(train_dataloader)
        acc_count, total, idx = 0., 0., 0
        # training stage
        net.train()
        for imgs, labs in pbar:
            if torch.cuda.is_available():
                imgs = imgs.cuda(non_blocking=True)
                labs = labs.cuda(non_blocking=True)
            # forward    
            outputs = net(imgs)
            loss = criterion(outputs, labs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # compute accu
            preds = torch.argmax(outputs, dim=-1)
            batch_count = (preds==labs).float().sum()
            acc_count += batch_count
            total += labs.size(0)
            batch_acc = batch_count/labs.size(0)
            
            if idx%10 == 0:
                pbar.set_description('Epoch:[{:2d}]-Loss:{:.3f}-Batch_Accu:{:.3f}'.\
                                     format(epoch+1, loss.item(), batch_acc))
            idx += 1
            
        train_accu = acc_count/total
        
        # test stage
        net.eval()
        acc_count, total = 0., 0.
        with torch.no_grad():
            for imgs, labs in tqdm(test_dataloader, desc='Testing!!!'):
                if torch.cuda.is_available():
                    imgs = imgs.cuda(non_blocking=True)
                    labs = labs.cuda(non_blocking=True)
                outputs = net(imgs)
                preds = torch.argmax(outputs,dim=-1)
                acc_count += (preds==labs).float().sum()
                total += labs.size(0)
        test_accu = acc_count/total     
        print('\033[1;31mEpoch:[{:2d}]-Train_Accu:{:.3f}-Test_Accu:{:.3f}\033[0m'.format(epoch+1, train_accu, test_accu),flush=True)
            
        

if __name__ == '__main__':
    main()
        


