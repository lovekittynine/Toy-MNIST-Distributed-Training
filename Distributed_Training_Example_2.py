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
import time

parser = argparse.ArgumentParser('Distributed Training Example')
parser.add_argument('--local_rank', type=int, default=0)
args = parser.parse_args()

class MnistCNN(nn.Module):
    
    def __init__(self, num_class=10):
        super(MnistCNN, self).__init__()
        self.num_class = num_class
        self.feature = nn.Sequential(
                        # 14x14x16
                        nn.Conv2d(1, 16, 3, padding=1),
                        nn.BatchNorm2d(16),
                        # nn.SyncBatchNorm(16),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(2, 2),
                        # 7x7x32
                        nn.Conv2d(16, 32, 3, padding=1),
                        nn.BatchNorm2d(32),
                        # nn.SyncBatchNorm(32),
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
    train_dset = datasets.MNIST(root='./mnist', train=True, transform=trans)
    test_dset = datasets.MNIST(root='./mnist', train=False, transform=trans)
    
    # Distributed Sampler
    train_sampler = torch.utils.data.DistributedSampler(train_dset)
    test_sampler = torch.utils.data.DistributedSampler(test_dset)
    
    train_dataloader = torch.utils.data.DataLoader(train_dset,
                                                   batch_size=256,
                                                   shuffle=False, # 分布式训练应该设置为False
                                                   drop_last=True,
                                                   pin_memory=True,
                                                   sampler=train_sampler)
    test_dataloader = torch.utils.data.DataLoader(test_dset,
                                                  batch_size=512,
                                                  pin_memory=True,
                                                  sampler=test_sampler)
    return train_dataloader, test_dataloader, train_sampler


def main():
    
    # set GPU environment
    torch.cuda.set_device(args.local_rank)
    net = MnistCNN(10)
    train_dataloader, test_dataloader, train_sampler = make_dataset()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    
    if torch.cuda.is_available():
        # convert to SyBN
        
        net = net.cuda()
        criterion = criterion.cuda()
        net = nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank], output_device=args.local_rank)
    
    for epoch in range(10):
        # pbar = tqdm(train_dataloader)
        train_sampler.set_epoch(epoch) # 注意这里打乱训练数据(shuffle)
        acc_count, total, train_loss, idx = 0., 0., 0., 0
        # training stage
        net.train()
        start = time.time()
        for imgs, labs in train_dataloader:
            if torch.cuda.is_available():
                imgs = imgs.cuda(non_blocking=True)
                labs = labs.cuda(non_blocking=True)
            # forward    
            outputs = net(imgs)
            loss = criterion(outputs, labs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss
            
            # compute accu
            preds = torch.argmax(outputs, dim=-1)
            batch_count = (preds==labs).float().sum()
            acc_count += batch_count
            total += labs.size(0)
            batch_acc = batch_count/labs.size(0)
            
            if idx%10 == 0 and args.local_rank==0:
                print( 'Epoch:[{:2d}]-Loss:{:.3f}-Batch_Accu:{:.3f}'.\
                                    format(epoch+1, loss.item(), batch_acc))
            idx += 1
            
        end = time.time()   
        # 汇总所有gpu上的预测结果，在计算准确率
        acc_count = reduce_tensor(acc_count)
        train_accu = acc_count/total
        train_loss = train_loss/idx
        elasped = end-start
        
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
                
        acc_count = reduce_tensor(acc_count)
        test_accu = acc_count/total
        if args.local_rank == 0:
            print('\033[1;31mEpoch:[{:2d}]-Train_Accu:{:.3f}-Train_Loss:{:.3f}-Test_Accu:{:.3f}-Elasped:{:.3f}(Sec)\033[0m'.\
                  format(epoch+1, train_accu, train_loss, test_accu, elasped))
            
        
               
def reduce_tensor(tensor):
    """
    跨gpu之间汇总数值
    Parameters
    ----------
    tensor : TYPE
        DESCRIPTION.

    Returns
    -------
    tensor : TYPE
        DESCRIPTION.

    """
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor                
            
            
            

if __name__ == '__main__':
    # 初始化多卡环境
    dist.init_process_group(backend='nccl', init_method='env://')
    # 销毁进程
    dist.barrier()
    main()
    
        


