import math
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
import numpy as np
from bisect import bisect
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from learning import EarlyStopping
from vae import VAE
from torchvision.utils import save_image
import argparse

import os
from os.path import join, exists
from os import mkdir
from sklearn.metrics import confusion_matrix
from PIL import Image
import cv2
import warnings
import matplotlib.pyplot as plt
import pandas as pd
from bisect import bisect
from myloader import RolloutObservationDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE ')


    parser.add_argument('--train', default="/home/mao/23Spring/cars/half_pre/01dataset/",
                        help='Best model is not reloaded if specified')
    parser.add_argument('--test',default="/home/mao/23Spring/cars/half_pre/013dataset/",
                        help='Does not save samples during training if specified')

    args = parser.parse_args()

    train_path=args.train
    test_path=args.test
    batch_size=64

    device = 'cuda:0'
    print(device)



    # net = Net()
    model = VAE(1, 32).to(device)
    best = torch.load("safe_vae/best.tar")
    model.load_state_dict(best["state_dict"])



    train_dataset=RolloutObservationDataset(train_path,leng=0)
    test_dataset=RolloutObservationDataset(test_path,leng=0)
    test_dataset.load_next_buffer()
    train_dataset.load_next_buffer()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last = True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,drop_last = True)

    #
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,drop_last = True)
    # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True,drop_last = True)

    lf2=nn.CrossEntropyLoss()
    model.eval()
    for epoch in range(401):
        running_loss = 0.0
        recon_loss=0
        ce_loss=0
        correct = 0
        total = 0
        losssum = 0
        model.train()
        # safe_monotor=0
        pbar = tqdm(total=len(train_loader),
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
        for i, data in enumerate(train_loader, 0):
            total = total + len(data[0])
            inputs, labels,actions = data
            inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()  # 优化器清零
            inputs = inputs.float()
            inputs = inputs.to(device)
            labels = labels.to(device)

            recon_batch, mu, logvar = model(inputs.unsqueeze(1))



        trainloss.append(losssum/total)
        train_celoss.append(ce_loss/total)
        train_reconloss.append(recon_loss/total)
        trainacc.append(correct/total)
        pbar.close()

        model.eval()
        test_loss = 0

        test_preds = []
        test_trues = []
        test_reconloss=[]
        test_celoss=[]
        total = 0
        correct=0
        recon_loss=0
        ce_loss=0
        with torch.no_grad():
            pbar = tqdm(total=len(test_loader),
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
            for i, data in enumerate(test_loader, 0):
                total = total + len(data[0])
                inputs, labels ,actions= data
                inputs, labels = Variable(inputs.unsqueeze(1)), Variable(labels)
                inputs = inputs.float()
                inputs = inputs.to(device)
                labels = labels.to(device)
                recon_batch, mu, logvar = model(inputs)


