import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


lr = 0.001

def train_vae(epochs, vae, opt, train_data, test_data, batch_size = 32):
    ep = []
    train_loss_list = []
    test_loss_list = []
    # train
    
    for epoch in range(epochs):
        train_loss = 0
        test_loss = 0
        for i, (x, y) in enumerate(tqdm(train_data, desc='training')):
            opt.zero_grad()
            x_hat = vae(x)
            loss = ((x - x_hat)**2).mean() + vae.encoder.kl
            loss.backward()
            opt.step()
            train_loss += loss

        with torch.no_grad():
            for i, (x, y) in enumerate(tqdm(test_data, desc='val')):
                x_hat_test = vae(x)
                test_loss += ((x - x_hat_test)**2).mean() + vae.encoder.kl

        ep.append(epoch + 1) 
        train_loss_list.append(train_loss.item() / (len(train_data) * batch_size))
        test_loss_list.append(test_loss.item() / (len(train_data) * batch_size))

    plt.title("vae loss")
    plt.xlabel("steps")
    plt.ylabel("loss")
    # plt.plot(ep, train_loss_list, 'red', label='training loss')
    plt.plot(ep, test_loss_list, 'blue', label='testing loss')
    plt.legend(loc="upper right")
    plt.show()
