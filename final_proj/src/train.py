import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def train_vae(epochs, vae, opt, train_loader, test_loader, batch_size = 32):
    ep = []
    train_loss_list = []
    test_loss_list = []
    # train
    
    for epoch in range(epochs):
        train_loss = 0
        test_loss = 0
        try:
            for i, (x, y) in enumerate(tqdm(train_loader, desc='training')):
                opt.zero_grad()
                x_hat = vae(x)
                loss = ((x - x_hat)**2).mean() + vae.encoder.kl
                loss.backward()
                opt.step()
                train_loss += loss

            with torch.no_grad():
                for i, (x, y) in enumerate(tqdm(test_loader, desc='val')):
                    x_hat_test = vae(x)
                    test_loss += ((x - x_hat_test)**2).mean() + vae.encoder.kl
        except:
            for i, x in enumerate(tqdm(train_loader, desc='training')):
                opt.zero_grad()
                x_hat = vae(x)
                loss = ((x - x_hat)**2).mean() + vae.encoder.kl
                loss.backward()
                opt.step()
                train_loss += loss

            with torch.no_grad():
                for i, x in enumerate(tqdm(test_loader, desc='val')):
                    x_hat_test = vae(x)
                    test_loss += ((x - x_hat_test)**2).mean() + vae.encoder.kl

        ep.append(epoch + 1) 
        train_loss_list.append(train_loss.item() / (len(train_loader) * batch_size))
        test_loss_list.append(test_loss.item() / (len(train_loader) * batch_size))

    plt.title("vae loss")
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.plot(ep, train_loss_list, 'red', label='training loss')
    plt.plot(ep, test_loss_list, 'blue', label='testing loss')
    plt.legend(loc="upper right")
    plt.show()

def train_resnet18(epochs, model, opt, train_loader, val_loader, batch_size): 
    ep = []
    train_loss_list = []
    test_loss_list = []

    N = len(train_loader) * batch_size
    loss_func = torch.nn.CrossEntropyLoss()  
    for ep in range(epochs):
        loss_ep = 0
        for (x, y) in tqdm(train_loader):
            opt.zero_grad()
            y = y.to(torch.long)
            y_ = model(x)
            loss = loss_func(y_, y)
            loss.backward()
            opt.step()
            loss_ep += loss
        print('for ep ', ep, ' training loss: ', loss_ep / N)

    with torch.no_grad():
        for epoch in range(epochs):
            val_loss_ep = 0
            acc = 0
            for (x, y) in tqdm(val_loader):
                y = y.to(torch.long)
                y_ = model(x)
                loss = loss_func(y_, y)
                val_loss_ep += loss
                pred = torch.argmax(y_, axis=1)
                acc += torch.sum(pred == y)   
            acc /= (len(val_loader) * batch_size)
        print('for ep ', ep, ' val loss: ', val_loss_ep / (len(val_loader) * batch_size))
        print('for ep ', ep, ' val acc: ',  acc)

        ep.append(epoch + 1) 
        train_loss_list.append(loss_ep.item() / (len(train_loader) * batch_size))
        test_loss_list.append(val_loss_ep.item() / (len(train_loader) * batch_size))

    plt.title("resnet18 loss")
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.plot(ep, train_loss_list, 'red', label='training loss')
    plt.plot(ep, test_loss_list, 'blue', label='testing loss')
    plt.legend(loc="upper right")
    plt.show()