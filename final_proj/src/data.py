import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_swiss_roll

# Cifar10
def cifar10(batch_size):
       '''
       return dataloader with a specific batch size.
       pixel values are scale to [0, 1].
       '''
       train_data = torchvision.datasets.CIFAR10('../data',
                            train=True,
                            transform=torchvision.transforms.ToTensor(),
                            download=True)
       train_data, val_data = torch.utils.data.random_split(train_data, [45000, 5000])
       test_data = torchvision.datasets.CIFAR10('../data',
                            train=False,
                            transform=torchvision.transforms.ToTensor(),
                            download=True)
       
       train_loader = torch.utils.data.DataLoader(
              train_data, batch_size=batch_size, shuffle=True)
       
       val_loader = torch.utils.data.DataLoader(
              val_data, batch_size=batch_size, shuffle=True)

       test_loader = torch.utils.data.DataLoader(
              test_data, batch_size=1, shuffle=True)

       return train_loader, val_loader, test_loader


# Swiss Roll

def get_swiss_roll(nr_samples):
       """
       Task: part2
       Get 3d  Swiss-roll dataset 
       """
       data, time = make_swiss_roll(nr_samples, noise=0.1)
       data = torch.Tensor(data)
       time = torch.Tensor(time)
       return data, time

# def plot_swiss_roll(nr_samples, x_k, l_color):
#        """
#        Task: part2
#        Plot 3d Swiss-roll dataset

#        """
#        fig = plt.figure(figsize=(10,5))

#        ax = fig.add_subplot(1,2,1)
#        idx_plot = np.random.permutation(nr_samples)[0:nr_samples]
#        print(idx_plot)
#        ax.scatter(x_k[idx_plot, 0], x_k[idx_plot, 2], c=l_color[idx_plot], cmap=plt.cm.Spectral)
#        ax.set_xlabel("x")
#        ax.set_ylabel(f"z")
#        ax.set_title("2D: Swiss Roll manifold ")

#        ax = fig.add_subplot(1,2,2, projection="3d")
#        idx_plot = np.random.permutation(nr_samples)[0:nr_samples]
#        ax.scatter(x_k[idx_plot, 0], x_k[idx_plot, 1],x_k[idx_plot, 2], c=l_color[idx_plot], cmap=plt.cm.Spectral)
#        ax.set_xlabel("x")
#        ax.set_ylabel("y")
#        ax.set_zlabel("z")
#        ax.set_title("3D: Swiss Roll manifold ")

#        fig.show()

def swiss_roll_data_loader(split=[3000, 1500, 1500], batch_size=32):
       train_data, _ = get_swiss_roll(split[0])
       val_data, _ = get_swiss_roll(split[1])
       test_data, _ = get_swiss_roll(split[2])

       train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
       val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
       test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)

       return train_loader, val_loader, test_loader

# Word2vec
