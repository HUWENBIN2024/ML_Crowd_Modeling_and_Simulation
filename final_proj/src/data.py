import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_swiss_roll


import gensim.downloader

# Cifar10
def cifar10(train_val_split = [45000, 5000], batch_size=32):
       '''
       return dataloader with a specific batch size.
       pixel values are scale to [0, 1].
       '''
       train_data = torchvision.datasets.CIFAR10('../data',
                            train=True,
                            transform=torchvision.transforms.ToTensor(),
                            download=True)
       train_data, val_data = torch.utils.data.random_split(train_data, train_val_split)
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
       if nr_samples  == 0:
              return None
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
def get_word_embedding_word2vec(num_data_sample=10000, seed=3407, gensim_model=None):
       '''
       embedding: np.array, shape: (num_data_sample, 300)
       '''
       if seed != None:
              np.random.seed(seed)
       index = np.random.randint(0, len(gensim_model.index_to_key), (num_data_sample,))
       words = [gensim_model.index_to_key[i] for i in index]
       embeddings = [gensim_model[word] for word in words]

       return words, embeddings

def get_word2vec_data_loader(split=[10000, 1000, 1000], seed=3407, batch_size=32):
       np.random.seed(seed)
       print(gensim.downloader.BASE_DIR)
       gensim_model  = gensim.downloader.load("word2vec-google-news-300")
       train_words, train_embeddings = get_word_embedding_word2vec(split[0], None, gensim_model)
       train_embeddings = np.array(train_embeddings)
       train_loader = torch.utils.data.DataLoader(train_embeddings, batch_size, shuffle=True)

       val_words, val_embeddings = get_word_embedding_word2vec(split[1], None, gensim_model)
       val_embeddings = np.array(val_embeddings)
       val_loader = torch.utils.data.DataLoader(val_embeddings, batch_size, shuffle=True)

       test_words, test_embeddings = get_word_embedding_word2vec(split[2], None, gensim_model)
       test_embeddings = np.array(test_embeddings)
       test_loader = torch.utils.data.DataLoader(test_embeddings, batch_size, shuffle=True)

       return train_loader, val_loader, test_loader