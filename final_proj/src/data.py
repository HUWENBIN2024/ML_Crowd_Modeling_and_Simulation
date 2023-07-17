import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from datafold.utils.plot import plot_pairwise_eigenvector
from matplotlib import ticker
from matplotlib import pylab
from sklearn.datasets import make_swiss_roll
import gensim.downloader

# Cifar10
def cifar10(train_val_split = [45000, 5000], batch_size=32):
       '''
       parameter
       train_val_split: list, the split of the train, val dataset
       batch_size: the size of a batch

       return 
       dataloaders with a specific batch size.
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
              train_data, batch_size=batch_size, shuffle=False)
       
       val_loader = torch.utils.data.DataLoader(
              val_data, batch_size=batch_size, shuffle=False)

       test_loader = torch.utils.data.DataLoader(
              test_data, batch_size=1, shuffle=False)

       return train_loader, val_loader, test_loader

def cifar10_latent_plot(latent, y):
       '''
       plot the latent space of cifar10 dataset.

       param
       latent: (N, 2), latent points
       y: label of those points

       return None
       '''
       plt.scatter(latent[:, 0], latent[:, 1], c=y, cmap='Set3')

# Swiss Roll
def get_swiss_roll(nr_samples):
       """
       Get 3d  Swiss-roll dataset 

       param
       paramnr_samples: number of samples

       return
       data: swiss roll data points, (paramnr_samples, 3)
       time: time series
       """
       if nr_samples  == 0:
              return None
       data, time = make_swiss_roll(nr_samples, noise=0.1)
       data = torch.Tensor(data)
       time = torch.Tensor(time)
       return data, time


def swiss_roll_data(split=[3000, 1500, 1500], batch_size=32):
       '''
       generate swiss roll datasets and use them to build dataloaders

       param
       split: train, val, test split
       batch_size: the size of a batch
       
       return 
       dataloader: dataloaders with a specific batch size.
       data: train, val, test data
       time: times series
       '''
       train_data, train_time = get_swiss_roll(split[0])
       val_data, val_time = get_swiss_roll(split[1])
       test_data, test_time = get_swiss_roll(split[2])

       train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
       val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
       test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

       return [train_loader, val_loader, test_loader], [train_data, val_data, test_data], [train_time, val_time, test_time]

def plot_swiss_roll(nr_samples, x_k, l_color, latent_data):
       '''
       plot swiss roll in 3d space and the projection to xy plane.

       param
       nr_samples: number of samples
       x_k: data points, (N, 3)
       l_color: color setup
       latent_data: data points in latent space

       return None
       '''
       fig = plt.figure(figsize=(15,5))
       
       ax = fig.add_subplot(1,3,1)
       idx_plot = np.random.permutation(nr_samples)[0:nr_samples]
       ax.scatter(x_k[idx_plot, 0], x_k[idx_plot, 2], c=l_color[idx_plot], cmap=plt.cm.Spectral)
       ax.set_xlabel("x")
       ax.set_ylabel(f"z")
       ax.set_title("2D: Swiss Roll manifold ")

       ax = fig.add_subplot(1,3,2, projection="3d")
       idx_plot = np.random.permutation(nr_samples)[0:nr_samples]
       ax.scatter(x_k[idx_plot, 0], x_k[idx_plot, 1],x_k[idx_plot, 2], c=l_color[idx_plot], cmap=plt.cm.Spectral)
       ax.set_xlabel("x")
       ax.set_ylabel("y")
       ax.set_zlabel("z")
       ax.set_title("3D: Swiss Roll manifold ")

       ax = fig.add_subplot(1,3,3)
       idx_plot = np.random.permutation(nr_samples)[0:nr_samples]
       ax.scatter(latent_data[idx_plot, 0], latent_data[idx_plot, 1], c=l_color[idx_plot], cmap=plt.cm.Spectral)
       ax.set_xlabel("x")
       ax.set_ylabel("y")
       ax.set_title("3D: Swiss Roll manifold ")
       
       fig.show()

#swissroll_sun
def plot_swiss_roll_dmap(nr_samples,  x_k, l_color):
        """
        Plot 2d and 3d Swiss-roll dataset
        
        """
        fig = plt.figure(figsize=(10,5))
        
        ax = fig.add_subplot(1,2,1)
        idx_plot = np.random.permutation(nr_samples)[0:nr_samples]
        ax.scatter(x_k[idx_plot, 0], x_k[idx_plot, 2], c=l_color[idx_plot], cmap=plt.cm.Spectral)
        ax.set_xlabel("x")
        ax.set_ylabel(f"z")
        ax.set_title("2D: Swiss Roll manifold ")

        ax = fig.add_subplot(1,2,2, projection="3d")
        idx_plot = np.random.permutation(nr_samples)[0:nr_samples]
        ax.scatter(x_k[idx_plot, 0], x_k[idx_plot, 1],x_k[idx_plot, 2], c=l_color[idx_plot], cmap=plt.cm.Spectral)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("3D: Swiss Roll manifold ")
        
        fig.show()

def plot_diffusion_maps_swissRoll(nr_samples, L, phi_l, lambda_l ,time=None):
     """
     Plot graphs for  different eigenvectors applied with Diffusion Maps.
     """
     print(f"Print {L} largest eigenvalues for the corresponding dataset: \n {lambda_l}")
     idx_plot = np.random.permutation(nr_samples)[0:nr_samples]
     plot_pairwise_eigenvector(
                eigenvectors=phi_l[idx_plot,:],
                n=1,
                fig_params=dict(figsize=[15, 10]),
                scatter_params=dict(cmap=plt.cm.Spectral,c=time[idx_plot])
            )
       
     plt.show()

def plot_dmap_2d(points, points_color, title):
    """
     Plot the selected eigenvectors in Diffusion Maps in 2D.
     """
    fig, ax = plt.subplots(figsize=(7, 4), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    
    x, y = points.T
    ax.scatter(x, y, c=points_color, s=0.5, alpha=0.8,cmap=plt.cm.Spectral)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    plt.show()

# Word2vec
def get_word_embedding_word2vec(num_data_sample=10000, seed=3407, gensim_model=None):
       '''
       get word2vec embeddings from gensim
       
       param
       num_data_sample: number of samples
       seed: random seed
       gensim_model: gensim model

       return
       words: original words
       embedding: np.array, shape: (num_data_sample, 300)
       '''
       if seed != None:
              np.random.seed(seed)
       index = np.random.randint(0, len(gensim_model.index_to_key), (num_data_sample,))
       words = [gensim_model.index_to_key[i] for i in index]
       embeddings = [gensim_model[word] for word in words]

       return words, embeddings

def get_word2vec_data(split=[10000, 1000, 1000], seed=3407, batch_size=32):
       '''
       get train, val, test data from gensim model, build data loader for them

       param
       split: train, val, test split
       seed: random seed
       batch_size: the size of a batch

       return 
       dataloader: dataloaders with a specific batch size.
       embeddings: word2vec embeddings 
       words: origal words
       '''
       np.random.seed(seed)
       print(gensim.downloader.BASE_DIR)
       gensim_model  = gensim.downloader.load("word2vec-google-news-300")
       train_words, train_embeddings = get_word_embedding_word2vec(split[0], None, gensim_model)
       train_embeddings = np.array(train_embeddings)
       train_loader = torch.utils.data.DataLoader(train_embeddings, batch_size, shuffle=False)

       val_words, val_embeddings = get_word_embedding_word2vec(split[1], None, gensim_model)
       val_embeddings = np.array(val_embeddings)
       val_loader = torch.utils.data.DataLoader(val_embeddings, batch_size, shuffle=False)

       test_words, test_embeddings = get_word_embedding_word2vec(split[2], None, gensim_model)
       test_embeddings = np.array(test_embeddings)
       test_loader = torch.utils.data.DataLoader(test_embeddings, batch_size, shuffle=False)

       return [train_loader, val_loader, test_loader], [train_embeddings, val_embeddings, test_embeddings], [train_words, val_words, test_words], gensim_model


def word_embedding_plot(latent_vec, words):
       '''
       plot the latent space of word2vec

       param
       latent_vec: latent points, (N, 2)
       words: corresponding words
       '''
       plt.figure(figsize=(20, 20))
       plt.scatter(latent_vec[:,0],latent_vec[:,1],linewidths=1,color='blue')
       plt.title("Word Embedding Space",size=20)
       for i, word in enumerate(words):
              plt.annotate(word,xy=(latent_vec[i,0],latent_vec[i,1]))

#word2vwc_sun

def plot_word_embedding_damp(X_dmap, words):
    """
     Plot word embedding in Diffusion Maps after dimensionality reduction.
     """
    pylab.figure(figsize=(16, 9))
    for i, label in enumerate(words[:100]):
        x, y = X_dmap[i, :]
        pylab.scatter(x, y)
        pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                        ha='right', va='bottom')
    pylab.show()


# cifar
def cifar10_dmap(data_loader, num_examples):
    """
     Get images, labels from the given data_loader
     param
       data_loader: the given data_loader
       num_examples: number of samples we want to obtain from the given data_loader

       return 
       train_images obtain the data images from data_loader
       train_labels: obtain the data labels from data_loader

     """
    train_labels = np.array([])
    train_images = np.empty((0, 3072), dtype=np.float32)

    indices = np.random.choice(len(data_loader.dataset), size=num_examples, replace=False)

    for idx, (images, labels) in enumerate(data_loader):
        if idx in indices:
            images = images.view(images.size(0), -1)
            train_labels = np.append(train_labels, labels.numpy())
            train_images = np.concatenate((train_images, images.numpy()), axis=0)
        
        if len(train_labels) >= num_examples:
            break

    print("Train labels shape:", train_labels.shape)
    print("Train images shape:", train_images.shape)

    return train_images, train_labels

def plot_diffusion_maps_cifar(L, phi_l, lambda_l ,labels):
     """
     Plot images from Cifar10 after reducding the dimension by Diffusion Maps.
     """
     print(f"Print {L} largest eigenvalues for the corresponding dataset: \n {lambda_l}")
     plot_pairwise_eigenvector(
            eigenvectors=phi_l[:,1:],
            n=0,
            idx_start=1,
            fig_params=dict(figsize=(10, 10)),
            scatter_params=dict(c=labels)
) 
       
     plt.show()

