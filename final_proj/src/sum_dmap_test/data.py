import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_swiss_roll
import tensorflow as tf
from tensorflow import keras

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

def cifar10_dmap(data_loader,num_examples):
       train_labels = np.array([])
       train_images = np.empty((0, 3072), dtype=np.float32)

       # Randomly select a subset of examples
       indices = np.random.choice(len(data_loader.dataset), size=num_examples, replace=False)

       for idx, (images, labels) in enumerate(data_loader):
              if idx in indices:
                     # Reshape images to (batch_size, 3072)
                     images = images.view(images.size(0), -1)
                     
                     # Append the labels and images to the respective arrays
                     train_labels = np.append(train_labels, labels.numpy())
                     train_images = np.concatenate((train_images, images.numpy()), axis=0)
              
              if len(train_labels) >= num_examples:
                     break

       # Print the shapes of the arrays
       print("Train labels shape:", train_labels.shape)
       print("Train images shape:", train_images.shape)


       return train_images,train_labels
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