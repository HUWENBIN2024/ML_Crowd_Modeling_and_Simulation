import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import datafold.pcfold as pfold
from datafold.dynfold import DiffusionMaps

class Encoder(nn.Module):
    '''
    Encoder of our vae model, it consists 2 linear layers.
    '''
    def __init__(self, input_dimension, hidden_dims, latent_dims):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(input_dimension, hidden_dims)
        self.linear2 = nn.Linear(hidden_dims, hidden_dims)

        self.proj2mu = nn.Linear(hidden_dims, latent_dims)
        self.proj2sigma = nn.Linear(hidden_dims, latent_dims)

        self.N = torch.distributions.Normal(0, 1)

        self.kl = torch.zeros([0])

    def forward(self, x):
        # x = torch.flatten(x, start_dim=1) # (B, input_dimension)
        x = F.relu(self.linear1(x)) # (B, hidden_dims)
        x = F.relu(self.linear2(x)) # (B, hidden_dims)

        mu =  self.proj2mu(x) # (B, latent_dims)
        sigma = torch.exp(self.proj2sigma(x)) # (B, latent_dims)
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma**2 / 2 + mu**2 / 2 - torch.log(sigma) - 1/2).sum()
        return z
    
class Decoder(nn.Module):
    '''
    Decoder of our vae model, it consists 2 linear layers.
    '''
    def __init__(self, input_dimension, hidden_dims, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, hidden_dims)
        self.linear2 = nn.Linear(hidden_dims, input_dimension)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        # return z.reshape((-1, 1, 28, 28))
        return z
    
class VAE(nn.Module):
    '''
    VAE model, the combination of an encoder and a decoder.
    '''
    def __init__(self, input_dimension, hidden_dims, latent_dims):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dimension, hidden_dims, latent_dims)
        self.decoder = Decoder(input_dimension, hidden_dims, latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    

class DiffusionMap:

    "Implement the main utilities of a DiffusionMap"
    
    def diffusion_map_datafold(self,data,L,isSwissRoll=None):
         """
         Execute the Diffusion Map algorithm to compute L + 1 eigenfunctions by using Datafold.
         eigenfunctions means eigenvectors and eigenvalues

         Parameters:
         data: input to the function 
         L: L + 1 is the required number of eigenfunctions

         Returns:
         phi_l: eigenvector 
         lambda_l: eigenvalue 
         """
    
         # Create the Manifold and Estimates cut_off and kernel bandwidth epsilon for a Gaussian kernel.
         Data_pcm = pfold.PCManifold(data)
         if isSwissRoll:
          Data_pcm.optimize_parameters()
         else:
          Data_pcm.optimize_parameters(result_scaling=2)  
         # Compute the kernel and the eigenfunctions 
         dmap = DiffusionMaps(
             kernel=pfold.GaussianKernel(epsilon=Data_pcm.kernel.epsilon),
             n_eigenpairs=L,
             dist_kwargs=dict(cut_off=Data_pcm.cut_off),
         )
         dmap = dmap.fit(Data_pcm)
         phi_l, lambda_l = dmap.eigenvectors_, dmap.eigenvalues_
    
         return phi_l, lambda_l