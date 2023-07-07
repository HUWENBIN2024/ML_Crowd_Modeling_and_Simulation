import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# class Encoder(nn.Module):
#     def __init__(self, c_hid, latent_dim):
#         super(Encoder, self).__init__()

#         self.encoder_backbone = nn.Sequential(
#         nn.Conv2d(3, c_hid, kernel_size=3, padding=1, stride=2), # 32x32 => 16x16
#         nn.GELU(),

#         nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
#         nn.GELU(),

#         nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 16x16 => 8x8
#         nn.GELU(),

#         nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
#         nn.GELU(),

#         nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 8x8 => 4x4
#         nn.GELU(),

#         nn.Flatten(), # Image grid to single feature vector
#         nn.Linear(2*16*c_hid, latent_dim), # 4 x 4 => 16 
#         )

#         self.N = torch.distributions.Normal(0, 1)

#         self.kl = 0

#     def forward(self, x):
#         x = self.encoder_backbone(x)
#         x = F.relu(self.linear1(x)) # (B, 256)
#         x = F.relu(self.linear2(x)) # (B, 256)

#         mu =  self.proj2mu(x) # (B, 2)
#         sigma = torch.exp(self.proj2sigma(x)) # (B, 2)
#         z = mu + sigma * self.N.sample(mu.shape)
#         self.kl = (sigma**2 / 2 + mu**2 / 2 - torch.log(sigma) - 1/2).mean()
#         # self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
#         return z
    
# class Decoder(nn.Module):
#     def __init__(self, c_hid, latent_dims):
#         super(Decoder, self).__init__()
#         nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 4x4 => 8x8
#         nn.GELU(),

#         nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
#         nn.GELU(),

#         nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 8x8 => 16x16
#         nn.GELU(),

#         nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
#         nn.GELU(),

#         nn.ConvTranspose2d(c_hid, 3, kernel_size=3, output_padding=1, padding=1, stride=2), # 16x16 => 32x32
#         nn.Sigmoid() # The input images is scaled between -1 and 1, hence the output has to be bounded as well

#     def forward(self, z):
#         z = F.relu(self.linear1(z))
#         z = torch.sigmoid(self.linear2(z))
#         return z.reshape((-1, 1, 28, 28))
    
# class VAE(nn.Module):
#     def __init__(self, hidden_dims, latent_dims):
#         super(VAE, self).__init__()
#         self.encoder = Encoder(hidden_dims, latent_dims)
#         self.decoder = Decoder(hidden_dims, latent_dims)

#     def forward(self, x):
#         z = self.encoder(x)
#         return self.decoder(z)

class Encoder(nn.Module):
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
        self.kl = (sigma**2 / 2 + mu**2 / 2 - torch.log(sigma) - 1/2).mean()
        return z
    
class Decoder(nn.Module):
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
    def __init__(self, input_dimension, hidden_dims, latent_dims):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dimension, hidden_dims, latent_dims)
        self.decoder = Decoder(input_dimension, hidden_dims, latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    

class DiffusionMap:

    "Implement the main utilities of a DiffusionMap"

    def diffusion_map(self,data, L):
         
         """
         Execute the Diffusion Map algorithm to compute L + 1 eigenfunctions of the Laplace-Beltrami operator for the  input data.
         eigenfunctions means eigenvectors and eigenvalues

         Parameters:
         data: input to the function 
         L: L + 1 is the required number of eigenfunctions 
       
         Returns:
         phi_l: eigenvector 
         lambda_l: eigenvalue 
         """

         #step 1 :Form a saprse distance matrix D with entries D_ij = ||y_i - y_j|| 
         kd_tree = sp.spatial.KDTree(data)
         D = kd_tree.sparse_distance_matrix(kd_tree, 100).toarray()

         #step 2 :Set ε to 5% of the diameter of the dataset ε=0.05(max_ij_D_ij)
         eps=0.05 * np.max(D)

         #step 3 :Form the kernel matrix W with W_ij = exp(−(D_ij)^2/ε)
         W = np.exp(-np.power(D, 2) / eps)
        
         #step 4 :Form the diagonal normalization matrix P_ii = Sum (W_ij)
         P = np.diag(np.sum(W,axis=1))

         #step 5 :Normalize W to form the kernel matrix K = P^−1*W*P^−1
         P_inv = np.linalg.inv(P)
         K= P_inv @ W @ P_inv

         #step6  :Form the diagonal normalization matrix Q_ii=sum K_ij
         Q = np.diag(np.sum(K, axis=1))

         #step 7 :Form the symmetric matrix Tˆ = Q^−1/2 K Q^−1/2
         Q_inv = np.diag(1/np.sqrt(np.sum(K, axis=1)))
         T_hat = Q_inv @ K @ Q_inv

         #step 8 :Find the L + 1 largest eigenvalues a_l and associated eigenvectors v_l of Tˆ
         T_hat_evals , T_hat_evecs = np.linalg.eigh(T_hat)
         a_l = np.flip(T_hat_evals[-(L+1):], axis=-1)
         v_l = np.flip(T_hat_evecs[:,-(L+1):], axis=-1)

         #step 9 :Compute the eigenvalues of Tˆ^1/ε by (λ_l)^2 = al^1/ε
         lambda_l = np.sqrt( np.power(a_l, (1/eps)) )

         #step 10 :Compute the eigenvectors φl of the matrix T = Q^−1 K by φl = Q^−1/2v_l
         phi_l = Q_inv @ v_l

         return phi_l, lambda_l