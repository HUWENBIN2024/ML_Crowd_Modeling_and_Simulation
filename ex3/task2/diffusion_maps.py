import numpy as np
import scipy as sp

import datafold.dynfold as dfold
import datafold.pcfold as pfold

class DiffusionMap:

    "Implement the main utilities of a DiffusionMap and  implement DissusionMap with Datafold "

    def diffusion_map_algorithm(self,data, L):
         
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
        #  alpha_l =  a_l ** (1/eps) 

         #step 10 :Compute the eigenvectors φl of the matrix T = Q^−1 K by φl = Q^−1/2v_l
         phi_l = Q_inv @ v_l

         return phi_l, lambda_l
    
    def diffusion_map_datafold_bonus(self, L, data):
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
         Data_pcm.optimize_parameters(result_scaling=0.5)

         # Compute the kernel and the eigenfunctions 
         dmap = dfold.DiffusionMaps(
             kernel=pfold.GaussianKernel(epsilon=Data_pcm.kernel.epsilon),
             n_eigenpairs=L,
             dist_kwargs=dict(cut_off=Data_pcm.cut_off),
         )
         dmap = dmap.fit(Data_pcm)
         phi_l, lambda_l = dmap.eigenvectors_, dmap.eigenvalues_

         return phi_l, lambda_l