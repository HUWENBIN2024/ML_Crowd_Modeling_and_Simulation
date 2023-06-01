
import numpy as np
import matplotlib.pyplot as plt

from datafold.utils.plot import plot_pairwise_eigenvector

def plot_diffusion_maps( part, nr_samples, L, phi_l, lambda_l ,time):
     """
     Plot graphs for  different parts.
     """
     print(f"Print {L} largest eigenvalues for the corresponding dataset: \n {lambda_l}")
     idx_plot = np.random.permutation(nr_samples)[0:nr_samples]

     if part == 1:
        fig, axes = plt.subplots(int(np.ceil(L / 3)), 3, figsize=(15, 7))
        fig.subplots_adjust(hspace=0.4)

        for i, ax in enumerate(axes.flatten(), start=0):
            ax.scatter(time[idx_plot], phi_l[:, i][idx_plot], c=time[idx_plot],cmap=plt.cm.Spectral)
            ax.set_xlabel("Time")
            ax.set_ylabel(f"Eigenvector  {i}")
            ax.set_title(f"Eigenfunction {i}")
            # if i == 0:
            #  ax.set_ylim([-0.5, 0.5])

        fig.suptitle(f"Eigenfunctions values against time.")

     elif part in [2, 4]:
         plot_pairwise_eigenvector(
                eigenvectors=phi_l[idx_plot, :],
                n=1,
                fig_params=dict(figsize=[15, 10]),
                scatter_params=dict(cmap=plt.cm.Spectral,c=time[idx_plot])
            ) 

     elif part == 3:
         plot_pairwise_eigenvector(
            eigenvectors=phi_l,
            n=1,
            fig_params=dict(figsize=[15, 10]),
            scatter_params=dict(cmap=plt.cm.Spectral, c=time)
        )

         fig = plt.figure(figsize=(10, 4.5))
       
     plt.show()
  

