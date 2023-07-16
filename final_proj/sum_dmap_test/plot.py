import numpy as np
import matplotlib.pyplot as plt
from datafold.utils.plot import plot_pairwise_eigenvector
from matplotlib import ticker
from matplotlib import pylab

def plot_swiss_roll(nr_samples,  x_k, l_color):
        """
        Task: part2
        Plot 3d Swiss-roll dataset
        
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

def plot_diffusion_maps(nr_samples, L, phi_l, lambda_l ,time=None):
     """
     Plot graphs for  different parts.
     """
     print(f"Print {L} largest eigenvalues for the corresponding dataset: \n {lambda_l}")
     idx_plot = np.random.permutation(nr_samples)[0:nr_samples]
     plot_pairwise_eigenvector(
                eigenvectors=phi_l[idx_plot, :],
                n=1,
                fig_params=dict(figsize=[15, 10]),
                scatter_params=dict(cmap=plt.cm.Spectral,c=time[idx_plot])
            )
       
     plt.show()

def plot_diffusion_maps_cifar(L, phi_l, lambda_l ,labels):
     """
     Plot graphs for  different parts.
     """
     print(f"Print {L} largest eigenvalues for the corresponding dataset: \n {lambda_l}")
     plot_pairwise_eigenvector(
            eigenvectors=phi_l[:,:],
            n=0,
            idx_start=1,
            fig_params=dict(figsize=(10, 10)),
            scatter_params=dict(c=labels)
) 
       
     plt.show()

def plot_2d(points, points_color, title):
    fig, ax = plt.subplots(figsize=(7, 4), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    
    x, y = points.T
    ax.scatter(x, y, c=points_color, s=0.5, alpha=0.8)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    plt.show()

def plot_word_embedding_damp(X_dmap, words):
    pylab.figure(figsize=(16, 9))
    for i, label in enumerate(words[:100]):
        x, y = X_dmap[i, :]
        pylab.scatter(x, y)
        pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                        ha='right', va='bottom')
    pylab.show()


