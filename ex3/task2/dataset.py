
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_swiss_roll


class Dataset:

    def __init__(self, nr_samples):

        self.nr_samples = nr_samples   
    
    def get_periodic_data(self):
        """
        Task: part1
        Get 2d point cloud for the periodic dataset 

        parameters: 
        n:number of samples to create 
        n_plot: number of samples to plot

        returns:
        x_k, t_k: coordinates  for the dataset 

        """
        k = np.array(np.arange(0, self.nr_samples))
        t_k = 2 * np.pi * k / (self.nr_samples + 1)
        x_k = np.asarray([np.cos(t_k), np.sin(t_k)]).T 

        return x_k, t_k

    def plot_periodic_data(self, x_k, t_k):
        """
        Task: part1
        Plot dimension reduction, eigenfunction for the periodic dataset with respect to time
        
        """
        fig = plt.figure(figsize=(10, 4.5))
        
        #plot a 2d version for the given dataset
        ax  = fig.add_subplot(1,2,1)
        ax.scatter(x_k[:, 0], x_k[: ,1], c=t_k,cmap=plt.cm.Spectral)
        ax.set_title("2d: periodic dataset with 1000 points")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        #plot a 3d version for the given dataset
        ax  = fig.add_subplot(1,2,2,projection="3d")
        ax.scatter(x_k[:, 0], x_k[:, 1], t_k, c=t_k,cmap=plt.cm.Spectral)
        ax.set_title("3d: periodic dataset with 1000 points")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel('tk')
        fig.show()


    def get_swiss_roll(self):
        """
        Task: part2
        Get 3d  Swiss-roll dataset 
        """

        return make_swiss_roll(self.nr_samples, noise=0.1)

    def plot_swiss_roll(self,  x_k, l_color):
        """
        Task: part2
        Plot 3d Swiss-roll dataset
        
        """
        fig = plt.figure(figsize=(10,5))
        
        ax = fig.add_subplot(1,2,1)
        idx_plot = np.random.permutation(self.nr_samples)[0:self.nr_samples]
        ax.scatter(x_k[idx_plot, 0], x_k[idx_plot, 2], c=l_color[idx_plot], cmap=plt.cm.Spectral)
        ax.set_xlabel("x")
        ax.set_ylabel(f"z")
        ax.set_title("2D: Swiss Roll manifold ")

        ax = fig.add_subplot(1,2,2, projection="3d")
        idx_plot = np.random.permutation(self.nr_samples)[0:self.nr_samples]
        ax.scatter(x_k[idx_plot, 0], x_k[idx_plot, 1],x_k[idx_plot, 2], c=l_color[idx_plot], cmap=plt.cm.Spectral)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("3D: Swiss Roll manifold ")
        
        fig.show()

    def plot_swiss_roll_PCA(self,  reconstruction, time):
        """
        Task: part2
        Plot  Swiss-roll with PCA method as comparision
        
        """
        fig = plt.figure(figsize=(17,5))
        
        ax = fig.add_subplot(1,3,1)
        idx_plot = np.random.permutation(self.nr_samples)[0:self.nr_samples]
        ax.scatter(reconstruction[idx_plot, 0], reconstruction[idx_plot, 1],c=time[idx_plot], cmap=plt.cm.Spectral)
        # plt.scatter(reconstruction[:,0],reconstruction[:,1], c=time, cmap=plt.cm.Spectral)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("2D: Swiss Roll manifold on xy plane")

         
        ax = fig.add_subplot(1,3,2)
        idx_plot = np.random.permutation(self.nr_samples)[0:self.nr_samples]
        ax.scatter(reconstruction[idx_plot, 0], reconstruction[idx_plot, 2], c=time[idx_plot], cmap=plt.cm.Spectral)
        #  plt.scatter(reconstruction[:,0],reconstruction[:,2],  c=time, cmap=plt.cm.Spectral) 
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_title("2D: Swiss Roll manifold on xz plane")

        ax = fig.add_subplot(1,3,3,projection="3d")
        idx_plot = np.random.permutation(self.nr_samples)[0:self.nr_samples]
        ax.scatter(reconstruction[idx_plot, 0], reconstruction[idx_plot, 1],reconstruction[idx_plot, 2], c=time[idx_plot], cmap=plt.cm.Spectral)
        #  plt.scatter(reconstruction[:,0],reconstruction[:,1], reconstruction[:,1],  c=time, cmap=plt.cm.Spectral) 
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("3D: Swiss Roll manifold")

        fig.show()
