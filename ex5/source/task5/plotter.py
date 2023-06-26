import matplotlib.pyplot as plt
import numpy as np
from utils import time_delay, t2_trajectory


def plot_pca(x: np.ndarray,
             u: np.ndarray,
             window_shape: np.ndarray,
             row: int, col: int) -> None:
    """
    Plots Principal component analysis based on different measurements taken in different areas.
    """
    
    fig, ax = plt.subplots(row, col, figsize=(row * 5, col * 5), subplot_kw=dict(projection='3d'))
    for i in range(row):
        for j in range(col):
            ax[i][j].scatter(*x.T, s=1, c=u[:window_shape[0], row * i + j])
    plt.show()
    

def plot_arclength_velocities(vel: np.ndarray, arclength: np.ndarray) -> None:
    """
    Plots the velocity on archlength over arclength of the curve

    Args:
        vel (np.ndarray): velocity
        time (np.ndarray): time
    """
    plt.rcParams["figure.figsize"] = (10, 5)
    curve_arclength = 2 * np.pi * arclength / np.max(arclength)
    plt.plot(curve_arclength, vel)
    plt.xlabel("arclength of the curve")
    plt.ylabel("velocity on arclength")
    plt.xticks([0, 6.28], ['0', '2π'], rotation='horizontal')
    plt.show()


def plot_vector_field(v_field: np.ndarray, arclength: np.ndarray) -> None:
    """
    Plots the vector field in each period

    Args:
        v_field: the vector field
    """
    plt.rcParams["figure.figsize"] = (10, 5)
    plt.plot(arclength, v_field)
    plt.xlabel("arclength")
    plt.ylabel("vector field")
    plt.show()