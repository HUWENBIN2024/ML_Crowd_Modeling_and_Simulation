import numpy as np
from scipy.spatial.distance import cdist
from scipy import integrate
import matplotlib.pyplot as plt


def solve_ivp(x0: np.ndarray,  t_end: float, A: np.ndarray) -> np.ndarray:
    """
    Using solve_ivp to integrate the linear system.
    :param X0: initial conditions.
    :param t_end: end time .
    :param A: a linear transformation matrix.
    :returns: x1_pre: store the solutions
    """   
    linear_system = lambda t, x, A: A @ x
    x1_pre = np.zeros(x0.shape)
    for i in range(len(x0)):
        x1_pre[i, :] = integrate.solve_ivp( linear_system, t_span=[0,  t_end], 
                                           y0=x0[i, :],t_eval=[ t_end], args=[A])["y"].reshape(2,)
    return x1_pre

def solve_trajectory(x0_point: np.ndarray, t_end: float,A: np.ndarray) -> np.ndarray: 
    """
    Using solve_ivp to integrate the linear system.
    :param x0_point:  initial conditions of a trajectory in a system 
    :param t_end: end time .
    :param A: a linear transformation matrix.
    :returns: x1_pre: store the solutions
    """   
    
    linear_system = lambda t, x, A: A @ x
    t = np.linspace(0,  t_end, 10000)
    x1_pre = integrate.solve_ivp( linear_system, t_span=[0, t_end], y0=x0_point, t_eval=t, args=[A] )["y"]

    return x1_pre


def plot_dataset(*data: np.ndarray,ax):
    """
    Plots the given datasets in two dimensional.
    Each set of points gets a different color.
    :param  *data (np.ndarray): datasets
    :param  ax: position of the plot
    """
    labels = [f'X_ {i}' for i in range(len(data))]
    for xy_array, label in zip(data, labels):
        ax.scatter(xy_array[:, 0], xy_array[:, 1], s=1, label=label)
    ax.legend()
    ax.set_title("Plot dataset")
   

def plot_phase_portrait_linear(w: int, A: np.ndarray):
    """
    Plots phase portrait in a streamplot, defined by matrix A.
    :param w:  the range of values for the x and y axes in the plot
    :param A:  the linear transformation matrix.
    """
    Y, X = np.mgrid[-w:w:100j, -w:w:100j]
    UV = A @ np.row_stack([X.ravel(), Y.ravel()])

    U = UV[0,:].reshape(X.shape)
    V = UV[1,:].reshape(X.shape)
    
    ax = plt.gca()
    ax.streamplot(X, Y, U, V, density=[1, 1])
    ax.set_aspect(1)

    ax.set_xlim(-10,10)
    ax.set_ylim(-10,10)
    ax.set_title("stream plot for linear vector field of Ax")

    return ax
    

def plot_trajectory_with_phase_portrait_linear(w: int,  A: np.ndarray,  x0_point: np.ndarray, t_end: int):
    """
    Visualize the trajectory of start point x0_point with the phase portrait
    """
    x1_pre =solve_trajectory(x0_point, t_end, A)
    
    ax = plot_phase_portrait_linear(w, A)
    ax.plot(x1_pre[0,:], x1_pre[1,:], label="trajectory")
    ax.set_title(f"the trajectory of {x0_point} with the phase portrait")
    
    ax.legend(loc='lower left')



