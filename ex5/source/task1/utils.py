import numpy as np
from scipy import spatial, linalg, integrate
import matplotlib.pyplot as plt


def radial_basis(X:     np.ndarray,
                 grid:  np.ndarray,
                 epsilon: float) -> np.ndarray:
    """
    Computes a matrix (i,j) with the values phi_j(x_i) of the j-th basis
    function at the i-th data point. The basis functions are defined as
    phi_j(x) = exp(-||grid_j - x||^2 / epsilon^2)

    Args:
        X (np.ndarray, shape (N)): Data array of independent parameters
        grid (np.ndarray, shape (L)): Center points of radial basis functions
        epsilon (float): parameter governing the smoothness / peakedness of the
            gaussians.

    Returns:
        np.ndarray, shape (N, L): matrix (i,j) with entries phi_j(x_i)
    """
    if X.ndim == 1: # reshaping required in case of 1-dim input
        out = np.exp(-(spatial.distance_matrix(X[:, None],grid[:, None]) / epsilon)**2)
    else:
        out = np.exp(-(spatial.distance_matrix(X, grid) / epsilon) ** 2)
    return out


def radial_basis_lst_sqr_approx(X: np.ndarray,
                                F: np.ndarray,
                                grid: np.ndarray,
                                epsilon: float,
                                cond: float) -> np.ndarray:
    """
    Calculates the optimal coefficients C for the ansatz function:
    f(x) = sum_{l=1}^L c_l phi_l(x) such that C minimizes
    ||F - phi(X) @ C.T||^2 where phi_l are the radial basis functions used in
    the radial_basis method.

    Args:
        X (np.ndarray): Data array of independent parameters (input)
        F (np.ndarray): Data array of dependent parameters (label)
        grid (np.ndarray): Center points of radial basis functions
        epsilon (float): parameter governing the smoothness / peakedness of the
            gaussians.
        cond (float): Cutoff for 'small' singular values; used to determine
            effective rank of matrix (see below). Singular values smaller than
            cond * largest_singular_value are considered zero.

    Returns:
        np.ndarray: optimal coefficients w.r.t the basis defined by grid
            in a least squares sense
    """
    phi_X = radial_basis(X, grid, epsilon)
    matrix = phi_X.T @ phi_X
    target = phi_X.T @ F
    coefficients, _, _, _ = linalg.lstsq(matrix, target, cond=cond)
    return coefficients


def linear_basis_lst_sqr_approx(X: np.ndarray, F: np.ndarray) -> float:
    """
    Calculates the optimal parameter A of the linear function f(x) = A * x.
    The x_i in X have to be scalars.
    The output is optimal in a least squares sense.

    Args:
        X (np.ndarray): Data array of independent parameters (input)
        F (np.ndarray): Data array of dependent parameters (label)

    Returns:
        float: slope a of the linear function f(x) = a * x such that
            ||F - a * X||^2 is minimized w.r.t. a.
    """
    Xt_X = X.T @ X
    Xt_F = X.T @ F
    return Xt_F / Xt_X

def compute_mse(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Computes the mean squared error between to arrays
    Assumes that 0 is the batch axis and 1 is the instance axis

    Args:
        x1 (np.ndarray): shape (N, dim)
        x2 (np.ndarray): shape (N, dim)

    Returns:
        float: MSE Error
    """
    N = x1.shape[0]
    return np.sum((x1 - x2)**2)  / N

def LSM(x, y):
    '''
    Solves the equation y = A @ x with least-squares minimization
    :param x: x
    :param y: f(x)
    :returns: coefficient matrix A
    '''
    return np.linalg.lstsq(x, y, rcond=10e-6)

def RBF(eps, domain):
    '''
    Defines a concatenation of L radial basis functions with L central
    points uniformly spaced over the given domain and denominator eps
    :param L: number of central points
    :param eps: denominator of the radial basis function formula
    :param domain: domain of the function to aproximate
    '''
    phi = lambda x: np.array([np.exp(-np.linalg.norm(x_l - x) ** 2 / eps) for x_l in domain])
    return phi
