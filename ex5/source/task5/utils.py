import numpy as np
import scipy
import os
import pandas as pd
import math
from scipy import spatial, linalg, integrate
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import Union, Iterable, Tuple
from scipy.spatial.distance import cdist
from scipy.integrate import solve_ivp








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


def compute_eigenfunctions_datafold(x, sample):
    """
    Compute the eigenfunctions for the given dataset 'u_pca' using a Datafold.
    Returns: eigenfunctions = [eigenvectors, eigenvalues]
    """

    # Create the Manifold using Datafold lib
    positions_pcm = pfold.PCManifold(x)
    positions_pcm.optimize_parameters()

    # Compute the kernel and the eigenfunctions using Datafold lib
    dmap = dfold.DiffusionMaps(
        kernel=pfold.GaussianKernel(epsilon=positions_pcm.kernel.epsilon),
        n_eigenpairs=sample,
        dist_kwargs=dict(cut_off=positions_pcm.cut_off),
    )
    dmap = dmap.fit(positions_pcm)
    evecs, evals = dmap.eigenvectors_, dmap.eigenvalues_

    return evecs, evals

