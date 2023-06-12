import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA


def solve_euler(f_ode, y0, time):
    """
    Solves the given ODE system in f_ode using forward Euler.
    :param f_ode: the right hand side of the ordinary differential equation d/dt x = f_ode(x(t)).
    :param y0: the initial condition to start the solution at.
    :param time: np.array of time values (equally spaced), where the solution must be obtained.
    :returns: (solution[time,values], time) tuple.
    """
    yt = np.zeros((len(time), len(y0)))
    yt[0, :] = y0
    step_size = time[1]-time[0]
    for k in range(1, len(time)):
        yt[k, :] = yt[k-1, :] + step_size * f_ode(yt[k-1, :])
    return yt, time


def plot_phase_portrait_nonlinear(U, V, X, Y, alpha):
    """
    Plots a non-linear vector field in a streamplot, defined with X and Y coordinates and the derivatives U and V.
    """
    fig1 = plt.figure(figsize=(7, 7))
    gs = gridspec.GridSpec(nrows=1, ncols=1, height_ratios=[1])

    #  Varying density along a streamlne
    ax1 = fig1.add_subplot(gs[0, 0])
    ax1.streamplot(X, Y, U, V, density=[0.9, 1])
    ax1.set_title(f'alpha= {alpha}');
    ax1.set_aspect(1)
    return ax1

def create_phase_portrait_derivative(system: list, alpha: float, save_plots=False,
                                     save_path: str = None, display=True, fig_size=10):
    """
    Plots the phase portrait of the given 'system', where 'system' is a 2 dimensional system given as couple of strings
    :param system: system ODEs
    :param alpha: system's parameter
    :param title_suffix: suffix to add to the title (after the value of alpha and A's eigenvalues
    :param save_plots: if True, saves the plots instead of displaying them
    :param save_path: path where to save the plots if save_plots is True
    :param display: if True, display the plots
    :param fig_size: gives width and height of plotted figure
    """
    # check if given parameters are acceptable
    if len(system) != 2:
        print("A 2 ODE system is required.")
        return
    # setting up grid width/height
    w = 5
    Y, X = np.mgrid[-w:w:100j, -w:w:100j]
    # dynamic system parameter, responsible for the change in behaviour
    alpha = alpha
    U, V = [], []
    for x2 in X[0]:
        for x1 in Y[:, 0]:
            u = eval(system[0])
            v = eval(system[1])
            U.append(u)
            V.append(v)
    U = np.reshape(U, X.shape)
    V = np.reshape(V, X.shape)
    plt.figure(figsize=(fig_size, fig_size))
    plt.streamplot(X, Y, U, V, density=2)
    plt.title(f"alpha= {alpha} ")
    if display:
        plt.show()
    if save_plots:
        plt.savefig(save_path)



def cusp_bifurcation():
    """
    Creates 3D and 2D plots of the cusp bifurcation 
    """
    # Sample (x, a2) uniformly
    a2_samples = [round(a2, 2) for a2 in np.random.uniform(0, 1.5, 50000)]
    x_samples = [round(x, 2) for x in np.random.uniform(-1.5, 1.5, 50000)]
    a1_samples = []
    solutions = {}
    for x, a2 in zip(x_samples, a2_samples):
        # Calculates a1 for every (x, a2)
        a1 = round(-a2 * x + x ** 3, 2)
        a1_samples.append(a1)
        
        # Keeps records of x for every (a1, a2)
        key = (a1, a2)
        if key in solutions:
            solutions[key].add(x)
        else:
            solutions[key] = {x}
    
    def create_axes(angle1, angle2, position):
        """
        Adds a 3D subplot to a figure given the angles of visualization and the position
        This subplot scatters the different samples 'a1_samples', 'a2_samples' and 'x_samples'
        :param angle1: first angle of visualization
        :param angle2: second angle of visualization
        :param position: position of the subplot in the figure
        """
        ax = fig.add_subplot(1, 3, position, projection='3d')
        ax.scatter(a1_samples, a2_samples, x_samples, cmap = "magma", c = a2_samples)
        ax.set_xlabel(r'$\alpha_1$')
        ax.set_ylabel(r'$\alpha_2$')
        ax.set_zlabel(r'$x$')
        ax.view_init(angle1, angle2)
        return ax
    
    # 3D plot
    fig = plt.figure(figsize=plt.figaspect(0.3))
    ax1 = create_axes(-110, 90, 1)
    ax2 = create_axes(110, -90, 2)
    ax3 = create_axes(-90, 90, 3)
    plt.show()
    
    # 2D plot
    fig = plt.figure()
    # Creates a colormap for every point (a1, a2)
    # If it has more than one solution x, a red point will be plotted, blue otherwise
    colors = ["orange" if len(solutions[(a1, a2)]) > 1 else "purple" for a1, a2 in zip(a1_samples, a2_samples)]
    ax2d = plt.axes()
    ax2d.set_xlabel(r'$\alpha_1$')
    ax2d.set_ylabel(r'$\alpha_2$')
    ax2d.scatter(a1_samples, a2_samples, s=1, c=colors)
    plt.show()