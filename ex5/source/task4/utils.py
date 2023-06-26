import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg, integrate,spatial


def lorenz_system(t,X_init,sigma,beta,rho):
     """
    define the Lorenz system functions
    :param X_init: the initial point(x,y,z) , 3D vector
    :param sigma,beta,rho: constant coefficients in Lorenz
    :return: x_dot, y_dot, z_dot: partial derivatives values of at the point X_init
    """
     x, y, z = X_init
     dx_dt= sigma*(y-x)  
     dy_dt= x*(rho-z)-y  
     dz_dt= x*y - beta*z
     dX_init_dt= np.array([dx_dt,dy_dt,dz_dt])

     return  dX_init_dt

def plot_lorenz_trajectory(X_init, sigma,beta,rho,t_end):

    """
    plot the lorenz trajectory
    :param X_init: the initial point(x,y,z) , 3D vector
    :param sigma,beta,rho: constant coefficients in Lorenz
    :param ax: posion of the plot
    """
    t_start=0
    num_steps=100000
    x0,y0,z0=X_init

    # Integrate the Lorenz equations.
    soln = integrate.solve_ivp(lorenz_system,(t_start,t_end),(x0,y0,z0), args=(sigma, beta, rho),dense_output=True)
    # Interpolate solution onto the time grid, t.
    t = np.linspace(0, t_end, num_steps)
    x, y, z = soln.sol(t)

    # Plot the trajectory of Lorenz attractor in 3D
    # Make the line multi-coloured by plotting it in segments of length s which change in colour across the whole time series.
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    s = 100
    cmap = plt.cm.winter
    for i in range(0,num_steps-s,s):
      ax.plot(x[i:i+s+1], y[i:i+s+1], z[i:i+s+1], color=cmap(i/num_steps), alpha=0.4)
     
    ax.plot(x[0], y[0], z[0], 'o', color="r", label="initial point")
    ax.plot(x[-1], y[-1], z[-1], 'o', color="g", label="end point")
    
    ax.set_xlabel("x",fontsize=12)
    ax.set_ylabel("y",fontsize=12)
    ax.set_zlabel("z",fontsize=12)
    ax.legend()
    ax.set_title(f"$sigma={sigma}, beta={round(beta,2)}, rho={rho}$")

    traj_path = np.array([x,y,z])
    return traj_path

def plot_dataset(X: np.ndarray,ax,is2D=None):
#     fig, ax = plt.subplots(figsize=(6, 5))
    if is2D:
      ax.plot(X[:,0],X[:,1])
      ax.set_xlabel(r"$X0(t)$")
      ax.set_ylabel(r"$X1(t)$")   
      ax.set_title(r"The mainfold in 2D")
      ax.grid()
    else:
      ax.plot(X)
      ax.set_xlabel("t: Time Steps")
      ax.set_ylabel(r"$X0(t)$")   
      ax.set_title(r"Periodic Data  $X_0$  against t")
      ax.grid()

def plot_delayed_ver(X: np.ndarray,  delta_t: np.ndarray, ax=None):
    """
    Plots the specified coordinate against its delayed version with different delta_t

    Args:
        X : the specified coordinate
        delta_t: the delay time Delta t
    """ 
    if ax == None:
     fig = plt.figure(figsize=(20, 5*len(delta_t)))  
     for i in range(len(delta_t)):
          x = np.array([X, np.roll(X, shift=delta_t[i]), np.roll(X, shift=2 * delta_t[i])])
          ax = fig.add_subplot(len(delta_t), 4, i+1,projection='3d')
          ax.plot(x[0], x[1],x[2])
          ax.set_xlabel("$x$")
          ax.set_ylabel("$x+ \Delta t$")
          ax.set_title(f"$\Delta t ={delta_t[i]}$")
    else:
         x = np.array([X, np.roll(X, shift=delta_t), np.roll(X, shift=2 * delta_t)])
         ax.plot(x[0], x[1],x[2])
         ax.set_xlabel("$x$")
         ax.set_ylabel("$x+ \Delta t$")
         ax.set_title(f"$\Delta t ={delta_t}$")

def plot_delayed_ver_lorenz(X: np.ndarray,  delta_t: np.ndarray,isY=bool,isZ=bool):
    """
    Plots the specified coordinate against its delayed version with different delta_t

    Args:
        X : the specified coordinate
        delta_t: the delay time Delta t
    """ 

    fig = plt.figure(figsize=(27, 6*len(delta_t)))  
    for i in range(len(delta_t)):
       if isZ:
          z=X[:,2]
          x = np.array([z, np.roll(z, shift=delta_t[i]), np.roll(z, shift=2 * delta_t[i])])
          ax = fig.add_subplot(len(delta_t), 4, i+1, projection='3d')
          ax.plot(x[0], x[1], x[2])
          ax.set_xlabel("$z$")
          ax.set_ylabel("$z+ \Delta t$")
          ax.set_zlabel("$z+ 2\Delta t$")
          ax.set_title(f"$\Delta t ={delta_t[i]}$")
       elif isY:
          y=X[:,1]
          x = np.array([y, np.roll(y, shift= delta_t[i]), np.roll(y, shift= 2 * delta_t[i])])
          ax = fig.add_subplot(len(delta_t), 4, i+1,projection='3d')
          ax.plot(x[0], x[1], x[2])
          ax.set_xlabel("$y$")
          ax.set_ylabel("$y+ \Delta t$")
          ax.set_zlabel("$y+ 2\Delta t$")
          ax.set_title(f"$\Delta t ={delta_t[i]}$")
       else:
          x=X[:,0]
          x = np.array([x, np.roll(x, shift= delta_t[i]), np.roll(x, shift= 2 * delta_t[i])])
          ax = fig.add_subplot(len(delta_t), 4, i+1,projection='3d')
          ax.plot(x[0], x[1], x[2])
          ax.set_xlabel("$x$")
          ax.set_ylabel("$x+ \Delta t$")
          ax.set_zlabel("$x+ 2\Delta t$")
          ax.set_title(f"$\Delta t ={delta_t[i]}$")


############################################# Bonus Part ######################################################

def radial_basis_fun(X:np.ndarray, X_l:np.ndarray,eps:int)-> np.ndarray:
    """
    radial basic function
    :param X: point
    :param X_l: center of the basis function
    :param eps: radius of gaussians
    :return: matrix contains radial basic function
    """
    out = np.exp(-(spatial.distance_matrix(X, X_l) / eps) ** 2)
    return out

def radial_basis_approx(X: np.ndarray, F: np.ndarray,center:np.ndarray,epsilon: float, cut: float) -> np.ndarray:
    """
    Calculates the optimal coefficients c for the ansatz function:
    f(x) = sum_{l=1}^L c_l phi_l(x) through least squares algorothm

    Args:
        X:  input data array.
        F: data array of dependent parameters (label)
        center: center points of radial basis functions
        epsilon: radius of gaussians
        cut : Cutoff for 'small' singular values

    Returns:
        c :the optimal coefficients 
    """
    
    phi_X = radial_basis_fun(X,center, epsilon)
    matrix = phi_X.T @ phi_X
    target = phi_X.T @ F
    c, _, _, _ = linalg.lstsq(matrix, target, cond=cut)
    return c

def radial_trajectory(X: np.ndarray,X_l: np.ndarray, epsilon, coefficients, t_end: float) -> None:
    """
    integrate x,y,z with respect to t by solve_ivp and plot the trajectory.
    """
    
    def radial(t, y):
        y = y.reshape((1, y.size))
        return radial_basis_fun(y,X_l, epsilon) @ coefficients
     
    x0,y0,z0=X

    # integrate the Lorenz equations
    t = np.linspace(0, t_end, 10000)
    soln = integrate.solve_ivp(radial, (0,t_end), (x0,y0,z0), dense_output=True)
    x, y, z = soln.sol(t)

    # plot the trajectory in three-dimensional phase space
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    s = 100
    cmap = plt.cm.winter
    for i in range(0,10000-s,s):
     ax.plot(x[i:i+s+1], y[i:i+s+1], z[i:i+s+1], color=cmap(i/10000), alpha=0.4)
    ax.plot(x[0], y[0], z[0], 'o', color="r", label="initial point")
    ax.plot(x[-1], y[-1], z[-1], 'o', color="g", label="end point")
    ax.set_xlabel("$x(t)$",fontsize=12)
    ax.set_ylabel("$x(t + \Delta t)$",fontsize=12)
    ax.set_zlabel("$x(t + 2*\Delta t)$",fontsize=12)
    ax.legend()
    ax.set_title("trajectories in the approximated vector field")