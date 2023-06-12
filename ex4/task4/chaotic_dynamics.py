import  numpy  as  np 
import  matplotlib.pyplot  as  plt
from scipy.integrate import solve_ivp
from matplotlib.pyplot import MultipleLocator



def  logistic_map (r, x):
    """
    define the logistic map function: x_n+1 = r * x_n * (1 - x_n) , n ∈ N
    :param r: reproduction rate
    :param x: x_n a number between zero and one,epresents the ratio of existing population to the maximum possible population.
    :returns: x_n+1 
    """
    return r * x * (1 - x)

def plot_logistic_map(r,x0):
    """
    plot the logistic map function with repspect to values of r
    :param r: reproduction rate
    :param x0: intitial value ot this dynamical system
    """
    n=100   # assume 100 values between r and x0 of this system.
    t = np.linspace(0, 1) 
    fig = plt.figure(figsize=(11 , 5))
    ax = fig.add_subplot(1,2,1)
    ax2= fig.add_subplot(1,2,2)


    ax.plot(t, logistic_map(r, t), 'k', lw=2) # plot function x_next = r * x * (1 - x)
    ax.plot([0, 1], [0, 1], 'k', lw=2) # plot functions x_next = x

    x= x0
    y=[] #store the position of x_n
    for i in range(n):
        y.append(x)
        x_next = logistic_map(r, x)
        # updating process x_n+1 = r * x_n * (1 - x_n)
        ax.plot([x, x], [x,x_next], 'k', lw=1)            # (x_n, x_n) -> (x_n, x_n+1)
        ax.plot([x, x_next], [x_next, x_next], 'k', lw=1) # (x_n, x_n+1) -> (x_n+1, x_n+1)
        # Plot the positions with increasing opacity.
        ax.plot([x], [x_next], 'ok', ms=8, alpha=(i + 1) / n)
        x = x_next

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("$x_n$",fontsize=12)
    ax.set_ylabel("$x_{n+1}$",fontsize=12)
    ax.set_title(f"$r={round(r,2)}, \, x_0={round(x0,2)}$")

    #plot x_n with respect to each update step
    ax2.plot(np.arange(0, 100, 1), y, 'ok-')
    ax2.set_xlabel("n",fontsize=12)
    ax2.set_ylabel("$x_n$",fontsize=12)
    ax2.set_title(f"r={round(r,2)}")

def plot_bifurcation_diagram(r_min,r_max):
    """
    Plot the bifurcations diagram for the logistic map system
    :param r_min: minimum r 
    :param r_max: maximum r
    """
    iter=1000 # 1000 iterations of the logistic map per run
    last=100  # select last 100 iterations to display the bifurcation diagram
    n=10000 # simulate this system for 10000 values of r
    r = np.linspace(r_min, r_max, n)

    fig, ax= plt.subplots(1, 1, figsize=(8, 4), sharex=True)

    x = 1e-5 * np.ones(n) # initial condition x0
    # plot the bifurcation diagram
    for i in range(iter):
        x = logistic_map(r, x)   # compute x_next
        if i >= (iter - last):
            ax.plot(r, x, ',k', alpha=.2)

    ax.set_xlim(r_min, r_max)
    ax.set_xlabel("r",fontsize=12)
    ax.set_ylabel("x",fontsize=12)
    ax.set_title("Bifurcation diagram")


  ###### part2 #######
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

def plot_lorenz_trajectory(X_init, sigma,beta,rho,ax=None):

    """
    plot the lorenz trajectory
    :param X_init: the initial point(x,y,z) , 3D vector
    :param sigma,beta,rho: constant coefficients in Lorenz
    :param ax: posion of the plot
    """
    t_start=0
    t_end=1000
    num_steps=100000
    x0,y0,z0=X_init

    # Integrate the Lorenz equations.
    soln = solve_ivp(lorenz_system,(t_start,t_end),(x0,y0,z0), args=(sigma, beta, rho),dense_output=True)
    # Interpolate solution onto the time grid, t.
    t = np.linspace(0, t_end, num_steps)
    x, y, z = soln.sol(t)

    # Plot the trajectory of Lorenz attractor in 3D
    # Make the line multi-coloured by plotting it in segments of length s which change in colour across the whole time series.
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

def comparsion (x1,x2):
    
    """
    plot the different bewtween the two lorenz trajectory X1 and X2 over time.
    :param x1,x2: two trajectories used to be comparaed
    """
   
    fig = plt.figure(figsize=(12, 6))
    
    # compute the difference
    d = x1.T - x2.T
    diff = np.linalg.norm(d, axis=1, keepdims=True) ** 2

    # plot the difference
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(np.arange(0, 1000, 0.01), diff)
    ax.set_xlabel("Time",fontsize=12)
    ax.set_ylabel("Difference",fontsize=12)
    ax.set_title("The difference between the given points on the trajectories")
     
     
    # compute when the difference larger than 1
    iter= 0
    for i, d in enumerate(diff):
        if d > 1:
            iter = i
            print(f"The difference between the given points on the trajectory larger than 1 at {round(0.01 * iter, 2)}s")

            # plot the difference  in more details
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.plot(np.arange(0, 1000, 0.01), diff)
            ax2.set_xlim(0, 40)
            ax2.set_ylim(0, 1)
            ax2.set_xlabel("Time",fontsize=12)
            ax2.set_ylabel("Difference",fontsize=12)
            ax2.set_title("zoom version on the trajectories larger than 1")
            break


# def plot_bifurcation_diagram_lorenz(X0, X0_hat, sigma, beta, rho_min, rho_max):
    
#     x0, y0, z0 = X0
#     x0_hat, y0_hat, z0_hat = X0_hat
#     rho_values = np.linspace(rho_min, rho_max, 500)
#     differences = []
#     bifurc_coords_r = []


#     for rho in rho_values:
#         sol1 = solve_ivp(lorenz_system, (0, 100), (x0, y0, z0), args=(sigma, beta, rho), dense_output=True)
#         sol2 = solve_ivp(lorenz_system, (0, 100), (x0_hat, y0_hat, z0_hat), args=(sigma, beta, rho), dense_output=True)
#         t = np.linspace(0, 100, 1000)
#         x, y, z = sol1.sol(t)
#         x_hat, y_hat, z_hat = sol2.sol(t)
#         traj_path1 = np.array([x, y, z])
#         traj_path2 = np.array([x_hat, y_hat, z_hat])
#         d = traj_path1.T - traj_path2.T
#         difference = np.linalg.norm(d, axis=1, keepdims=True) ** 2
#         differences.extend(difference)
#         bifurc_coords_r.extend([rho] * len(difference))


#     fig, ax = plt.subplots(figsize=(12, 6))
#     ax.plot(bifurc_coords_r, differences, 'k.', markersize=0.5)
#     ax.set_xlabel('rho')
#     ax.set_ylabel('Distance')
#     ax.set_title('Bifurcation Diagram - Lorenz System')

    
def bifurcation_diagram_lorenz(X0, sigma, beta, rho_min, rho_max):
    """
    Show the Stability of equilibrium point (x1, y1, z1) measured by z for the given intial condition against rho.
    """
    x0, y0, z0 = X0
    rho_values = np.linspace(rho_min, rho_max, 500)
    z_values = []
    r = []

    for rho in rho_values:
        sol = solve_ivp(lorenz_system, (0, 100), (x0, y0, z0), args=(sigma, beta, rho), dense_output=True)
        t = np.linspace(0, 100, 1000)
        x, y, z = sol.sol(t)
        z_values.append(z[-1])
        r.extend([rho] * len(z))

    fig, ax = plt.subplots(figsize=(12, 6))
    z_repeated = np.repeat(z_values, len(r) // len(z_values))
    ax.plot(r, z_repeated, 'k.', markersize=0.5)
    ax.set_xlabel('rho')
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    ax.set_ylabel('z')
    ax.set_title('Bifurcation Diagram - Lorenz System')


    