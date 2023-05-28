import numpy as np

def pca(data):
    """
    Parameters
    ----------
    data: np.ndarray, shape [N, D]
        dataset be do SVD
    Returns
    -------
    U: numpy.ndarray, shape [N, N]
        u matrix
    S: numpy.ndarray, shape [N, D]
        sigma matrix
    Vh: numpy.ndarray,  shape [D,D]
        transpose of v matrix
    energy: numpy.ndarray,  shape [D,1]
        the energy matrix
    """
    #center the data
    mean = data.mean(axis=0)
    data_centered = data - mean

    #decompose the data through SVD
    U, s, Vh = np.linalg.svd(data_centered)

    #construct the S matrix
    s_square = np.square(s)
    trace = np.sum(s_square)
    S = np.zeros(shape=(data_centered.shape[0], data_centered.shape[1]))
    s = np.diag(s)
    S[:s.shape[0], :s.shape[1]] = s

    #compute the energy for each principal component
    energy = s_square/trace

    return U, S, Vh, energy

def center_data(data: Union[np.ndarray, pd.DataFrame]):
    """
    Centers the data, i.e. removes the mean form it
    :param data: data to center
    :returns: the centered data in the form of a numpy.ndarray
    """
    return data - np.mean(data, axis=0)

def svd(data: Union[np.ndarray, pd.DataFrame], center=False):
    """
    Compute the Singular Value Decomposition (SVD) of the "data"
    :param data: data to compute the SVD of
    :param center: if True, center the data before performing SVD
    :returns: the 3 matrices forming the SVD decomposition of "data"
    """
    # make the data a numpy ndarray (if it isn't already)
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()

    # center the data by removing the mean
    if center:
        data = center_data(data)

    # decompose the data through SVD decomposition
    U, singular_values, Vt = np.linalg.svd(data)  # note that V is already transpose
    # starting from a vector containing the singular values, create the S matrix
    S = np.vstack((
        np.diag(singular_values),
        np.zeros(shape=(data.shape[0] - len(singular_values), len(singular_values)))
    ))
    return U, S, Vt.T


def plot_principal_directions(data: np.ndarray, V: np.ndarray):
    """
    Displays a 2D scatter plot of the data and adds the directions of the 2 principal components
    :param data: data to be plotted
    :param V: V matrix coming from a SVD decomposition of data, its columns contain the directions of the principal components
    """
    plt.scatter(data[:, 0], data[:, 1], label='Centered data')
    origin = np.mean(data, axis=0)
    x_values = [origin[0], V[:, 0][0]]
    y_values = [origin[1], V[:, 0][1]]
    plt.plot(x_values, y_values, color='r')
    x_values = [origin[0], V[:, 1][0]]
    y_values = [origin[1], V[:, 1][1]]
    plt.plot(x_values, y_values, color='r')
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Scatter plot of the data")
    plt.legend()
    plt.show()


def load_racoon(display=False):
    """
    Loads the image of the racoon
    :param display: if True, displays the image, otherwise simply returns it
    :returns: the image (np.ndarray)
    """
    img = scipy.misc.face(gray=True)
    img = cv2.resize(img, dsize=(249, 185))  # rescale image
    if display:
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.show()
    return img


def show_2_images_grayscale(img1: np.ndarray, img2: np.ndarray, vmin=0, vmax=255, titles: Tuple[str, str] = ("", "")):
    """
    Utility function to show to images, in grayscale, one next to the other
    :param img1: first image
    :param img2: second image
    :param vmin: minimum intensity value of the images in grayscale
    :param vmax: maximum intensity value of the images in grayscale
    :param titles: tuple containing the titles of the 2 images
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 15))
    ax[0].imshow(img1, cmap='gray', vmin=vmin, vmax=vmax)
    ax[0].set_title(titles[0])
    ax[1].imshow(img2, cmap='gray', vmin=vmin, vmax=vmax)
    ax[1].set_title(titles[1])
    plt.show()


def plot_2_trajectories(traj1: np.ndarray, traj2: np.ndarray):
    """
    Plots the trajectories of 2 pedestrians in a 2D space
    :param traj1: np.ndarray containing the 1st trajectory -> rows are time steps and columns are x and y position
    :param traj2: np.ndarray containing the 2nd trajectory -> rows are time steps and columns are x and y position
    """
    plt.plot(traj1[:, 0], traj1[:, 1], label="Pedestrian 1", color='tab:blue')
    plt.plot(traj2[:, 0], traj2[:, 1], label="Pedestrian 2", color='tab:orange')
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Trajectories of 2 pedestrians")
    plt.show()