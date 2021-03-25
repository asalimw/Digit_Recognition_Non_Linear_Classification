import numpy as np
import scipy.spatial.distance as distance
### Functions for you to fill in ###



def polynomial_kernel(X, Y, c, p):
    """
        Compute the polynomial kernel between two matrices X and Y::
            K(x, y) = (<x, y> + c)^p
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            c - a coefficient to trade off high-order and low-order terms (scalar)
            p - the degree of the polynomial kernel

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    # YOUR CODE HERE

    kernel_matrix = (((X @ Y.T) +c)** p) #K(x, y) = (<x, y> + c)^p
    return kernel_matrix
    raise NotImplementedError



def rbf_kernel(X, Y, gamma):
    """
        Compute the Gaussian RBF kernel between two matrices X and Y::
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            gamma - the gamma parameter of gaussian function (scalar)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    # YOUR CODE HERE
    n = len(X) #The row size of X
    m = len(Y) #The row size of Y
    dist = np.zeros((n,m)) #Initial (n,m) zeros array
    dist_eu = np.zeros((n, m))  # Initial (n,m) zeros array

    # for i in range(n):
    #     for j in range(m):
    #         dist[i][j] = np.linalg.norm(X[i] - Y[j]) #each pair of rows x in X and y in Y into dist array.
    # print(dist.shape)

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
    # Compute distance between each pair of the two collections of inputs.
    dist_eu = distance.cdist(X, Y, 'euclidean') #Faster method

    # print (dist_eu.shape)
    kernel_matrix = np.exp(-gamma * dist_eu**2) #K(x, y) = exp(-gamma ||x-y||^2)
    # print(kernel_matrix.shape)
    return kernel_matrix

    raise NotImplementedError
