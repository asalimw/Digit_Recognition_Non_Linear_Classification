import sys
sys.path.append("..")
import utils
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from scipy.sparse import coo_matrix


def augment_feature_vector(X):
    """
    Adds the x[i][0] = 1 feature for each data point x[i].

    Args:
        X - a NumPy matrix of n data points, each with d - 1 features

    Returns: X_augment, an (n, d) NumPy array with the added feature for each datapoint
    """
    column_of_ones = np.zeros([len(X), 1]) + 1
    return np.hstack((column_of_ones, X))

def compute_probabilities(X, theta, temp_parameter):
    """
    Computes, for each datapoint X[i], the probability that X[i] is labeled as j
    for j = 0, 1, ..., k-1

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        theta - (k, d) NumPy array, where row j represents the parameters of our model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)
    Returns:
        H - (k, n) NumPy array, where each entry H[j][i] is the probability that X[i] is labeled as j
    """
    # https://en.wikipedia.org/wiki/Softmax_function
    #YOUR CODE HERE
    expo = np.dot(theta, X.T)
    c = np.max(expo, axis=0)/temp_parameter #This is to fix numerical error due to exp function
    a = expo/temp_parameter - c # theta * X /Temp formula
    H = np.exp(a) / np.sum(np.exp(a), axis=0) #Python Formulae from Wikipedia
    # print(H, H.shape)
    return H

    raise NotImplementedError

def compute_cost_function(X, Y, theta, lambda_factor, temp_parameter):
    """
    Computes the total cost over every datapoint.

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns
        c - the cost value (scalar)
    """
    #YOUR CODE HERE


    # https://en.wikipedia.org/wiki/Loss_function
    # https://numpy.org/doc/stable/reference/generated/numpy.choose.html

    n = len(X)
    softmax = compute_probabilities(X, theta, temp_parameter) #For the log term using previous function
    # selected_softmax = np.choose(Y, softmax) #Choose softmax based on Y input selection
    selected_softmax = softmax[Y, np.arange(n)]
    # loss = -1/n*(sum(np.log(selected_softmax))) #Empirical Loss formula from project
    loss = -np.mean(np.log(selected_softmax))
    regularization = lambda_factor/2 * (np.sum(theta**2))   #Regularization formula from project
    c = loss + regularization #Formula supplied by the project

    return c

    raise NotImplementedError

def run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter):
    """
    Runs one step of batch gradient descent

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
    """
    #YOUR CODE HERE

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html#scipy.sparse.coo_matrix
    # https://www.youtube.com/watch?v=Lhef_jxzqCg - sparse matrix
    # https://en.wikipedia.org/wiki/Sparse_matrix

    n = len(X) #Number of element in X
    k = len(theta) #Number of element in theta

    prob = compute_probabilities(X, theta, temp_parameter) #Probability of X

    # scipy example of coo_matrix
    # row  = np.array([0, 0, 1, 3, 1, 0, 0])
    # col  = np.array([0, 2, 1, 3, 1, 0, 0])
    # data = np.array([1, 1, 1, 1, 1, 1, 1])
    # coo_matrix((data, (row, col)), shape=(4, 4)).toarray()
    inner = sparse.coo_matrix((n * [1], (Y, range(n))), shape=(k,n)).toarray() #y(i) == m

    grad_desc = -1/(temp_parameter*n)*((inner - prob).dot(X)) #Theta after Gradient Descent

    regularization = lambda_factor * theta #Regularization equation provided by the project

    cost_grad = grad_desc + regularization #Cost function after gradient descent
    theta -= alpha * cost_grad #Run gradient descent, update  θ  at each step
    return theta

    raise NotImplementedError

def update_y(train_y, test_y):
    """
    Changes the old digit labels for the training and test set for the new (mod 3)
    labels.

    Args:
        train_y - (n, ) NumPy array containing the labels (a number between 0-9)
                 for each datapoint in the training set
        test_y - (n, ) NumPy array containing the labels (a number between 0-9)
                for each datapoint in the test set

    Returns:
        train_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                     for each datapoint in the training set
        test_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                    for each datapoint in the test set
    """
    #YOUR CODE HERE

    # https://www.freecodecamp.org/news/the-python-modulo-operator-what-does-the-symbol-mean-in-python-solved/
    # print(train_y, train_y.shape)
    # print(test_y, test_y.shape)

    train_y_mod3 = train_y % 3
    test_y_mod3 = test_y % 3
    return train_y_mod3, test_y_mod3

    raise NotImplementedError

def compute_test_error_mod3(X, Y, theta, temp_parameter):
    """
    Returns the error of these new labels when the classifier predicts the digit. (mod 3)

    Args:
        X - (n, d - 1) NumPy array (n datapoints each with d - 1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-2) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        test_error - the error rate of the classifier (scalar)
    """
    #YOUR CODE HERE

    # 965:002 = 0.66667 error
    # 868:202 = 0.3333 error

    y_predicted_mod3 = get_classification(X, theta, temp_parameter) % 3
    # test_error = 1 - np.mean(y_predicted_mod3 == Y)
    test_error = np.mean(Y != y_predicted_mod3)
    return test_error

    # print (y_predicted)
    # print (y_predicted_mod3)
    # pring(test_error)
    raise NotImplementedError

def softmax_regression(X, Y, temp_parameter, alpha, lambda_factor, k, num_iterations):
    """
    Runs batch gradient descent for a specified number of iterations on a dataset
    with theta initialized to the all-zeros array. Here, theta is a k by d NumPy array
    where row j represents the parameters of our model for label j for
    j = 0, 1, ..., k-1

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d-1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        temp_parameter - the temperature parameter of softmax function (scalar)
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        k - the number of labels (scalar)
        num_iterations - the number of iterations to run gradient descent (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
        cost_function_progression - a Python list containing the cost calculated at each step of gradient descent
    """
    X = augment_feature_vector(X)
    theta = np.zeros([k, X.shape[1]])
    cost_function_progression = []
    for i in range(num_iterations):
        cost_function_progression.append(compute_cost_function(X, Y, theta, lambda_factor, temp_parameter))
        theta = run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter)
    return theta, cost_function_progression

def get_classification(X, theta, temp_parameter):
    """
    Makes predictions by classifying a given dataset

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d - 1 features)
        theta - (k, d) NumPy array where row j represents the parameters of our model for
                label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        Y - (n, ) NumPy array, containing the predicted label (a number between 0-9) for
            each data point
    """
    X = augment_feature_vector(X)
    probabilities = compute_probabilities(X, theta, temp_parameter)
    return np.argmax(probabilities, axis = 0)

def plot_cost_function_over_time(cost_function_history):
    plt.plot(range(len(cost_function_history)), cost_function_history)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    plt.show()

def compute_test_error(X, Y, theta, temp_parameter):
    error_count = 0.
    assigned_labels = get_classification(X, theta, temp_parameter)
    return 1 - np.mean(assigned_labels == Y)
