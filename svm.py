import numpy as np
from sklearn.svm import LinearSVC


### Functions for you to fill in ###

def one_vs_rest_svm(train_x, train_y, test_x):
    """
    Trains a linear SVM for binary classifciation

    Args:
        train_x - (n, d) NumPy array (n datapoints each with d features)
        train_y - (n, ) NumPy array containing the labels (0 or 1) for each training data point
        test_x - (m, d) NumPy array (m datapoints each with d features)
    Returns:
        pred_test_y - (m,) NumPy array containing the labels (0 or 1) for each test data point
    """

    #Code Here

    # Checkout SVM info on https: // scikit - learn.org / stable / modules / svm.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC

    clf = LinearSVC(random_state=0, C=0.05) #Stated to use random state = 0 and c = 0.1
    clf.fit(train_x, train_y) #Fit SVC model with training data X & Y
    pred_test_y = clf.predict(test_x) #Predict the fitted model with text data X
    return pred_test_y


    raise NotImplementedError


def multi_class_svm(train_x, train_y, test_x):
    """
    Trains a linear SVM for multiclass classifciation using a one-vs-rest strategy

    Args:
        train_x - (n, d) NumPy array (n datapoints each with d features)
        train_y - (n, ) NumPy array containing the labels (int) for each training data point
        test_x - (m, d) NumPy array (m datapoints each with d features)
    Returns:
        pred_test_y - (m,) NumPy array containing the labels (int) for each test data point
    """
    #Code Here

    # Checkout SVM info on https: // scikit - learn.org / stable / modules / svm.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC
    # multi_class{‘ovr’, ‘crammer_singer’}, default=’ovr’
    # Determines the multi - class strategy if y contains more than two classes."ovr" trains n_classes one-vs-rest classifiers,
    # while "crammer_singer" optimizes a joint objective over all classes.
    # While crammer_singer is interesting from a theoretical perspective as it is consistent,
    # it is seldom used in practice as it rarely leads to better accuracy and is more expensive to compute.
    # If "crammer_singer" is chosen, the options loss, penalty and dual will be ignored.

    clf = LinearSVC(random_state=0, C=0.1)  # Stated to use random state = 0 and c = 0.1
    clf.fit(train_x, train_y)  # Fit SVC model with training data X & Y
    pred_test_y = clf.predict(test_x)  # Predict the fitted model with text data X
    return pred_test_y
    raise NotImplementedError


def compute_test_error_svm(test_y, pred_test_y):
    return 1 - np.mean(pred_test_y == test_y)

