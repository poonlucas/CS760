import numpy as np
import matplotlib.pyplot as plt


def buggy_pca(X, d):
    D = X.shape[1]
    U, S, Vh = np.linalg.svd(X)
    V = Vh[range(d)].reshape((D, d))
    Z = np.matmul(X, V)
    new_X = np.matmul(Z, V.T)
    return V, Z, new_X


def demeaned_pca(X, d):
    mean = np.mean(X, axis=0)
    X_mean = X - mean
    V, Z, new_X = buggy_pca(X_mean, d)
    return V, Z, (new_X + mean)


def normalized_pca(X, d):
    mean = np.mean(X, axis=0)
    stdev = np.std(X, axis=0)
    X_normalized = (X - mean) / stdev
    V, Z, new_X = buggy_pca(X_normalized, d)
    return V, Z, (new_X * stdev) + mean


def plot_pca_2d(X, new_X, ax):
    ax.scatter(X[:, 0], X[:, 1], c="blue", marker='o')
    ax.scatter(new_X[:, 0], new_X[:, 1], c="red", marker='+')


def dro(X, d):
    D = X.shape[1]
    mean = np.mean(X, axis=0)
    X_mean = X - mean
    U, S, Vh = np.linalg.svd(X_mean)
    V = Vh[range(d)].reshape((D, d))
    Z = np.matmul(X_mean, V)
    return mean, V, Z


def reconstruction_error(X, new_X):
    return np.linalg.norm(X - new_X) ** 2
