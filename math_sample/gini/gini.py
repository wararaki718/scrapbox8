import numpy as np


def gini(X: np.ndarray) -> float:
    if X.shape[0] == 0:
        return -1.0
    
    score = 0.0
    for x_i in X:
        for x_j in X:
            score += abs(x_i - x_j)
    score /= (2 * X.shape[0] * np.sum(X))
    return score


def gini2(X: np.ndarray) -> float:
    if X.shape[0] == 0:
        return -1.0

    X = np.sort(X)
    n = len(X)
    score = 0.0
    for i, x in enumerate(X, start=1):
        score += x * (2 * i - n - 1)
    score /= (n * np.sum(X))
    return score


def gini3(X: np.ndarray) -> float:
    if X.shape[0] == 0:
        return -1.0

    X = np.sort(X)
    n = len(X)
    cumsum = np.cumsum(X)
    score = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    return score


def gini4(X: np.ndarray) -> float:
    if X.shape[0] == 0:
        return -1.0

    X = np.sort(X)
    n = len(X)
    X_mean = np.mean(X)
    score = 2 * np.sum(np.arange(1, n + 1) * (X - X_mean)) / (n * n * X_mean)
    return score
