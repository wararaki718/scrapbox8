import numpy as np


def get_features(n_data: int, n_dim: int) -> np.ndarray:
    return np.random.rand(n_data, n_dim)


def get_labels(n_data: int, n_classes: int=2) -> np.ndarray:
    return np.random.randint(0, n_classes, size=n_data)


def get_pairs(n_data: int, n_users: int, n_items: int) -> np.ndarray:
    users = np.random.randint(0, n_users, size=n_data)
    items = np.random.randint(0, n_items, size=n_data)
    return np.column_stack((users, items))
