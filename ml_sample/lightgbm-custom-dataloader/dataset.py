import numpy as np
from lightgbm import Sequence


class CustomDataset(Sequence):
    def __init__(self, users: np.ndarray, items: np.ndarray, pairs: np.ndarray) -> None:
        self._users = users
        self._items = items
        self._pairs = pairs
    
    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, index: int) -> np.ndarray:
        pair = self._pairs[index]
        x = np.concatenate((self._users[pair[0]], self._items[pair[1]]))
        return x
