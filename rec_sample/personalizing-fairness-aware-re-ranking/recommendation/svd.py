import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds


class SVDRecommender:
    def compute(self, rating_df: pd.DataFrame, k: int = 20) -> None:
        user_ids = rating_df['user_id'].unique()
        self._item_ids = rating_df['item_id'].unique()
        self._user2index = {user_id: index for index, user_id in enumerate(user_ids)}
        item2index = {item_id: index for index, item_id in enumerate(self._item_ids)}

        matrix = np.zeros((len(user_ids), len(self._item_ids)))
        for row in rating_df.itertuples():
            user_index = self._user2index[row.user_id]
            item_index = item2index[row.item_id]
            matrix[user_index, item_index] = row.rating

        u, sigma, vt = svds(matrix, k=k)
        sigma = np.diag(sigma)
        self._predicted_ratings: np.ndarray = np.dot(np.dot(u, sigma), vt)

    def score_items(self, user_id: int, top_n: int=50) -> tuple[list[int], dict[int, float]]:
        """
        Score items for a given user based on SVD.
        Args:
            user_id (int): The ID of the user for whom to score items.
            top_n (int, optional): Number of top items to return. Defaults to 50.
        Returns:
            tuple[list[int], dict[int, float]]: A tuple containing a list of top N item IDs and a dictionary of item scores.
        """
        assert self._predicted_ratings is not None, "Please run compute() before score_items()."

        user_index = self._user2index[user_id]
        scores = self._predicted_ratings[user_index, :].copy()

        indices = np.argsort(scores)[::-1][:top_n]
        ranked_items = self._item_ids[indices]

        score_dict = {item_id: scores[i] for i, item_id in enumerate(self._item_ids)}
        return ranked_items.tolist(), score_dict
