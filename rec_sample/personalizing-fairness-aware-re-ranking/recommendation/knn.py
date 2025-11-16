import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class KNNRecommender:
    def compute(self, rating_df: pd.DataFrame) -> None:
        self._mat = rating_df.pivot(index="user_id", columns="item_id", values="rating").fillna(0)
        self._similarities = cosine_similarity(self._mat)

    def score_items(self, user_id: int, top_n: int=50) -> tuple[list[int], dict[int, float]]:
        """
        Score items for a given user based on KNN.
        Args:
            user_id (int): The ID of the user for whom to score items.
            top_n (int, optional): Number of top items to return. Defaults to 50.
        Returns:
            tuple[list[int], dict[int, float]]: A tuple containing a list of top N item IDs and a dictionary of item scores.
        """

        assert self._mat is not None and self._similarities is not None, "Please run compute() before score_items()."

        user_index = user_id - 1
        user_ratings = self._mat.values
        sim_vec = self._similarities[user_index]

        # Calculate scores
        scores = sim_vec @ user_ratings / (np.abs(sim_vec).sum() + 1e-6)

        # Exclude already rated items
        rated_items = (user_ratings[user_index] > 0)
        scores[rated_items] = -1

        # Get top N items
        item_ids = self._mat.columns
        ranked_items = item_ids[np.argsort(scores)[::-1][:top_n]]

        score_dict = {item: scores[i] for i, item in enumerate(item_ids)}
        return list(ranked_items), score_dict
