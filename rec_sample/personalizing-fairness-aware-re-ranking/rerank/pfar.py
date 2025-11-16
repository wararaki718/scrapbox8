import numpy as np
import pandas as pd

# personalized fairness-aware
class PfarReranker:
    def compute(self, rating_df: pd.DataFrame, movie_df: pd.DataFrame) -> None:
        """
        Precompute tau values for each user based on their rating history.

        Args:
            rating_df (pd.DataFrame): DataFrame containing user-item ratings.
            movie_df (pd.DataFrame): DataFrame containing movie information.
        """
        merge_df = rating_df.merge(
            movie_df[['movie_id','provider']],
            left_on='item_id',
            right_on='movie_id',
        )

        self._tau = dict()
        for user_id, group in merge_df.groupby("user_id"):
            provider_counts = group["provider"].value_counts().values
            p = provider_counts / provider_counts.sum()
            self._tau[user_id] = -np.sum(p * np.log(p))
    
    def rerank(
        self,
        user_id: int,
        movie_ids: list[int],
        movie_scores: dict[int, float],
        movie2provider: dict[int, str],
        lambda_: float = 0.5,
        top_n: int = 10,
    ) -> list[int]:
        """
        Rerank items for a user to enhance diversity.
        
        Args:
            user_id (int): The ID of the user for whom to rerank items.
            movie_ids (list[int]): List of movie IDs to consider for reranking.
            movie_scores (dict[int, float]): Dictionary mapping movie IDs to their base scores.
            movie2provider (dict[int, str]): Dictionary mapping movie IDs to their providers.
            lambda_ (float, optional): Weight for the diversity bonus. Defaults to 0.5.
            top_n (int, optional): Number of top items to return. Defaults to 10.
        Returns:
            list[int]: Reranked list of movie IDs.
        """
        assert self._tau is not None, "Please run compute() before rerank()."

        result = []
        selected_providers = set()
        tau = self._tau.get(user_id, 0.0)

        for _ in range(top_n):
            best_item = None
            best_score = -float("inf")

            for movie_id in movie_ids:
                if movie_id in result:
                    continue

                provider = movie2provider.get(movie_id, "Unknown")
                base_score = movie_scores.get(movie_id, 0.0)

                # calculate diversity bonus
                diversity_bonus = 1.0 if provider not in selected_providers else 0.0

                # calculate final score
                score = base_score + lambda_ * tau * diversity_bonus
    
                if score > best_score:
                    best_score = score
                    best_item = movie_id

            if best_item is None:
                break

            result.append(best_item)
            selected_providers.add(movie2provider[best_item])

        return result