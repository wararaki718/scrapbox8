import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def mmr_rerank(x_candidates: np.ndarray, rel_scores: np.ndarray, lambda_: float=0.5, top_k: int = 10) -> list[int]:
    selected = []
    remaining = list(range(len(x_candidates)))

    for _ in range(top_k):
        best_item = None
        best_score = -1e9

        for idx in remaining:
            relevance = rel_scores[idx]

            if len(selected) == 0:
                diversity_penalty = 0.0
            else:
                similarity: float = max(
                    cosine_similarity(
                        x_candidates[idx].reshape(1, -1),
                        x_candidates[j].reshape(1, -1)
                    )[0][0] for j in selected
                )
                diversity_penalty = similarity

            score = lambda_ * relevance - (1 - lambda_) * diversity_penalty

            if score > best_score:
                best_score = score
                best_item = idx

        selected.append(best_item)
        remaining.remove(best_item)

    return selected
