import numpy as np


def dpp_rerank(x_candidates: np.ndarray, candidate_scores: np.ndarray, top_k: float=10) -> list[int]:
    item_embs = x_candidates @ x_candidates.T

    selected = []
    remaining = list(range(len(candidate_scores)))
    for _ in range(top_k):
        best_idx = -1
        best_val = -1e9

        for i in remaining:
            if not selected:
                val = candidate_scores[i]
            else:
                sub_K = item_embs[np.ix_(selected, selected)]
                det_old = np.linalg.det(sub_K + np.eye(len(selected))*1e-6)
                sub_K_new = item_embs[np.ix_(selected+[i], selected+[i])]
                det_new = np.linalg.det(sub_K_new + np.eye(len(selected)+1)*1e-6)
                val = candidate_scores[i] + np.log(det_new+1e-9) - np.log(det_old+1e-9)

            if val > best_val:
                best_val = val
                best_idx = i

        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected
