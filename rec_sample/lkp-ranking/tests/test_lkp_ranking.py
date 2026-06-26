import itertools
import unittest

import torch

from loss import PersonalizedKDPPNLLLoss
from model import PersonalizedKDPP
from utils import log_k_dpp_probability, sum_k_dpp_partition


class TestLKPRanking(unittest.TestCase):
    def test_log_probability_matches_bruteforce(self) -> None:
        V = torch.tensor(
            [
                [1.0, 0.2],
                [0.1, 0.9],
                [0.6, 0.4],
                [0.4, 1.1],
            ],
            dtype=torch.float64,
        )
        L = V @ V.T

        basket = [0, 2]
        log_prob = log_k_dpp_probability(L, basket=basket, k=2)

        denom = sum_k_dpp_partition(L, k=2)
        numer = torch.det(L[basket][:, basket])
        brute_force = torch.log(numer) - torch.log(denom)

        self.assertTrue(torch.allclose(log_prob, brute_force, atol=1e-8, rtol=1e-8))

    def test_negative_log_likelihood_is_finite(self) -> None:
        model = PersonalizedKDPP(num_users=2, num_items=5, rank=3, k=2)
        criterion = PersonalizedKDPPNLLLoss(l2_weight=model.l2_weight)
        user_ids = torch.tensor([0, 1], dtype=torch.long)
        baskets = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)

        loss = criterion(model, user_ids=user_ids, baskets=baskets)

        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(loss.item(), 0.0)

    def test_recommendation_excludes_seen_items(self) -> None:
        torch.manual_seed(0)
        model = PersonalizedKDPP(num_users=1, num_items=8, rank=4, k=3)

        recs = model.recommend_next_items(user_id=0, seen_items=[0, 1, 2], top_k=3)

        self.assertEqual(len(recs), 3)
        self.assertEqual(len(set(recs)), 3)
        self.assertEqual(set(recs) & {0, 1, 2}, set())


if __name__ == "__main__":
    unittest.main()
