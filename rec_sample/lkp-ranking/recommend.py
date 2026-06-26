from __future__ import annotations

from collections.abc import Sequence

from model import PersonalizedKDPP


def show_recommendation_example(model: PersonalizedKDPP, user_id: int, seen_items: Sequence[int]) -> None:
    recs = model.recommend_next_items(user_id=user_id, seen_items=seen_items, top_k=5)
    print("--- recommendation ---")
    print(f"user={user_id}")
    print(f"seen={list(seen_items)}")
    print(f"next_items={recs}")
