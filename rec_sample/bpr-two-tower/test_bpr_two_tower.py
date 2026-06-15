import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent))

from dataset import build_synthetic_bpr_data
from metrics import recall_at_k
from model import TwoTowerModel
from trainer import Trainer


def test_two_tower_forward_shapes() -> None:
    model = TwoTowerModel(user_feat_dim=5, item_feat_dim=7, emb_dim=8, hidden_dim=16)

    x_user = torch.randn(4, 5)
    x_item = torch.randn(4, 7)

    scores = model(x_user, x_item)

    assert scores.shape == (4,)


def test_bpr_loss_prefers_higher_positive_scores() -> None:
    from loss import bpr_loss

    pos_scores = torch.tensor([3.0, 2.0], dtype=torch.float32)
    neg_scores = torch.tensor([1.0, 0.5], dtype=torch.float32)

    good_loss = bpr_loss(pos_scores=pos_scores, neg_scores=neg_scores)

    pos_scores_bad = torch.tensor([0.5, 0.1], dtype=torch.float32)
    neg_scores_bad = torch.tensor([1.0, 0.5], dtype=torch.float32)

    bad_loss = bpr_loss(pos_scores=pos_scores_bad, neg_scores=neg_scores_bad)

    assert good_loss.item() < bad_loss.item()


def test_single_step_training_improves_margin() -> None:
    torch.manual_seed(7)

    model = TwoTowerModel(user_feat_dim=2, item_feat_dim=2, emb_dim=4, hidden_dim=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    from loss import bpr_loss

    x_user = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    x_pos = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    x_neg = torch.tensor([[0.0, 1.0]], dtype=torch.float32)

    with torch.no_grad():
        margin_before = (model(x_user, x_pos) - model(x_user, x_neg)).item()

    for _ in range(100):
        optimizer.zero_grad()
        pos_scores = model(x_user, x_pos)
        neg_scores = model(x_user, x_neg)
        loss = bpr_loss(pos_scores=pos_scores, neg_scores=neg_scores)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        margin_after = (model(x_user, x_pos) - model(x_user, x_neg)).item()

    assert margin_after > margin_before


def test_recall_at_10_improves_with_training_on_synthetic_data() -> None:
    torch.manual_seed(42)

    bundle = build_synthetic_bpr_data(
        num_users=16,
        num_items=80,
        user_latent_dim=6,
        triples_per_user=20,
        seed=42,
    )
    dataloader = DataLoader(bundle.train_dataset, batch_size=128, shuffle=True)

    model = TwoTowerModel(
        user_feat_dim=bundle.user_features.shape[1],
        item_feat_dim=bundle.item_features.shape[1],
        emb_dim=16,
        hidden_dim=32,
    )
    trainer = Trainer(model=model, lr=1e-3)

    before = recall_at_k(
        model=model,
        user_features=bundle.user_features,
        item_features=bundle.item_features,
        positive_item_per_user=bundle.eval_positive_item_per_user,
        k=10,
    )

    trainer.train(
        dataloader=dataloader,
        user_features=bundle.user_features,
        item_features=bundle.item_features,
        num_epochs=5,
    )

    after = recall_at_k(
        model=model,
        user_features=bundle.user_features,
        item_features=bundle.item_features,
        positive_item_per_user=bundle.eval_positive_item_per_user,
        k=10,
    )

    assert after >= before
