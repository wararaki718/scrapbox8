import torch
from torch.utils.data import DataLoader

from download import download_movielens
from load import load_items, load_users, load_ratings
from preprocess import ItemPreprocessor, UserPreprocessor
from vectorizer import ItemVectorizer, UserVectorizer
from dataset import ML100kDataset
from model import TwoTowerModel
from trainer import Trainer
from rerank import mmr_rerank


def main() -> None:
    # download
    download_movielens()

    df_users = load_users()
    df_items = load_items()
    df_ratings = load_ratings()
    print("users:", df_users.shape)
    print("items:", df_items.shape)
    print("ratings:", df_ratings.shape)
    print()

    user_preprocessor = UserPreprocessor()
    item_preprocessor = ItemPreprocessor()
    df_users = user_preprocessor.transform(df_users)
    df_items = item_preprocessor.transform(df_items)
    print("After preprocessing")
    print("users:", df_users.shape)
    print("items:", df_items.shape)
    print()

    user_vectorizer = UserVectorizer()
    item_vectorizer = ItemVectorizer()
    user_features = user_vectorizer.transform(df_users)
    item_features = item_vectorizer.transform(df_items)
    print("After vectorization")
    print("user features:", user_features.shape)
    print("item features:", item_features.shape)
    print()

    dataset = ML100kDataset(
        df_ratings=df_ratings,
        user_features=user_features,
        item_features=item_features,
        num_neg=4,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    print("Dataset size:", len(dataset))

    # model definition
    model = TwoTowerModel(
        user_feat_dim=user_features.shape[1],
        item_feat_dim=item_features.shape[1],
        emb_dim=64,
        hidden_dim=128,
    )

    # train
    print("Start training...")
    trainer = Trainer(model, lambda_div=0.01, lr=1e-3)
    trainer.train(dataloader, num_epochs=5)
    print("Training completed.")
    print()

    # rerank example
    user_index = 0
    user_input = torch.tensor(user_features[user_index], dtype=torch.float32)
    candidate_item_feats = torch.tensor(item_features, dtype=torch.float32)
    candidate_scores = model(user_input.unsqueeze(0).repeat(candidate_item_feats.size(0),1), candidate_item_feats).detach().numpy()

    top_k = 10
    top_indices = mmr_rerank(candidate_item_feats.numpy(), candidate_scores, top_k=top_k)
    print(f"Top-{top_k} recommended item indices for user {user_index}: {top_indices}")

    print("DONE")


if __name__ == "__main__":
    main()
