import lightgbm as lgb

from dataset import CustomDataset
from utils import get_features, get_labels, get_pairs


def main() -> None:
    # Set parameters for synthetic data generation
    n_data = 10000
    n_users = 10
    n_items = 20
    n_user_dim = 30
    n_item_dim = 40

    # Generate synthetic data
    users = get_features(n_users, n_user_dim)
    items = get_features(n_items, n_item_dim)
    pairs = get_pairs(n_data, n_users, n_items)
    labels = get_labels(n_data)
    print(users.shape, items.shape, pairs.shape, labels.shape)

    # Create custom dataset
    dataset = CustomDataset(users, items, pairs)

    # Train a simple model
    model = lgb.LGBMClassifier(objective='binary', metric='binary_logloss')
    model.fit(dataset, labels)
    print("Model trained successfully with LGBMClassifier!")
    print("DONE")


if __name__ == "__main__":
    main()
