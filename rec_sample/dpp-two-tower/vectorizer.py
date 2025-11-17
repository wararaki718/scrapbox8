import numpy as np
import pandas as pd


class UserVectorizer:
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        n_occupations = df.occupation.nunique()
        user_occupations = np.eye(n_occupations)[df.occupation_id]
        user_features = np.concatenate([
            df.age_norm.values.reshape(-1,1),
            df.gender_id.values.reshape(-1,1),
            user_occupations
        ], axis=1, dtype=np.float32)

        return user_features


class ItemVectorizer:
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        genre_columns = [f"g{i}" for i in range(19)]
        genre_vecs = df[genre_columns].astype(np.float32).values
        item_features = np.concatenate([genre_vecs, df.year_norm.values.reshape(-1,1)], axis=1).astype(np.float32)
        return item_features
