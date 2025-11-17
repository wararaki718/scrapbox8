import re

import numpy as np
import pandas as pd


class UserPreprocessor:
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # normalize
        df["age_norm"] = (df.age - df.age.min()) / (df.age.max()-df.age.min()+1e-6)

        # gender
        gender_map = {"M":0.0, "F":1.0}
        df["gender_id"] = df.gender.map(gender_map).astype(np.float32)

        # occupation
        occupation_list = sorted(df.occupation.unique())
        occupation2index = {occupation: i for i, occupation in enumerate(occupation_list)}
        df["occupation_id"] = df.occupation.map(occupation2index)
        
        return df


class ItemPreprocessor:
    def _extract_year(self, title: str) -> float:
        m = re.search(r"(\d{4})", str(title))
        return float(m.group(1)) if m else 0.0

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # extract year from title
        df["year"] = df.title.apply(self._extract_year)
        df["year_norm"] = (df.year - df.year.min())/(df.year.max()-df.year.min()+1e-6)

        return df
