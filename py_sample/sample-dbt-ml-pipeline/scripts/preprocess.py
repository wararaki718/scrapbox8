import json
import pandas as pd


def main() -> None:
    with open("data/raw/users.json") as f:
        raw = json.load(f)

    df = pd.json_normalize(raw["users"])
    df = df[
        [
            "id",
            "age",
            "gender",
            "company.department",
            "address.state",
            "bank.cardType",
        ]
    ]

    df.columns = [
        "user_id",
        "age",
        "gender",
        "department",
        "state",
        "card_type",
    ]

    # 疑似 target 作成
    df["high_age"] = (
        df["age"] >= 50
    ).astype(int)

    df.to_parquet(
        "data/processed/users.parquet",
        index=False,
    )

    print(df.head())
    print("DONE")


if __name__ == "__main__":
    main()
