import pandas as pd


def main() -> None:
    df = pd.read_csv("data/raw/customers.csv")

    # 例: null 補完
    df["income"] = df["income"].fillna(
        df["income"].median()
    )

    # parquet 化
    df.to_parquet("data/processed/customers.parquet", index=False)
    print(df.head())
    print("DONE")


if __name__ == "__main__":
    main()
