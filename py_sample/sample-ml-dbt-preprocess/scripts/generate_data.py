import pandas as pd
import numpy as np


def main() -> None:
    np.random.seed(42)

    # generate
    n = 1000
    df = pd.DataFrame({
        "customer_id": range(n),
        "age": np.random.randint(18, 70, n),
        "income": np.random.randint(30000, 150000, n),
        "transactions": np.random.randint(1, 50, n),
    })

    # 疑似 churn label
    df["churn"] = (
        (df["transactions"] < 10) & (df["income"] < 60000)
    ).astype(int)

    # save
    df.to_csv("data/raw/customers.csv", index=False)
    print(df.head())
    print("DONE")


if __name__ == "__main__":
    main()
