from pathlib import Path

import pandas as pd
import numpy as np


def main() -> None:
    np.random.seed(42)

    n = 1000
    df = pd.DataFrame({
        "customer_id": range(n),
        "age": np.random.randint(18, 70, n),
        "income": np.random.randint(30000, 150000, n),
        "transactions": np.random.randint(1, 50, n),
    })

    # 疑似 target
    df["churn"] = (
        (df["transactions"] < 10)
        & (df["income"] < 60000)
    ).astype(int)
    print(df.head())

    # save csv
    output_path = Path("data/raw/customers.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

    print("DONE")


if __name__ == "__main__":
    main()
