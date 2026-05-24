import duckdb
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import joblib


def main() -> None:
    # DuckDB 接続
    with duckdb.connect("warehouse/ml.duckdb") as con:
        # feature table 読み込み
        df = con.execute("""
        SELECT *
        FROM customer_features
        """).fetchdf()

    # feature / target
    X = df.drop(columns=["customer_id", "churn"])
    y = df["churn"]

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # predict
    pred = model.predict(X_test)
    print(classification_report(y_test, pred))

    # save
    joblib.dump(model, "models/model.pkl")
    print("model saved")

    print("DONE")


if __name__ == "__main__":
    main()
