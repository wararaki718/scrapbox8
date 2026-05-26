import duckdb
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def main() -> None:
    with duckdb.connect("warehouse/ml.duckdb") as con:
        df = con.execute("""
            SELECT *
            FROM fct_training_dataset
        """).fetchdf()
        X = df.drop(columns=["user_id", "high_age"])
        y = df["high_age"]

        # train test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

        # model train
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        )
        model.fit(X_train, y_train)

        # prediction
        pred = model.predict(X_test)
        print(classification_report(y_test, pred))

        # save model
        joblib.dump(model, "models/churn_model.pkl")
        print("model saved")

        print("DONE")


if __name__ == "__main__":
    main()
