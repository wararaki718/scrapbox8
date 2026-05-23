import duckdb
import pandas as pd


def main() -> None:
    df = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "country": ["JP", "US", "JP"],
            "score": [100, 80, 120],
        }
    )
    print(df.shape)

    with duckdb.connect("tmp.db") as con:
        con.register("user_df", df)
        con.execute(
            """
            CREATE OR REPLACE TABLE users AS
            SELECT * FROM user_df
            """
        )
    
    with duckdb.connect("tmp.db") as con:
        result = con.execute(
            """
            SELECT country, AVG(score) AS avg_score
            FROM users
            GROUP BY country
            """
        ).fetchdf()
        print(result)

    print("DONE")


if __name__ == "__main__":
    main()
