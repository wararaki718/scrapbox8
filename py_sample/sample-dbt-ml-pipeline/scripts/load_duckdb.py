import duckdb


def main() -> None:
    with duckdb.connect("warehouse/ml.duckdb") as con:
        con.execute("""
            CREATE OR REPLACE TABLE raw_users AS
            SELECT *
            FROM read_parquet(
                'data/processed/users.parquet'
            )
        """)

        result = con.execute("SELECT COUNT(*) FROM raw_users").fetchall()
        print(result)
    print("DONE")


if __name__ == "__main__":
    main()
