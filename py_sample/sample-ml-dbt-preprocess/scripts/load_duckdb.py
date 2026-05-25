import duckdb


def main() -> None:
    with duckdb.connect("warehouse/ml.duckdb") as con:
        con.execute("""
            CREATE OR REPLACE TABLE raw_customers AS
            SELECT *
            FROM read_parquet(
                'data/processed/customers.parquet'
            )
        """)
        result = con.execute("SELECT COUNT(*) FROM raw_customers").fetchall()
        print(result)
    print("DONE")


if __name__ == "__main__":
    main()
