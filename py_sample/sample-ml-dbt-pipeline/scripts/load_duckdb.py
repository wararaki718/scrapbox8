import duckdb


def main() -> None:
    with duckdb.connect("warehouse/ml.duckdb") as con:
        con.execute("""
            CREATE OR REPLACE TABLE customers AS
            SELECT *
            FROM read_csv_auto('data/raw/customers.csv')
        """)

        print(con.execute("SELECT COUNT(*) FROM customers").fetchall())
    print("DONE")


if __name__ == "__main__":
    main()
