import duckdb
import pandas as pd


def main() -> None:
    duckdb.sql("select 42").show()

    tmp = duckdb.sql("select 42 as i")
    duckdb.sql("select i * 2 as k from tmp").show()

    # load sample.csv
    duckdb.read_csv("sample.csv").show()

    duckdb.sql("select * from 'sample.csv'").show()

    # load dataframe
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    duckdb.from_df(df).show()

    duckdb.sql("select * from df").show()

    # persistent storage
    with duckdb.connect("tmp.db") as con:
        con.execute(
            f"""CREATE OR REPLACE TABLE items AS
            SELECT *
            FROM read_csv_auto('sample.csv', HEADER = TRUE)
            """
        )

        row_count = con.execute("SELECT COUNT(*) AS row_count FROM items"
        ).fetchone()[0]
        print(f"registered rows: {row_count} -> table: items in tmp.db")

    # use the persistent storage
    with duckdb.connect("tmp.db") as con:
        output = con.execute("SELECT * FROM items").fetchall()
        print(pd.DataFrame(output))

    print("DONE")


if __name__ == "__main__":
    main()
