import duckdb


def main() -> None:
    with duckdb.connect("dev.duckdb") as con:
        df = con.execute("SELECT * FROM main.user_stats").fetchdf()
        print(df)
    print("DONE")


if __name__ == "__main__":
    main()
