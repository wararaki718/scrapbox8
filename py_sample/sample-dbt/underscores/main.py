import duckdb
import subprocess


def main() -> None:
    result = subprocess.run(
        ["dbt", "run"],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    print(result.stderr)

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
