# dbt

## setup

```shell
pip install dbt-duckdb pandas duckdb scikit-learn
```

## run

```shell
python scripts/ingest.py
```

```shell
python scripts/preprocess.py
```

```shell
python scripts/load_duckdb.py
```

```shell
cd mldbt
dbt run --profiles-dir .
```

```shell
python scripts/train.py
```
