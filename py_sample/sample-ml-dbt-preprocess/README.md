# dbt + duckdb

## setup

```shell
pip install dbt-duckdb pandas duckdb scikit-learn
```

## run

```shell
python scripts/generate_data.py
```

```shell
python scripts/preprocess.py
```

```shell
python scripts/load_duckdb.py
```

```shell
cd mldbt
dbt run
```

```shell
python scripts/train.py
```
