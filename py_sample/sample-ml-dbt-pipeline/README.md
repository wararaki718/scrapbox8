# ml pipeline with dbt

## setup

```shell
pip install dbt-duckdb duckdb scikit-learn pandas
```

## run

```shell
python scripts/generate_data.py
```

```shell
python scripts/load_duckdb.py
```

```shell
cd mldbt
dbt run
```

```shell
python scripts/train_model.py
```
