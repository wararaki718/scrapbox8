# dbt duckdb pipeline

dbt + DuckDB でローカルのサンプルデータパイプラインを動かす最小構成です。

## 1) サンプルデータ生成

以下を実行すると `data/raw/*.csv` が生成されます。

python3 scripts/generate_sample_data.py

## 2) dbt 実行

このリポジトリでは dbt プロファイルをプロジェクト直下の `profiles.yml` に置いています。

mkdir -p warehouse
~/.local/bin/dbt debug --profiles-dir .
~/.local/bin/dbt run --profiles-dir .

生成先 DuckDB ファイル:
- `warehouse/sample.duckdb`

## 3) モデル構成

- raw 層（table）
  - `models/raw/raw_customers.sql`
  - `models/raw/raw_products.sql`
  - `models/raw/raw_orders.sql`
  - `models/raw/raw_order_items.sql`

- staging 層（view）
  - `models/staging/stg_customers.sql`
  - `models/staging/stg_products.sql`
  - `models/staging/stg_orders.sql`
  - `models/staging/stg_order_items.sql`

- marts 層（table）
  - `models/marts/fct_orders.sql`
  - `models/marts/dim_customers.sql`
  - `models/marts/mart_daily_sales.sql`
  - `models/marts/mart_weekly_sales.sql`
  - `models/marts/customer_ltv_rank.sql`
  - `models/marts/mart_cancellation_rate.sql`
  - `models/marts/mart_monthly_sales.sql`
  - `models/marts/mart_customer_rfm.sql`
  - `models/marts/mart_retention.sql`

source 定義:
- `models/staging/sources.yml`（`main_raw` スキーマを参照）

- metrics 層（table）
  - `models/metrics/metric_monthly_growth_rate.sql`
  - `models/metrics/metric_retention_rate.sql`
  - `models/metrics/metric_arpu.sql`

テスト定義:
- `models/marts/schema.yml`
- `tests/assert_fct_orders_has_rows.sql`
- `tests/assert_mart_daily_sales_has_rows.sql`
- `tests/assert_mart_weekly_sales_has_rows.sql`
- `tests/assert_customer_ltv_rank_has_rows.sql`
- `tests/assert_mart_cancellation_rate_has_rows.sql`
- `tests/assert_mart_monthly_sales_has_rows.sql`
- `tests/assert_mart_customer_rfm_has_rows.sql`
- `tests/assert_mart_retention_has_rows.sql`
- `tests/assert_metric_monthly_growth_rate_has_rows.sql`
- `tests/assert_metric_retention_rate_has_rows.sql`
- `tests/assert_metric_arpu_has_rows.sql`

## 4) 実行コマンド

全体実行:
~/.local/bin/dbt run --profiles-dir .

marts のみ実行:
~/.local/bin/dbt run --profiles-dir . --select marts

テスト実行:
~/.local/bin/dbt test --profiles-dir . --select marts

## 5) 件数確認例

python3 - <<'PY'
import duckdb
con=duckdb.connect('warehouse/sample.duckdb')
for q in [
    "select count(*) from main_staging.stg_customers",
    "select count(*) from main_staging.stg_products",
    "select count(*) from main_staging.stg_orders",
    "select count(*) from main_staging.stg_order_items",
    "select count(*) from main_marts.fct_orders",
    "select count(*) from main_marts.dim_customers",
    "select count(*) from main_marts.mart_daily_sales",
    "select count(*) from main_marts.mart_weekly_sales",
    "select count(*) from main_marts.customer_ltv_rank",
    "select count(*) from main_marts.mart_cancellation_rate",
    "select count(*) from main_marts.mart_monthly_sales",
    "select count(*) from main_marts.mart_customer_rfm",
    "select count(*) from main_marts.mart_retention",
    "select count(*) from main_metrics.metric_monthly_growth_rate",
    "select count(*) from main_metrics.metric_retention_rate",
    "select count(*) from main_metrics.metric_arpu",
]:
    print(q, con.sql(q).fetchone()[0])
PY
