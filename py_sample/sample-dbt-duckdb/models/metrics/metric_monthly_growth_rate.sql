with monthly as (
  select
    month_start_date,
    sum(monthly_gross_sales) as monthly_revenue
  from {{ ref('mart_monthly_sales') }}
  group by 1
),
with_prev as (
  select
    month_start_date,
    monthly_revenue,
    lag(monthly_revenue) over (order by month_start_date) as prev_month_revenue
  from monthly
)
select
  month_start_date,
  monthly_revenue,
  prev_month_revenue,
  case
    when prev_month_revenue is null or prev_month_revenue = 0 then null
    else (monthly_revenue - prev_month_revenue) / prev_month_revenue
  end as growth_rate
from with_prev
