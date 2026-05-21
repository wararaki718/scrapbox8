with monthly_order_revenue as (
  select
    date_trunc('month', order_date) as month_start_date,
    sum(line_amount) as monthly_revenue,
    count(distinct customer_id) as paying_customers
  from {{ ref('fct_orders') }}
  where order_status != 'cancelled'
  group by 1
)
select
  month_start_date,
  monthly_revenue,
  paying_customers,
  case
    when paying_customers = 0 then null
    else monthly_revenue / paying_customers
  end as arpu
from monthly_order_revenue
