with customer_orders as (
  select
    customer_id,
    min(ordered_at) as first_ordered_at,
    max(ordered_at) as last_ordered_at,
    count(distinct order_id) as total_orders,
    sum(line_amount) as total_spend
  from {{ ref('fct_orders') }}
  group by 1
)
select
  c.customer_id,
  c.customer_name,
  c.email,
  c.region,
  c.segment,
  c.created_at,
  coalesce(o.first_ordered_at, null) as first_ordered_at,
  coalesce(o.last_ordered_at, null) as last_ordered_at,
  coalesce(o.total_orders, 0) as total_orders,
  coalesce(o.total_spend, 0) as total_spend
from {{ ref('stg_customers') }} c
left join customer_orders o
  on c.customer_id = o.customer_id
