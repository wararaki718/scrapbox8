with customer_spend as (
  select
    customer_id,
    sum(line_amount) as total_spend,
    count(distinct order_id) as total_orders,
    min(ordered_at) as first_ordered_at,
    max(ordered_at) as last_ordered_at
  from {{ ref('fct_orders') }}
  group by 1
)
select
  c.customer_id,
  c.customer_name,
  c.region,
  c.segment,
  coalesce(cs.total_spend, 0) as total_spend,
  coalesce(cs.total_orders, 0) as total_orders,
  cs.first_ordered_at,
  cs.last_ordered_at,
  dense_rank() over (order by coalesce(cs.total_spend, 0) desc) as ltv_rank
from {{ ref('stg_customers') }} c
left join customer_spend cs
  on c.customer_id = cs.customer_id
