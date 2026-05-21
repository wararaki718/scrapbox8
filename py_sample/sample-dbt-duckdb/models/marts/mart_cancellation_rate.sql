with order_level as (
  select distinct
    order_id,
    order_date,
    order_status,
    customer_id
  from {{ ref('fct_orders') }}
)
select
  order_date,
  c.region,
  c.segment,
  count(*) as total_orders,
  sum(case when order_status = 'cancelled' then 1 else 0 end) as cancelled_orders,
  case
    when count(*) = 0 then 0
    else cast(sum(case when order_status = 'cancelled' then 1 else 0 end) as double) / count(*)
  end as cancellation_rate
from order_level o
inner join {{ ref('stg_customers') }} c
  on o.customer_id = c.customer_id
group by 1,2,3
