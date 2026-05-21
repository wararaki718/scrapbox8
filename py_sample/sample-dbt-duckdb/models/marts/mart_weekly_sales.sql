select
  date_trunc('week', f.order_date) as week_start_date,
  c.region,
  c.segment,
  p.category,
  count(distinct f.order_id) as weekly_orders,
  sum(f.quantity) as weekly_units_sold,
  sum(f.line_amount) as weekly_gross_sales
from {{ ref('fct_orders') }} f
inner join {{ ref('stg_customers') }} c
  on f.customer_id = c.customer_id
inner join {{ ref('stg_products') }} p
  on f.product_id = p.product_id
group by 1,2,3,4
