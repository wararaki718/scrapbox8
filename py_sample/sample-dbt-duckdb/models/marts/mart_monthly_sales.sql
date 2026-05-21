select
  date_trunc('month', f.order_date) as month_start_date,
  c.region,
  c.segment,
  p.category,
  count(distinct f.order_id) as monthly_orders,
  sum(f.quantity) as monthly_units_sold,
  sum(f.line_amount) as monthly_gross_sales
from {{ ref('fct_orders') }} f
inner join {{ ref('stg_customers') }} c
  on f.customer_id = c.customer_id
inner join {{ ref('stg_products') }} p
  on f.product_id = p.product_id
group by 1,2,3,4
