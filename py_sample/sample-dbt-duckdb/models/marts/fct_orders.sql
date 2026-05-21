select
  oi.order_item_id,
  o.order_id,
  o.customer_id,
  oi.product_id,
  o.order_status,
  o.ordered_at,
  cast(o.ordered_at as date) as order_date,
  oi.quantity,
  p.unit_price,
  oi.quantity * p.unit_price as line_amount
from {{ ref('stg_order_items') }} oi
inner join {{ ref('stg_orders') }} o
  on oi.order_id = o.order_id
inner join {{ ref('stg_products') }} p
  on oi.product_id = p.product_id
