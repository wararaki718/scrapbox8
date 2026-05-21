select
  cast(order_id as integer) as order_id,
  cast(customer_id as integer) as customer_id,
  cast(order_status as varchar) as order_status,
  cast(ordered_at as timestamp) as ordered_at
from {{ source('raw', 'raw_orders') }}
