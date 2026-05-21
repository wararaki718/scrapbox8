select
  cast(customer_id as integer) as customer_id,
  cast(customer_name as varchar) as customer_name,
  cast(email as varchar) as email,
  cast(region as varchar) as region,
  cast(segment as varchar) as segment,
  cast(created_at as timestamp) as created_at
from {{ source('raw', 'raw_customers') }}
