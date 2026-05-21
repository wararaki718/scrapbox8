select
  cast(product_id as integer) as product_id,
  cast(product_name as varchar) as product_name,
  cast(category as varchar) as category,
  cast(unit_price as double) as unit_price,
  cast(is_active as boolean) as is_active
from {{ source('raw', 'raw_products') }}
