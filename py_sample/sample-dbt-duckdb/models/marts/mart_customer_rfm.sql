with customer_base as (
  select
    customer_id,
    customer_name,
    region,
    segment
  from {{ ref('stg_customers') }}
),
customer_metrics as (
  select
    customer_id,
    count(distinct order_id) as frequency,
    sum(line_amount) as monetary,
    max(ordered_at) as last_ordered_at
  from {{ ref('fct_orders') }}
  group by 1
),
scored as (
  select
    b.customer_id,
    b.customer_name,
    b.region,
    b.segment,
    coalesce(m.frequency, 0) as frequency,
    coalesce(m.monetary, 0) as monetary,
    m.last_ordered_at,
    case when m.last_ordered_at is null then 999999
         else date_diff('day', cast(m.last_ordered_at as date), current_date)
    end as recency_days
  from customer_base b
  left join customer_metrics m
    on b.customer_id = m.customer_id
)
select
  customer_id,
  customer_name,
  region,
  segment,
  recency_days,
  frequency,
  monetary,
  ntile(5) over (order by recency_days desc) as r_score,
  ntile(5) over (order by frequency asc) as f_score,
  ntile(5) over (order by monetary asc) as m_score,
  cast(ntile(5) over (order by recency_days desc) as varchar)
    || cast(ntile(5) over (order by frequency asc) as varchar)
    || cast(ntile(5) over (order by monetary asc) as varchar) as rfm_segment
from scored
