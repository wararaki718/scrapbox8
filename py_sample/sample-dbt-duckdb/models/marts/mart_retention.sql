with order_level as (
  select distinct
    order_id,
    customer_id,
    order_date
  from {{ ref('fct_orders') }}
),
cohorts as (
  select
    customer_id,
    date_trunc('month', min(order_date)) as cohort_month
  from order_level
  group by 1
),
activity as (
  select
    o.customer_id,
    c.cohort_month,
    date_trunc('month', o.order_date) as activity_month
  from order_level o
  inner join cohorts c
    on o.customer_id = c.customer_id
)
select
  cohort_month,
  activity_month,
  date_diff('month', cohort_month, activity_month) as months_since_cohort,
  count(distinct customer_id) as active_customers
from activity
group by 1,2,3
