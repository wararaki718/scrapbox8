with cohort_size as (
  select
    cohort_month,
    max(case when months_since_cohort = 0 then active_customers end) as cohort_customers
  from {{ ref('mart_retention') }}
  group by 1
)
select
  r.cohort_month,
  r.activity_month,
  r.months_since_cohort,
  r.active_customers,
  c.cohort_customers,
  case
    when c.cohort_customers is null or c.cohort_customers = 0 then null
    else cast(r.active_customers as double) / c.cohort_customers
  end as retention_rate
from {{ ref('mart_retention') }} r
inner join cohort_size c
  on r.cohort_month = c.cohort_month
