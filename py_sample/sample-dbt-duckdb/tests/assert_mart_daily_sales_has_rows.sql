-- returns a row when the test should fail
select 1 as should_fail
where (select count(*) from {{ ref('mart_daily_sales') }}) = 0
