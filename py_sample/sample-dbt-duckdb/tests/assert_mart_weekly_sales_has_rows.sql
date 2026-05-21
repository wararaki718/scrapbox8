-- RED test: model 未作成時は失敗する
select 1 as should_fail
where (select count(*) from main_marts.mart_weekly_sales) = 0
