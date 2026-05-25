SELECT
    customer_id,
    age,
    income,
    transactions,
    income / NULLIF(age, 0) AS income_per_age,
    CASE
        WHEN transactions >= 20
        THEN 1
        ELSE 0
    END AS high_activity,
    churn
FROM {{ ref('stg_customers') }}
