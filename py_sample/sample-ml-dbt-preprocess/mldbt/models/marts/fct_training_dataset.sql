SELECT
    customer_id,
    age,
    income,
    transactions,
    income_per_age,
    high_activity,
    churn
FROM {{ ref('int_customer_features') }}
