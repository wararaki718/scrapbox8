SELECT
    customer_id,
    CAST(age AS INTEGER) AS age,
    CAST(income AS DOUBLE) AS income,
    CAST(transactions AS INTEGER) AS transactions,
    CAST(churn AS INTEGER) AS churn
FROM raw_customers
