SELECT
    user_id,
    age,
    gender,
    department,
    state,
    card_type,
    CASE
        WHEN age >= 40
        THEN 'senior'
        ELSE 'junior'
    END AS age_bucket,
    high_age
FROM {{ ref('stg_users') }}
