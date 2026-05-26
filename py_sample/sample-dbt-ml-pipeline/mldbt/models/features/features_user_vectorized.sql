SELECT
    user_id,
    -- scaling
    (age - 40.0) / 10.0 AS age_scaled,
    -- gender one-hot
    CASE
        WHEN gender = 'male'
        THEN 1
        ELSE 0
    END AS gender_male,
    CASE
        WHEN gender = 'female'
        THEN 1
        ELSE 0
    END AS gender_female,
    -- age bucket one-hot
    CASE
        WHEN age_bucket = 'senior'
        THEN 1
        ELSE 0
    END AS senior_flag,
    -- department one-hot
    CASE
        WHEN department = 'Engineering'
        THEN 1
        ELSE 0
    END AS dept_engineering,
    CASE
        WHEN department = 'Support'
        THEN 1
        ELSE 0
    END AS dept_support,
    high_age
FROM {{ ref('int_user_features') }}
