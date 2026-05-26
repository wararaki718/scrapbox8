SELECT
    CAST(user_id AS INTEGER) AS user_id,
    CAST(age AS INTEGER) AS age,
    gender,
    department,
    state,
    card_type,
    CAST(high_age AS INTEGER) AS high_age
FROM raw_users