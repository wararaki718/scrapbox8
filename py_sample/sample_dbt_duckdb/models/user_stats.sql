SELECT
    country,
    COUNT(*) AS user_count,
    AVG(score) AS avg_score
FROM {{ ref('users') }}
GROUP BY country
