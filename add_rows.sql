COPY candidates (
    sex,
    expected_salary_rub,
    desired_position,
    city,
    ready_to_relocate,
    ready_for_business_trips,
    employment_type,
    work_schedule,
    work_experience,
    last_company,
    last_job_title,
    education_level_and_university,
    resume_updated_at,
    has_car
)
FROM '/path/to/candidates.csv'
WITH (
    FORMAT csv,
    HEADER true,
    DELIMITER ',',
    ENCODING 'UTF8'
);