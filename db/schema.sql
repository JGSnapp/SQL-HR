CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE candidates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    sex TEXT,
    age INTEGER,
    expected_salary_rub INTEGER,
    desired_position TEXT,
    city TEXT,
    ready_to_relocate BOOLEAN,
    ready_for_business_trips BOOLEAN,
    employment_type TEXT,
    work_schedule TEXT,
    work_experience TEXT,
    last_company TEXT,
    last_job_title TEXT,
    education_level_and_university TEXT,
    resume_updated_at TIMESTAMP,
    has_car BOOLEAN
);
