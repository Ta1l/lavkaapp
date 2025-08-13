-- db/init.sql

DROP TABLE IF EXISTS shifts;
DROP TABLE IF EXISTS users;
DROP TYPE IF EXISTS shift_status;

CREATE TYPE shift_status AS ENUM ('pending', 'confirmed', 'cancelled', 'available');

CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL,
    full_name TEXT
);

CREATE TABLE shifts (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id) ON DELETE SET NULL, 
    shift_date DATE NOT NULL,
    day_of_week SMALLINT NOT NULL CHECK (day_of_week BETWEEN 1 AND 7),
    shift_code TEXT NOT NULL,
    status shift_status NOT NULL DEFAULT 'available',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    CONSTRAINT shifts_date_code_unique UNIQUE (shift_date, shift_code)
);

CREATE INDEX idx_shifts_shift_date ON shifts(shift_date);
CREATE INDEX idx_shifts_user_date ON shifts(user_id, shift_date);

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_shifts_updated_at
    BEFORE UPDATE ON shifts
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();