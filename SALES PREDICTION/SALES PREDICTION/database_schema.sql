-- PostgreSQL schema for Sales Forecasting project
-- Database name: sales_forecasting_db

CREATE TABLE IF NOT EXISTS "user" (
    id SERIAL PRIMARY KEY,
    username VARCHAR(150) UNIQUE NOT NULL,
    password VARCHAR(150) NOT NULL
);

CREATE TABLE IF NOT EXISTS prediction (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES "user"(id),
    rate DOUBLE PRECISION NOT NULL,
    sales_first DOUBLE PRECISION NOT NULL,
    sales_second DOUBLE PRECISION NOT NULL,
    predicted_sales DOUBLE PRECISION NOT NULL,
    actual_sales DOUBLE PRECISION,
    timestamp TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sales_record (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    product VARCHAR(120) NOT NULL,
    region VARCHAR(120) NOT NULL,
    season VARCHAR(40) NOT NULL,
    festival_name VARCHAR(80) NOT NULL DEFAULT 'None',
    is_festival_day BOOLEAN NOT NULL DEFAULT FALSE,
    discount_percentage DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    sales_amount DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sales_record_date ON sales_record(date);
CREATE INDEX IF NOT EXISTS idx_sales_record_product ON sales_record(product);
CREATE INDEX IF NOT EXISTS idx_sales_record_region ON sales_record(region);
