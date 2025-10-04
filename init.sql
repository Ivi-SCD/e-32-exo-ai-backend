-- Initialization script for PostgreSQL database for NASA Exoplanets
-- This script is executed automatically when the PostgreSQL container is created

-- Create necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Configure timezone
SET timezone = 'UTC';

-- Comment about the database
COMMENT ON DATABASE nasa_exoplanets IS 'Database for NASA Exoplanets detection and analysis system';

-- The tables will be created automatically by the Python application using SQLAlchemy
-- This file serves only for initial database configurations
