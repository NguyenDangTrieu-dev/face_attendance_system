CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE roles (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL
);

CREATE TABLE users (
    id TEXT PRIMARY KEY,
    full_name TEXT NOT NULL,
    password TEXT NOT NULL,
    role_id INTEGER REFERENCES roles(id),
    image TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


CREATE TABLE embeddings (
    user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    embedding bytea,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE courses (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    lecturer_id TEXT REFERENCES users(id) ON DELETE SET NULL,
    semester TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE course_students (
    course_id INTEGER REFERENCES courses(id) ON DELETE CASCADE,
    student_id TEXT REFERENCES users(id) ON DELETE CASCADE,
    PRIMARY KEY (course_id, student_id)
);

CREATE TABLE attendance (
    id SERIAL PRIMARY KEY,
    course_id INTEGER REFERENCES courses(id) ON DELETE CASCADE,
    student_id TEXT REFERENCES users(id) ON DELETE CASCADE,
    time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    recognized BOOLEAN DEFAULT TRUE,
    image_base64 TEXT
);
