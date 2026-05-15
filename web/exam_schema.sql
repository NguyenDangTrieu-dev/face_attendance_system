CREATE TABLE IF NOT EXISTS exam_sessions (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    course_id INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    lecturer_id VARCHAR(50) REFERENCES users(id),
    exam_date DATE NOT NULL,
    start_time TIME NOT NULL,
    end_time TIME NOT NULL,
    room VARCHAR(100),
    note TEXT,
    status VARCHAR(30) DEFAULT 'upcoming',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS exam_attendance (
    id SERIAL PRIMARY KEY,
    exam_session_id INTEGER NOT NULL REFERENCES exam_sessions(id) ON DELETE CASCADE,
    student_id VARCHAR(50) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    image_base64 TEXT,
    recognized BOOLEAN DEFAULT TRUE,
    similarity FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_exam_attendance_student UNIQUE (exam_session_id, student_id)
);

CREATE INDEX IF NOT EXISTS idx_exam_sessions_course_id ON exam_sessions(course_id);
CREATE INDEX IF NOT EXISTS idx_exam_sessions_lecturer_id ON exam_sessions(lecturer_id);
CREATE INDEX IF NOT EXISTS idx_exam_attendance_exam_session_id ON exam_attendance(exam_session_id);
CREATE INDEX IF NOT EXISTS idx_exam_attendance_student_id ON exam_attendance(student_id);
