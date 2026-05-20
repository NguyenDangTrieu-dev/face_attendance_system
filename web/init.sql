
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- ROLES
-- ============================================================
CREATE TABLE IF NOT EXISTS roles (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL
);

-- ============================================================
-- USERS
-- role_id:
--   1: Admin
--   2: Giảng viên
--   3: Sinh viên
--
-- admission_course:
--   Khóa trúng tuyển của sinh viên, dạng 4 năm, ví dụ: 2022-2026
-- major:
--   Ngành học của sinh viên
-- ============================================================
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    full_name TEXT NOT NULL,
    password TEXT NOT NULL,
    role_id INTEGER REFERENCES roles(id),
    image TEXT,
    admission_course VARCHAR(20),
    major VARCHAR(120),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_users_role_admission_course_major
ON users(role_id, admission_course, major);

-- Nếu hệ thống cũ từng dùng cột cohort, có thể copy dữ liệu sang admission_course:
-- UPDATE users
-- SET admission_course = cohort
-- WHERE admission_course IS NULL AND cohort IS NOT NULL;

-- ============================================================
-- FACE EMBEDDINGS
-- ============================================================
CREATE TABLE IF NOT EXISTS embeddings (
    user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    embedding BYTEA,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- COURSES
-- semester lưu theo dạng: HK1-2025-2026, HK2-2025-2026, HK Hè-2025-2026
-- Trong đó 2025-2026 là khóa học phần / năm học 2 năm.
-- ============================================================
CREATE TABLE IF NOT EXISTS courses (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    lecturer_id TEXT REFERENCES users(id) ON DELETE SET NULL,
    semester TEXT,
    room TEXT,
    period TEXT,
    credits INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_courses_lecturer_id ON courses(lecturer_id);
CREATE INDEX IF NOT EXISTS idx_courses_semester ON courses(semester);

-- ============================================================
-- COURSE STUDENTS
-- ============================================================
CREATE TABLE IF NOT EXISTS course_students (
    course_id INTEGER REFERENCES courses(id) ON DELETE CASCADE,
    student_id TEXT REFERENCES users(id) ON DELETE CASCADE,
    PRIMARY KEY (course_id, student_id)
);

CREATE INDEX IF NOT EXISTS idx_course_students_student_id
ON course_students(student_id);

-- ============================================================
-- COURSE ATTENDANCE
-- ============================================================
CREATE TABLE IF NOT EXISTS attendance (
    id SERIAL PRIMARY KEY,
    course_id INTEGER REFERENCES courses(id) ON DELETE CASCADE,
    student_id TEXT REFERENCES users(id) ON DELETE CASCADE,
    time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    recognized BOOLEAN DEFAULT TRUE,
    image_base64 TEXT
);

CREATE INDEX IF NOT EXISTS idx_attendance_course_id ON attendance(course_id);
CREATE INDEX IF NOT EXISTS idx_attendance_student_id ON attendance(student_id);
CREATE INDEX IF NOT EXISTS idx_attendance_time ON attendance(time);

-- ============================================================
-- EXAM SESSIONS
-- status được API tự tính theo exam_date, start_time, end_time:
--   upcoming : chưa tới giờ bắt đầu
--   ongoing  : đang trong thời gian thi
--   finished : đã qua giờ kết thúc
-- ============================================================
CREATE TABLE IF NOT EXISTS exam_sessions (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    course_id INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    lecturer_id TEXT REFERENCES users(id) ON DELETE SET NULL,
    exam_date DATE NOT NULL,
    start_time TIME NOT NULL,
    end_time TIME NOT NULL,
    room VARCHAR(100),
    note TEXT,
    status VARCHAR(30) DEFAULT 'upcoming',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_exam_sessions_time_range CHECK (end_time > start_time),
    CONSTRAINT chk_exam_sessions_status CHECK (
        status IN ('upcoming', 'ongoing', 'finished')
    )
);

CREATE INDEX IF NOT EXISTS idx_exam_sessions_course_id
ON exam_sessions(course_id);

CREATE INDEX IF NOT EXISTS idx_exam_sessions_lecturer_id
ON exam_sessions(lecturer_id);

CREATE INDEX IF NOT EXISTS idx_exam_sessions_date_time
ON exam_sessions(exam_date, start_time, end_time);

CREATE INDEX IF NOT EXISTS idx_exam_sessions_status
ON exam_sessions(status);

-- ============================================================
-- EXAM ATTENDANCE
-- Mỗi sinh viên chỉ có 1 bản ghi điểm danh cho 1 ca thi.
-- Nếu điểm danh lại, API sẽ cập nhật bản ghi cũ.
-- ============================================================
CREATE TABLE IF NOT EXISTS exam_attendance (
    id SERIAL PRIMARY KEY,
    exam_session_id INTEGER NOT NULL REFERENCES exam_sessions(id) ON DELETE CASCADE,
    student_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    image_base64 TEXT,
    recognized BOOLEAN DEFAULT TRUE,
    similarity FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_exam_attendance_student UNIQUE (exam_session_id, student_id)
);

CREATE INDEX IF NOT EXISTS idx_exam_attendance_exam_session_id
ON exam_attendance(exam_session_id);

CREATE INDEX IF NOT EXISTS idx_exam_attendance_student_id
ON exam_attendance(student_id);

CREATE INDEX IF NOT EXISTS idx_exam_attendance_time
ON exam_attendance(time);

-- ============================================================
-- OPTIONAL SEED DATA
-- Bỏ comment nếu muốn init DB có sẵn 3 role cơ bản.
-- ============================================================
-- INSERT INTO roles (id, name) VALUES
--     (1, 'admin'),
--     (2, 'lecturer'),
--     (3, 'student')
-- ON CONFLICT (id) DO NOTHING;
