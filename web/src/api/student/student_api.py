import base64

import cv2
import numpy as np
from flask import Blueprint, jsonify, request
from psycopg2.extras import RealDictCursor

from auth import role_required
from db import get_db_connection

# Functions used by the original monolithic main.py
from register import register_face
from face_rec_SM_api import invalidate_embedding_cache
from src.api.student_import import import_from_zip


student_bp = Blueprint("student", __name__)


def get_current_user():
    return getattr(request, "user", {}) or {}


def get_current_student_id():
    return get_current_user().get("user_id")


def table_exists(cur, table_name):
    cur.execute("""
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name = %s
        ) AS exists
    """, (table_name,))
    return bool(cur.fetchone()["exists"])


def column_exists(cur, table_name, column_name):
    cur.execute("""
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = %s
              AND column_name = %s
        ) AS exists
    """, (table_name, column_name))
    return bool(cur.fetchone()["exists"])


def ensure_student_profile_columns(cur):
    cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS admission_course VARCHAR(20)")
    cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS major VARCHAR(120)")


def user_can_manage_course(cur, course_id):
    current_user = get_current_user()
    role_id = int(current_user.get("role_id") or 0)
    if role_id == 1:
        return True
    if role_id != 2:
        return False

    lecturer_id = current_user.get("user_id")
    cur.execute("SELECT 1 FROM courses WHERE id = %s AND lecturer_id = %s", (course_id, lecturer_id))
    return cur.fetchone() is not None


def decode_data_url_to_cv2_image(data_url):
    if not data_url:
        return None

    if "," in data_url:
        data_url = data_url.split(",", 1)[1]

    img_bytes = base64.b64decode(data_url)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def safe_invalidate_embedding_cache():
    try:
        invalidate_embedding_cache()
    except TypeError:
        pass
    except Exception:
        pass


def fetch_students(cur):
    ensure_student_profile_columns(cur)
    has_embeddings_table = table_exists(cur, "embeddings")

    if has_embeddings_table:
        cur.execute("""
            SELECT
                u.id,
                u.full_name,
                u.role_id,
                u.admission_course,
                u.major,
                EXISTS (
                    SELECT 1
                    FROM embeddings e
                    WHERE e.user_id = u.id
                ) AS has_embedding
            FROM users u
            WHERE u.role_id = 3
            ORDER BY u.id
        """)
    else:
        cur.execute("""
            SELECT
                u.id,
                u.full_name,
                u.role_id,
                u.admission_course,
                u.major,
                FALSE AS has_embedding
            FROM users u
            WHERE u.role_id = 3
            ORDER BY u.id
        """)

    return cur.fetchall()


# ============================================================
# Old admin / lecturer APIs that dashboard_admin.html still uses
# ============================================================

@student_bp.route("/api/students", methods=["GET"])
@role_required([1, 2])
def get_students():
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        students = fetch_students(cur)
        conn.commit()
        return jsonify(students), 200
    except Exception as e:
        return jsonify({
            "error": "Loi tai danh sach sinh vien",
            "detail": str(e)
        }), 500
    finally:
        if conn:
            conn.close()


@student_bp.route("/api/students/register_face", methods=["POST"])
@role_required([1, 2])
def register_student_face_from_dashboard():
    data = request.get_json(silent=True) or {}

    student_id = data.get("student_id")
    full_name = data.get("full_name") or data.get("name") or ""
    frames_base64 = data.get("frames", [])

    if not student_id:
        return jsonify({"error": "Thieu student_id"}), 400

    if not full_name:
        return jsonify({"error": "Thieu ho ten sinh vien"}), 400

    if not frames_base64:
        return jsonify({"error": "Chua co anh khuon mat"}), 400

    frames = []

    try:
        for item in frames_base64:
            frame = decode_data_url_to_cv2_image(item)
            if frame is not None:
                frames.append(frame)

        if not frames:
            return jsonify({"error": "Khong decode duoc anh khuon mat"}), 400

        result = register_face(student_id, full_name, frames)
        safe_invalidate_embedding_cache()

        return jsonify({
            "message": "Dang ky khuon mat thanh cong",
            "student_id": student_id,
            "full_name": full_name,
            "result": result
        }), 200

    except Exception as e:
        return jsonify({
            "error": "Dang ky khuon mat that bai",
            "detail": str(e)
        }), 500


@student_bp.route("/api/students/register/zip", methods=["POST"])
@role_required([1, 2])
def register_student_zip():
    zip_file = request.files.get("zip")

    if not zip_file:
        return jsonify({"error": "Chua upload file ZIP"}), 400

    try:
        results = import_from_zip(zip_file)
        safe_invalidate_embedding_cache()
        return jsonify(results), 200
    except Exception as e:
        return jsonify({
            "error": "Import ZIP that bai",
            "detail": str(e)
        }), 500


@student_bp.route("/api/students/register/camera", methods=["POST"])
@role_required([1, 2])
def register_student_camera():
    student_id = request.form.get("student_id")
    name = request.form.get("name")
    images = request.files.getlist("images")

    if not student_id or not name:
        return jsonify({"error": "Thieu MSSV hoac ho ten"}), 400

    if not images or len(images) < 5:
        return jsonify({"error": "Can it nhat 5 anh"}), 400

    try:
        from student_camera import register_from_uploaded_images
        result, status = register_from_uploaded_images(student_id, name, images)
        safe_invalidate_embedding_cache()
        return jsonify(result), status
    except Exception as e:
        return jsonify({
            "error": "Dang ky bang camera that bai",
            "detail": str(e)
        }), 500


# ============================================================
# Course enrollment APIs used by admin / lecturer dashboards
# ============================================================

@student_bp.route("/api/courses/<int:course_id>/students", methods=["POST"])
@role_required([1, 2])
def add_student_to_course(course_id):
    data = request.get_json(silent=True) or {}
    student_id = str(data.get("student_id") or data.get("id") or "").strip()
    full_name = str(data.get("full_name") or data.get("name") or "").strip()
    password = str(data.get("password") or "123456").strip() or "123456"
    admission_course = str(data.get("admission_course") or "").strip() or None
    major = str(data.get("major") or "").strip() or None

    if not student_id:
        return jsonify({"error": "Thiếu MSSV"}), 400

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        ensure_student_profile_columns(cur)

        if not user_can_manage_course(cur, course_id):
            return jsonify({"error": "Không có quyền sửa học phần này"}), 403

        cur.execute("SELECT id, role_id FROM users WHERE id = %s", (student_id,))
        existing_user = cur.fetchone()

        if existing_user and int(existing_user["role_id"]) != 3:
            return jsonify({"error": "Mã số này không thuộc tài khoản sinh viên"}), 400

        if not existing_user:
            if not full_name:
                return jsonify({"error": "Sinh viên chưa tồn tại, cần nhập họ tên để tạo mới"}), 400
            cur.execute("""
                INSERT INTO users (id, full_name, password, role_id, admission_course, major)
                VALUES (%s, %s, %s, 3, %s, %s)
            """, (student_id, full_name, password, admission_course, major))
        else:
            updates = []
            params = []
            if full_name:
                updates.append("full_name = %s")
                params.append(full_name)
            if admission_course is not None:
                updates.append("admission_course = %s")
                params.append(admission_course)
            if major is not None:
                updates.append("major = %s")
                params.append(major)
            if updates:
                params.append(student_id)
                cur.execute(f"UPDATE users SET {', '.join(updates)} WHERE id = %s", params)

        cur.execute("""
            SELECT 1
            FROM course_students
            WHERE course_id = %s AND student_id = %s
        """, (course_id, student_id))
        if cur.fetchone():
            conn.commit()
            return jsonify({"message": "Sinh viên đã có trong học phần", "already_exists": True})

        cur.execute("""
            INSERT INTO course_students (course_id, student_id)
            VALUES (%s, %s)
        """, (course_id, student_id))

        conn.commit()
        return jsonify({"message": "Đã thêm sinh viên vào học phần"}), 201

    except Exception as e:
        if conn:
            conn.rollback()
        return jsonify({"error": "Thêm sinh viên vào học phần thất bại", "detail": str(e)}), 500
    finally:
        if conn:
            conn.close()


@student_bp.route("/api/courses/<int:course_id>/students/<student_id>", methods=["DELETE"])
@role_required([1, 2])
def remove_student_from_course(course_id, student_id):
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        if not user_can_manage_course(cur, course_id):
            return jsonify({"error": "Không có quyền sửa học phần này"}), 403

        cur.execute("""
            DELETE FROM course_students
            WHERE course_id = %s AND student_id = %s
        """, (course_id, student_id))

        conn.commit()
        return jsonify({"message": "Đã xóa sinh viên khỏi học phần"}), 200

    except Exception as e:
        if conn:
            conn.rollback()
        return jsonify({"error": "Xóa sinh viên khỏi học phần thất bại", "detail": str(e)}), 500
    finally:
        if conn:
            conn.close()


# ============================================================
# New APIs for student dashboard
# ============================================================

@student_bp.route("/api/student/profile", methods=["GET"])
@role_required([3])
def student_profile():
    student_id = get_current_student_id()
    conn = None

    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        ensure_student_profile_columns(cur)
        conn.commit()
        has_embeddings_table = table_exists(cur, "embeddings")

        if has_embeddings_table:
            cur.execute("""
                SELECT
                    u.id,
                    u.full_name,
                    u.role_id,
                    u.admission_course,
                    u.major,
                    EXISTS (
                        SELECT 1
                        FROM embeddings e
                        WHERE e.user_id = u.id
                    ) AS has_embedding
                FROM users u
                WHERE u.id = %s AND u.role_id = 3
            """, (student_id,))
        else:
            cur.execute("""
                SELECT
                    u.id,
                    u.full_name,
                    u.role_id,
                    u.admission_course,
                    u.major,
                    FALSE AS has_embedding
                FROM users u
                WHERE u.id = %s AND u.role_id = 3
            """, (student_id,))

        student = cur.fetchone()
        if not student:
            return jsonify({"error": "Khong tim thay sinh vien"}), 404
        return jsonify(student), 200

    except Exception as e:
        return jsonify({
            "error": "Loi tai thong tin sinh vien",
            "detail": str(e)
        }), 500
    finally:
        if conn:
            conn.close()


@student_bp.route("/api/student/courses", methods=["GET"])
@role_required([3])
def student_courses():
    student_id = get_current_student_id()
    conn = None

    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT
                c.id,
                c.name,
                c.semester,
                c.room,
                c.period,
                c.credits,
                c.lecturer_id,
                lecturer.full_name AS lecturer_name,
                COUNT(a.id) AS total_attendance,
                COUNT(a.id) FILTER (WHERE a.recognized = TRUE) AS present_count,
                COUNT(a.id) FILTER (WHERE a.recognized = FALSE) AS unrecognized_count,
                MAX(a.time) AS last_attendance_time
            FROM course_students cs
            JOIN courses c
                ON c.id = cs.course_id
            LEFT JOIN users lecturer
                ON lecturer.id = c.lecturer_id
            LEFT JOIN attendance a
                ON a.course_id = c.id
               AND a.student_id = CAST(cs.student_id AS VARCHAR)
            WHERE cs.student_id = CAST(%s AS VARCHAR)
            GROUP BY
                c.id,
                c.name,
                c.semester,
                c.room,
                c.period,
                c.credits,
                c.lecturer_id,
                lecturer.full_name
            ORDER BY c.id DESC
        """, (student_id,))
        return jsonify(cur.fetchall()), 200

    except Exception as e:
        return jsonify({
            "error": "Loi tai hoc phan cua sinh vien",
            "detail": str(e)
        }), 500
    finally:
        if conn:
            conn.close()


@student_bp.route("/api/student/attendance", methods=["GET"])
@role_required([3])
def student_attendance():
    student_id = get_current_student_id()
    course_id = request.args.get("course_id")
    date_value = request.args.get("date")
    conn = None

    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        conditions = ["a.student_id = %s"]
        params = [student_id]

        if course_id:
            conditions.append("a.course_id = %s")
            params.append(course_id)

        if date_value:
            conditions.append("DATE(a.time) = %s")
            params.append(date_value)

        where_sql = " AND ".join(conditions)

        cur.execute(f"""
            SELECT
                a.id,
                a.student_id,
                student.full_name AS student_name,
                a.course_id,
                c.name AS course_name,
                c.semester,
                a.time,
                a.image_base64,
                a.recognized
            FROM attendance a
            LEFT JOIN users student
                ON student.id = a.student_id
            LEFT JOIN courses c
                ON c.id = a.course_id
            WHERE {where_sql}
            ORDER BY a.time DESC
            LIMIT 300
        """, params)
        return jsonify(cur.fetchall()), 200

    except Exception as e:
        return jsonify({
            "error": "Loi tai lich su diem danh",
            "detail": str(e)
        }), 500
    finally:
        if conn:
            conn.close()


@student_bp.route("/api/student/summary", methods=["GET"])
@role_required([3])
def student_summary():
    student_id = get_current_student_id()
    conn = None

    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute("""
            SELECT COUNT(*) AS total_courses
            FROM course_students
            WHERE student_id = %s
        """, (student_id,))
        total_courses = cur.fetchone()["total_courses"] or 0

        cur.execute("""
            SELECT
                COUNT(*) AS total_attendance,
                COUNT(*) FILTER (WHERE recognized = TRUE) AS present_count,
                COUNT(*) FILTER (WHERE recognized = FALSE) AS unrecognized_count
            FROM attendance
            WHERE student_id = %s
        """, (student_id,))
        row = cur.fetchone()

        total_attendance = row["total_attendance"] or 0
        present_count = row["present_count"] or 0
        unrecognized_count = row["unrecognized_count"] or 0

        attendance_rate = 0
        if total_attendance > 0:
            attendance_rate = round((present_count / total_attendance) * 100, 2)

        return jsonify({
            "total_courses": total_courses,
            "total_attendance": total_attendance,
            "present_count": present_count,
            "unrecognized_count": unrecognized_count,
            "attendance_rate": attendance_rate
        }), 200

    except Exception as e:
        return jsonify({
            "error": "Loi tai thong ke sinh vien",
            "detail": str(e)
        }), 500
    finally:
        if conn:
            conn.close()


# ============================================================
# Fallback endpoints if UI calls by student_id
# ============================================================

@student_bp.route("/api/students/<student_id>/courses", methods=["GET"])
@role_required([1, 2, 3])
def student_courses_by_id(student_id):
    current_user = get_current_user()

    if current_user.get("role_id") == 3 and current_user.get("user_id") != student_id:
        return jsonify({"error": "Khong co quyen truy cap"}), 403

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT
                c.id,
                c.name,
                c.semester,
                c.room,
                c.period,
                c.credits,
                c.lecturer_id,
                lecturer.full_name AS lecturer_name
            FROM course_students cs
            JOIN courses c
                ON c.id = cs.course_id
            LEFT JOIN users lecturer
                ON lecturer.id = c.lecturer_id
            WHERE cs.student_id = CAST(%s AS VARCHAR)
            ORDER BY c.id DESC
        """, (student_id,))
        return jsonify(cur.fetchall()), 200
    except Exception as e:
        return jsonify({
            "error": "Loi tai hoc phan cua sinh vien",
            "detail": str(e)
        }), 500
    finally:
        if conn:
            conn.close()


@student_bp.route("/api/students/<student_id>/attendance", methods=["GET"])
@role_required([1, 2, 3])
def student_attendance_by_id(student_id):
    current_user = get_current_user()

    if current_user.get("role_id") == 3 and current_user.get("user_id") != student_id:
        return jsonify({"error": "Khong co quyen truy cap"}), 403

    course_id = request.args.get("course_id")
    date_value = request.args.get("date")
    conn = None

    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        conditions = ["a.student_id = %s"]
        params = [student_id]

        if course_id:
            conditions.append("a.course_id = %s")
            params.append(course_id)

        if date_value:
            conditions.append("DATE(a.time) = %s")
            params.append(date_value)

        where_sql = " AND ".join(conditions)

        cur.execute(f"""
            SELECT
                a.id,
                a.student_id,
                student.full_name AS student_name,
                a.course_id,
                c.name AS course_name,
                c.semester,
                a.time,
                a.image_base64,
                a.recognized
            FROM attendance a
            LEFT JOIN users student
                ON student.id = a.student_id
            LEFT JOIN courses c
                ON c.id = a.course_id
            WHERE {where_sql}
            ORDER BY a.time DESC
            LIMIT 300
        """, params)
        return jsonify(cur.fetchall()), 200
    except Exception as e:
        return jsonify({
            "error": "Loi tai lich su diem danh",
            "detail": str(e)
        }), 500
    finally:
        if conn:
            conn.close()