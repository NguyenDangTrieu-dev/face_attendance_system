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
    has_embeddings_table = table_exists(cur, "embeddings")

    if has_embeddings_table:
        cur.execute("""
            SELECT
                u.id,
                u.full_name,
                u.role_id,
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
        return jsonify(fetch_students(cur)), 200
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
        has_embeddings_table = table_exists(cur, "embeddings")

        if has_embeddings_table:
            cur.execute("""
                SELECT
                    u.id,
                    u.full_name,
                    u.role_id,
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
                COUNT(a.*) AS total_attendance,
                COUNT(a.*) FILTER (WHERE a.recognized = TRUE) AS present_count,
                COUNT(a.*) FILTER (WHERE a.recognized = FALSE) AS unrecognized_count,
                MAX(a.time) AS last_attendance_time
            FROM course_students cs
            JOIN courses c
                ON c.id = cs.course_id
            LEFT JOIN users lecturer
                ON lecturer.id = c.lecturer_id
            LEFT JOIN attendance a
                ON a.course_id = c.id
               AND a.student_id = cs.student_id
            WHERE cs.student_id = %s
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
            WHERE cs.student_id = %s
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
