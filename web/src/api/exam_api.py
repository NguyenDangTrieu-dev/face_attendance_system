from datetime import date, datetime, time

from flask import Blueprint, jsonify, request
from psycopg2.extras import RealDictCursor

from auth import role_required
from db import get_db_connection

exam_bp = Blueprint("exam", __name__)


def _val(v):
    if isinstance(v, (datetime, date, time)):
        return v.isoformat()
    return v


def row_to_dict(row):
    return {k: _val(v) for k, v in dict(row).items()}


def rows_to_list(rows):
    return [row_to_dict(r) for r in rows]


def get_payload():
    return request.get_json(silent=True) or {}


def required(data, fields):
    miss = [f for f in fields if not data.get(f)]
    return f"Thiếu thông tin: {', '.join(miss)}" if miss else None


def get_exam(cur, exam_id):
    cur.execute("""
        SELECT es.*, c.name AS course_name, u.full_name AS lecturer_name
        FROM exam_sessions es
        JOIN courses c ON c.id = es.course_id
        LEFT JOIN users u ON u.id = es.lecturer_id
        WHERE es.id = %s
    """, (exam_id,))
    return cur.fetchone()


@exam_bp.route("/api/admin/exam-sessions", methods=["GET"])
@role_required([1])
def admin_list_exam_sessions():
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT
                es.id, es.name, es.course_id, c.name AS course_name,
                es.lecturer_id, u.full_name AS lecturer_name,
                es.exam_date, es.start_time, es.end_time,
                es.room, es.note, es.status, es.created_at,
                COUNT(cs.student_id) AS total_students,
                COUNT(ea.student_id) FILTER (WHERE ea.recognized = TRUE) AS present_count
            FROM exam_sessions es
            JOIN courses c ON c.id = es.course_id
            LEFT JOIN users u ON u.id = es.lecturer_id
            LEFT JOIN course_students cs ON cs.course_id = es.course_id
            LEFT JOIN exam_attendance ea
                ON ea.exam_session_id = es.id
               AND ea.student_id = cs.student_id
            GROUP BY es.id, c.name, u.full_name
            ORDER BY es.exam_date DESC, es.start_time DESC, es.id DESC
        """)
        return jsonify(rows_to_list(cur.fetchall())), 200
    except Exception as e:
        return jsonify({"error": "Không tải được danh sách ca thi", "detail": str(e)}), 500
    finally:
        if conn:
            conn.close()


@exam_bp.route("/api/admin/exam-sessions", methods=["POST"])
@role_required([1])
def admin_create_exam_session():
    data = get_payload()
    err = required(data, ["name", "course_id", "lecturer_id", "exam_date", "start_time", "end_time"])
    if err:
        return jsonify({"error": err}), 400
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            INSERT INTO exam_sessions
                (name, course_id, lecturer_id, exam_date, start_time, end_time, room, note, status)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,COALESCE(%s,'upcoming'))
            RETURNING *
        """, (
            data["name"], data["course_id"], data["lecturer_id"],
            data["exam_date"], data["start_time"], data["end_time"],
            data.get("room"), data.get("note"), data.get("status")
        ))
        row = cur.fetchone()
        conn.commit()
        return jsonify(row_to_dict(row)), 201
    except Exception as e:
        if conn:
            conn.rollback()
        return jsonify({"error": "Không tạo được ca thi", "detail": str(e)}), 500
    finally:
        if conn:
            conn.close()


@exam_bp.route("/api/admin/exam-sessions/<int:exam_id>", methods=["PUT"])
@role_required([1])
def admin_update_exam_session(exam_id):
    data = get_payload()
    err = required(data, ["name", "course_id", "lecturer_id", "exam_date", "start_time", "end_time"])
    if err:
        return jsonify({"error": err}), 400
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            UPDATE exam_sessions
            SET name=%s, course_id=%s, lecturer_id=%s, exam_date=%s,
                start_time=%s, end_time=%s, room=%s, note=%s, status=%s
            WHERE id=%s
            RETURNING *
        """, (
            data["name"], data["course_id"], data["lecturer_id"],
            data["exam_date"], data["start_time"], data["end_time"],
            data.get("room"), data.get("note"), data.get("status", "upcoming"), exam_id
        ))
        row = cur.fetchone()
        if not row:
            conn.rollback()
            return jsonify({"error": "Không tìm thấy ca thi"}), 404
        conn.commit()
        return jsonify(row_to_dict(row)), 200
    except Exception as e:
        if conn:
            conn.rollback()
        return jsonify({"error": "Không cập nhật được ca thi", "detail": str(e)}), 500
    finally:
        if conn:
            conn.close()


@exam_bp.route("/api/admin/exam-sessions/<int:exam_id>", methods=["DELETE"])
@role_required([1])
def admin_delete_exam_session(exam_id):
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM exam_sessions WHERE id=%s", (exam_id,))
        if cur.rowcount == 0:
            conn.rollback()
            return jsonify({"error": "Không tìm thấy ca thi"}), 404
        conn.commit()
        return jsonify({"message": "Đã xóa ca thi"}), 200
    except Exception as e:
        if conn:
            conn.rollback()
        return jsonify({"error": "Không xóa được ca thi", "detail": str(e)}), 500
    finally:
        if conn:
            conn.close()


@exam_bp.route("/api/lecturer/<lecturer_id>/exam-sessions", methods=["GET"])
@role_required([1, 2])
def lecturer_list_exam_sessions(lecturer_id):
    user = getattr(request, "user", {}) or {}
    if user.get("role_id") == 2 and user.get("user_id") != lecturer_id:
        return jsonify({"error": "Không có quyền truy cập"}), 403
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT
                es.id, es.name, es.course_id, c.name AS course_name,
                es.lecturer_id, u.full_name AS lecturer_name,
                es.exam_date, es.start_time, es.end_time,
                es.room, es.note, es.status, es.created_at,
                COUNT(cs.student_id) AS total_students,
                COUNT(ea.student_id) FILTER (WHERE ea.recognized = TRUE) AS present_count
            FROM exam_sessions es
            JOIN courses c ON c.id = es.course_id
            LEFT JOIN users u ON u.id = es.lecturer_id
            LEFT JOIN course_students cs ON cs.course_id = es.course_id
            LEFT JOIN exam_attendance ea
                ON ea.exam_session_id = es.id
               AND ea.student_id = cs.student_id
            WHERE es.lecturer_id = %s
            GROUP BY es.id, c.name, u.full_name
            ORDER BY es.exam_date DESC, es.start_time DESC, es.id DESC
        """, (lecturer_id,))
        return jsonify(rows_to_list(cur.fetchall())), 200
    except Exception as e:
        return jsonify({"error": "Không tải được ca thi của giảng viên", "detail": str(e)}), 500
    finally:
        if conn:
            conn.close()


@exam_bp.route("/api/exam-sessions/<int:exam_id>/students", methods=["GET"])
@role_required([1, 2])
def exam_session_students(exam_id):
    user = getattr(request, "user", {}) or {}
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        exam = get_exam(cur, exam_id)
        if not exam:
            return jsonify({"error": "Không tìm thấy ca thi"}), 404
        if user.get("role_id") == 2 and exam["lecturer_id"] != user.get("user_id"):
            return jsonify({"error": "Không có quyền truy cập"}), 403
        cur.execute("""
            SELECT
                u.id, u.full_name,
                EXISTS (SELECT 1 FROM embeddings e WHERE e.user_id = u.id) AS has_embedding,
                ea.time AS attendance_time,
                COALESCE(ea.recognized, FALSE) AS recognized,
                ea.similarity
            FROM course_students cs
            JOIN users u ON u.id = cs.student_id
            LEFT JOIN exam_attendance ea
                ON ea.exam_session_id = %s
               AND ea.student_id = u.id
            WHERE cs.course_id = %s
            ORDER BY u.id
        """, (exam_id, exam["course_id"]))
        return jsonify(rows_to_list(cur.fetchall())), 200
    except Exception as e:
        return jsonify({"error": "Không tải được sinh viên ca thi", "detail": str(e)}), 500
    finally:
        if conn:
            conn.close()


@exam_bp.route("/api/exam-sessions/<int:exam_id>/attendance", methods=["GET"])
@role_required([1, 2])
def exam_session_attendance(exam_id):
    user = getattr(request, "user", {}) or {}
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        exam = get_exam(cur, exam_id)
        if not exam:
            return jsonify({"error": "Không tìm thấy ca thi"}), 404
        if user.get("role_id") == 2 and exam["lecturer_id"] != user.get("user_id"):
            return jsonify({"error": "Không có quyền truy cập"}), 403
        cur.execute("""
            SELECT ea.id, ea.exam_session_id, ea.student_id,
                   u.full_name AS student_name,
                   ea.time, ea.image_base64, ea.recognized, ea.similarity
            FROM exam_attendance ea
            JOIN users u ON u.id = ea.student_id
            WHERE ea.exam_session_id = %s
            ORDER BY ea.time DESC
        """, (exam_id,))
        return jsonify(rows_to_list(cur.fetchall())), 200
    except Exception as e:
        return jsonify({"error": "Không tải được lịch sử điểm danh ca thi", "detail": str(e)}), 500
    finally:
        if conn:
            conn.close()


@exam_bp.route("/api/exam-sessions/<int:exam_id>/summary", methods=["GET"])
@role_required([1, 2])
def exam_session_summary(exam_id):
    user = getattr(request, "user", {}) or {}
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        exam = get_exam(cur, exam_id)
        if not exam:
            return jsonify({"error": "Không tìm thấy ca thi"}), 404
        if user.get("role_id") == 2 and exam["lecturer_id"] != user.get("user_id"):
            return jsonify({"error": "Không có quyền truy cập"}), 403
        cur.execute("""
            SELECT
                COUNT(cs.student_id) AS total_students,
                COUNT(ea.student_id) FILTER (WHERE ea.recognized = TRUE) AS present_count,
                COUNT(cs.student_id) - COUNT(ea.student_id) FILTER (WHERE ea.recognized = TRUE) AS absent_count
            FROM course_students cs
            LEFT JOIN exam_attendance ea
                ON ea.exam_session_id = %s
               AND ea.student_id = cs.student_id
            WHERE cs.course_id = %s
        """, (exam_id, exam["course_id"]))
        row = cur.fetchone()
        total = row["total_students"] or 0
        present = row["present_count"] or 0
        absent = row["absent_count"] or 0
        rate = round((present / total) * 100, 2) if total else 0
        return jsonify({
            "exam": row_to_dict(exam),
            "total_students": total,
            "present_count": present,
            "absent_count": absent,
            "attendance_rate": rate
        }), 200
    except Exception as e:
        return jsonify({"error": "Không tải được thống kê ca thi", "detail": str(e)}), 500
    finally:
        if conn:
            conn.close()
