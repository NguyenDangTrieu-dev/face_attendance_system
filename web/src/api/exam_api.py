from datetime import date, datetime, time, timezone, timedelta

from flask import Blueprint, jsonify, request
from psycopg2.extras import RealDictCursor

from auth import role_required
from db import get_db_connection
from face_rec_SM_api import image_to_base64

exam_bp = Blueprint("exam", __name__)

# ── Timezone Vietnam (UTC+7) ───────────────────────────────────────────────────
VN_TZ = timezone(timedelta(hours=7))


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


def compute_exam_status(exam_date_val, start_time_val, end_time_val):
    """
    Tự động tính status dựa trên thời gian thực (Vietnam UTC+7).

    exam_date_val : date | str  (YYYY-MM-DD)
    start_time_val: time | str  (HH:MM hoặc HH:MM:SS)
    end_time_val  : time | str  (HH:MM hoặc HH:MM:SS)

    Returns: 'upcoming' | 'ongoing' | 'finished'
    """
    try:
        if isinstance(exam_date_val, str):
            exam_date_val = date.fromisoformat(exam_date_val)
        if isinstance(start_time_val, str):
            start_time_val = time.fromisoformat(start_time_val)
        if isinstance(end_time_val, str):
            end_time_val = time.fromisoformat(end_time_val)

        now = datetime.now(VN_TZ).replace(tzinfo=None)
        dt_start = datetime.combine(exam_date_val, start_time_val)
        dt_end   = datetime.combine(exam_date_val, end_time_val)

        if now < dt_start:
            return "upcoming"
        if now <= dt_end:
            return "ongoing"
        return "finished"
    except Exception:
        return "upcoming"


def validate_exam_times(data):
    """
    Kiểm tra logic thời gian khi tạo / cập nhật ca thi.
    Trả về chuỗi lỗi nếu có, None nếu hợp lệ.
    """
    try:
        exam_date_val  = data.get("exam_date")
        start_time_val = data.get("start_time")
        end_time_val   = data.get("end_time")

        if isinstance(exam_date_val, str):
            exam_date_val = date.fromisoformat(exam_date_val)
        if isinstance(start_time_val, str):
            start_time_val = time.fromisoformat(start_time_val)
        if isinstance(end_time_val, str):
            end_time_val = time.fromisoformat(end_time_val)

        if end_time_val <= start_time_val:
            return "Thời gian kết thúc phải sau thời gian bắt đầu"

        now_date = datetime.now(VN_TZ).date()
        if exam_date_val < now_date:
            return "Ngày thi không được ở trong quá khứ"

        return None
    except (ValueError, TypeError) as exc:
        return f"Định dạng ngày/giờ không hợp lệ: {exc}"


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
        rows = rows_to_list(cur.fetchall())
        for r in rows:
            r["status"] = compute_exam_status(r.get("exam_date"), r.get("start_time"), r.get("end_time"))
        return jsonify(rows), 200
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
    time_err = validate_exam_times(data)
    if time_err:
        return jsonify({"error": time_err}), 400
    auto_status = compute_exam_status(data["exam_date"], data["start_time"], data["end_time"])
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            INSERT INTO exam_sessions
                (name, course_id, lecturer_id, exam_date, start_time, end_time, room, note, status)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING *
        """, (
            data["name"], data["course_id"], data["lecturer_id"],
            data["exam_date"], data["start_time"], data["end_time"],
            data.get("room"), data.get("note"), auto_status
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
    time_err = validate_exam_times(data)
    if time_err:
        return jsonify({"error": time_err}), 400
    auto_status = compute_exam_status(data["exam_date"], data["start_time"], data["end_time"])
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
            data.get("room"), data.get("note"), auto_status, exam_id
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
        rows = rows_to_list(cur.fetchall())
        for r in rows:
            r["status"] = compute_exam_status(r.get("exam_date"), r.get("start_time"), r.get("end_time"))
        return jsonify(rows), 200
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
                (ea.time AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Ho_Chi_Minh') AS attendance_time,
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

# ============================================================
# STUDENT: ca thi của sinh viên
# ============================================================

@exam_bp.route("/api/student/exam-sessions", methods=["GET"])
@role_required([3])
def student_list_exam_sessions():
    user = getattr(request, "user", {}) or {}
    student_id = user.get("user_id")

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute("""
            SELECT
                es.id,
                es.name,
                es.course_id,
                c.name AS course_name,
                es.lecturer_id,
                u.full_name AS lecturer_name,
                es.exam_date,
                es.start_time,
                es.end_time,
                es.room,
                es.note,
                es.status,
                es.created_at,
                ea.time AS attendance_time,
                COALESCE(ea.recognized, FALSE) AS recognized,
                ea.similarity
            FROM exam_sessions es
            JOIN courses c
                ON c.id = es.course_id
            JOIN course_students cs
                ON cs.course_id = es.course_id
               AND cs.student_id = %s
            LEFT JOIN users u
                ON u.id = es.lecturer_id
            LEFT JOIN exam_attendance ea
                ON ea.exam_session_id = es.id
               AND ea.student_id = cs.student_id
            ORDER BY es.exam_date DESC, es.start_time DESC, es.id DESC
        """, (student_id,))

        rows = rows_to_list(cur.fetchall())
        for r in rows:
            r["status"] = compute_exam_status(r.get("exam_date"), r.get("start_time"), r.get("end_time"))
        return jsonify(rows), 200

    except Exception as e:
        return jsonify({
            "error": "Không tải được ca thi của sinh viên",
            "detail": str(e)
        }), 500

    finally:
        if conn:
            conn.close()


@exam_bp.route("/api/students/<student_id>/exam-sessions", methods=["GET"])
@role_required([1, 2, 3])
def student_list_exam_sessions_by_id(student_id):
    user = getattr(request, "user", {}) or {}

    if user.get("role_id") == 3 and user.get("user_id") != student_id:
        return jsonify({"error": "Không có quyền truy cập"}), 403

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute("""
            SELECT
                es.id,
                es.name,
                es.course_id,
                c.name AS course_name,
                es.lecturer_id,
                u.full_name AS lecturer_name,
                es.exam_date,
                es.start_time,
                es.end_time,
                es.room,
                es.note,
                es.status,
                es.created_at,
                ea.time AS attendance_time,
                COALESCE(ea.recognized, FALSE) AS recognized,
                ea.similarity
            FROM exam_sessions es
            JOIN courses c
                ON c.id = es.course_id
            JOIN course_students cs
                ON cs.course_id = es.course_id
               AND cs.student_id = %s
            LEFT JOIN users u
                ON u.id = es.lecturer_id
            LEFT JOIN exam_attendance ea
                ON ea.exam_session_id = es.id
               AND ea.student_id = cs.student_id
            ORDER BY es.exam_date DESC, es.start_time DESC, es.id DESC
        """, (student_id,))

        rows = rows_to_list(cur.fetchall())
        for r in rows:
            r["status"] = compute_exam_status(r.get("exam_date"), r.get("start_time"), r.get("end_time"))
        return jsonify(rows), 200

    except Exception as e:
        return jsonify({
            "error": "Không tải được ca thi của sinh viên",
            "detail": str(e)
        }), 500

    finally:
        if conn:
            conn.close()


@exam_bp.route("/api/student/exam-sessions/<int:exam_id>/attendance", methods=["GET"])
@role_required([3])
def student_exam_attendance(exam_id):
    user = getattr(request, "user", {}) or {}
    student_id = user.get("user_id")

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute("""
            SELECT 1
            FROM exam_sessions es
            JOIN course_students cs
                ON cs.course_id = es.course_id
            WHERE es.id = %s
              AND cs.student_id = %s
            LIMIT 1
        """, (exam_id, student_id))

        if not cur.fetchone():
            return jsonify({"error": "Bạn không thuộc ca thi này"}), 403

        cur.execute("""
            SELECT
                ea.id,
                ea.exam_session_id,
                ea.student_id,
                u.full_name AS student_name,
                ea.time,
                ea.image_base64,
                ea.recognized,
                ea.similarity
            FROM exam_attendance ea
            JOIN users u
                ON u.id = ea.student_id
            WHERE ea.exam_session_id = %s
              AND ea.student_id = %s
            LIMIT 1
        """, (exam_id, student_id))

        row = cur.fetchone()

        if not row:
            return jsonify({
                "exam_session_id": exam_id,
                "student_id": student_id,
                "recognized": False,
                "time": None,
                "image_base64": None,
                "similarity": None
            }), 200

        return jsonify(row_to_dict(row)), 200

    except Exception as e:
        return jsonify({
            "error": "Không tải được điểm danh ca thi của sinh viên",
            "detail": str(e)
        }), 500

    finally:
        if conn:
            conn.close()


@exam_bp.route("/api/students/<student_id>/exam-sessions/<int:exam_id>/attendance", methods=["GET"])
@role_required([1, 2, 3])
def student_exam_attendance_by_id(student_id, exam_id):
    user = getattr(request, "user", {}) or {}

    if user.get("role_id") == 3 and user.get("user_id") != student_id:
        return jsonify({"error": "Không có quyền truy cập"}), 403

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute("""
            SELECT 1
            FROM exam_sessions es
            JOIN course_students cs
                ON cs.course_id = es.course_id
            WHERE es.id = %s
              AND cs.student_id = %s
            LIMIT 1
        """, (exam_id, student_id))

        if not cur.fetchone():
            return jsonify({"error": "Sinh viên không thuộc ca thi này"}), 403

        cur.execute("""
            SELECT
                ea.id,
                ea.exam_session_id,
                ea.student_id,
                u.full_name AS student_name,
                ea.time,
                ea.image_base64,
                ea.recognized,
                ea.similarity
            FROM exam_attendance ea
            JOIN users u
                ON u.id = ea.student_id
            WHERE ea.exam_session_id = %s
              AND ea.student_id = %s
            LIMIT 1
        """, (exam_id, student_id))

        row = cur.fetchone()

        if not row:
            return jsonify({
                "exam_session_id": exam_id,
                "student_id": student_id,
                "recognized": False,
                "time": None,
                "image_base64": None,
                "similarity": None
            }), 200

        return jsonify(row_to_dict(row)), 200

    except Exception as e:
        return jsonify({
            "error": "Không tải được điểm danh ca thi của sinh viên",
            "detail": str(e)
        }), 500

    finally:
        if conn:
            conn.close()
            
# ============================================================
# REALTIME ĐIỂM DANH CA THI
# ============================================================
# Cách hoạt động:
# - main.py gọi bind_exam_realtime_engine(engine)
# - Khi chạy realtime học phần bình thường, callback trả False, RealtimeEngine lưu vào attendance như cũ.
# - Khi chạy realtime ca thi, callback lưu vào exam_attendance và trả True để RealtimeEngine KHÔNG lưu vào attendance.

_EXAM_REALTIME_ENGINE = None
_EXAM_REALTIME_CONTEXT = {
    "mode": "course",       # course | exam
    "exam_session_id": None,
    "course_id": None,
}


def bind_exam_realtime_engine(engine):
    """
    Gắn RealtimeEngine vào exam_api.

    Gọi trong main.py ngay sau khi tạo engine:
        engine = RealtimeEngine()
        bind_exam_realtime_engine(engine)
    """
    global _EXAM_REALTIME_ENGINE

    _EXAM_REALTIME_ENGINE = engine
    engine.on_attendance = _handle_exam_realtime_attendance


def _track_attr(track, attr, default=None):
    if isinstance(track, dict):
        return track.get(attr, default)
    return getattr(track, attr, default)


def _student_in_exam_course(student_id, exam_session_id):
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT 1
            FROM exam_sessions es
            JOIN course_students cs
                ON cs.course_id = es.course_id
            WHERE es.id = %s
              AND cs.student_id = %s
            LIMIT 1
        """, (exam_session_id, student_id))
        return cur.fetchone() is not None
    finally:
        if conn:
            conn.close()


def _save_exam_attendance(student_id, exam_session_id, image_base64, similarity=None):
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO exam_attendance (
                exam_session_id,
                student_id,
                time,
                image_base64,
                recognized,
                similarity
            )
            VALUES (%s, %s, NOW() AT TIME ZONE 'Asia/Ho_Chi_Minh', %s, TRUE, %s)
            ON CONFLICT (exam_session_id, student_id)
            DO UPDATE SET
                time = EXCLUDED.time,
                image_base64 = COALESCE(EXCLUDED.image_base64, exam_attendance.image_base64),
                recognized = TRUE,
                similarity = COALESCE(EXCLUDED.similarity, exam_attendance.similarity)
        """, (
            exam_session_id,
            student_id,
            image_base64,
            similarity
        ))

        conn.commit()

    except Exception:
        if conn:
            conn.rollback()
        raise

    finally:
        if conn:
            conn.close()


def _handle_exam_realtime_attendance(track, course_id, face_img, timestamp):
    """
    Callback được RealtimeEngine gọi khi nhận diện được sinh viên.

    Return:
        True  -> đã xử lý, RealtimeEngine không lưu attendance thường.
        False -> chưa xử lý, RealtimeEngine tiếp tục lưu attendance thường.
    """
    if _EXAM_REALTIME_CONTEXT.get("mode") != "exam":
        return False

    exam_session_id = _EXAM_REALTIME_CONTEXT.get("exam_session_id")

    if not exam_session_id:
        return True

    student_id = _track_attr(track, "student_id")
    if not student_id:
        return True

    student_id = str(student_id)

    # Nếu người được nhận diện không thuộc học phần của ca thi,
    # không lưu vào exam_attendance và cũng không lưu nhầm vào attendance.
    if not _student_in_exam_course(student_id, exam_session_id):
        return True

    similarity = _track_attr(track, "similarity")

    try:
        if face_img is not None:
            image_base64 = image_to_base64(face_img)
        else:
            image_base64 = None

        _save_exam_attendance(
            student_id=student_id,
            exam_session_id=exam_session_id,
            image_base64=image_base64,
            similarity=similarity
        )

        return True

    except Exception:
        # Nếu lưu exam_attendance lỗi, vẫn trả True để tránh lưu nhầm vào attendance.
        raise


@exam_bp.route("/api/exam-sessions/<int:exam_id>/realtime/start", methods=["POST"])
@role_required([2])
def exam_realtime_start(exam_id):
    user = getattr(request, "user", {}) or {}

    if _EXAM_REALTIME_ENGINE is None:
        return jsonify({"error": "RealtimeEngine chưa được gắn vào exam_api"}), 500

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        exam = get_exam(cur, exam_id)
        if not exam:
            return jsonify({"error": "Không tìm thấy ca thi"}), 404

        if exam["lecturer_id"] != user.get("user_id"):
            return jsonify({"error": "Không có quyền mở ca thi này"}), 403

        # Kiểm tra thời gian: chỉ cho phép bật trong khoảng [start_time - 15 phút, end_time]
        current_status = compute_exam_status(
            exam["exam_date"], exam["start_time"], exam["end_time"]
        )
        if current_status == "finished":
            return jsonify({
                "error": "Ca thi đã kết thúc, không thể bật điểm danh"
            }), 400
        if current_status == "upcoming":
            # Tính thời gian còn lại đến khi bắt đầu
            now = datetime.now(VN_TZ).replace(tzinfo=None)
            exam_date = exam["exam_date"] if isinstance(exam["exam_date"], date) else date.fromisoformat(str(exam["exam_date"]))
            start_t   = exam["start_time"] if isinstance(exam["start_time"], time) else time.fromisoformat(str(exam["start_time"]))
            dt_start  = datetime.combine(exam_date, start_t)
            minutes_until = (dt_start - now).total_seconds() / 60
            if minutes_until > 15:
                return jsonify({
                    "error": (
                        f"Ca thi chưa đến giờ bắt đầu. "
                        f"Chỉ có thể bật điểm danh sớm nhất 15 phút trước giờ thi "
                        f"({int(minutes_until)} phút nữa mới đến giờ)."
                    )
                }), 400

        # Reset camera nếu đang chạy ở học phần hoặc ca thi khác
        if _EXAM_REALTIME_ENGINE.is_running():
            _EXAM_REALTIME_ENGINE.stop()

        _EXAM_REALTIME_CONTEXT["mode"] = "exam"
        _EXAM_REALTIME_CONTEXT["exam_session_id"] = exam_id
        _EXAM_REALTIME_CONTEXT["course_id"] = exam["course_id"]

        _EXAM_REALTIME_ENGINE.start(course_id=int(exam["course_id"]))

        return jsonify({
            "running": True,
            "mode": "exam",
            "exam_session_id": exam_id,
            "course_id": exam["course_id"],
            "exam_name": exam["name"]
        }), 200

    except Exception as e:
        _EXAM_REALTIME_CONTEXT["mode"] = "course"
        _EXAM_REALTIME_CONTEXT["exam_session_id"] = None
        _EXAM_REALTIME_CONTEXT["course_id"] = None

        return jsonify({
            "error": "Không thể khởi động realtime ca thi",
            "detail": str(e)
        }), 500

    finally:
        if conn:
            conn.close()


@exam_bp.route("/api/exam-sessions/<int:exam_id>/realtime/stop", methods=["POST"])
@role_required([2])
def exam_realtime_stop(exam_id):
    try:
        if _EXAM_REALTIME_ENGINE is not None and _EXAM_REALTIME_ENGINE.is_running():
            _EXAM_REALTIME_ENGINE.stop()

        _EXAM_REALTIME_CONTEXT["mode"] = "course"
        _EXAM_REALTIME_CONTEXT["exam_session_id"] = None
        _EXAM_REALTIME_CONTEXT["course_id"] = None

        return jsonify({
            "running": False,
            "mode": "exam",
            "exam_session_id": exam_id
        }), 200

    except Exception as e:
        return jsonify({
            "error": "Không thể dừng realtime ca thi",
            "detail": str(e)
        }), 500


@exam_bp.route("/api/exam-sessions/<int:exam_id>/realtime/status", methods=["GET"])
@role_required([2])
def exam_realtime_status(exam_id):
    running = bool(
        _EXAM_REALTIME_ENGINE is not None
        and _EXAM_REALTIME_ENGINE.is_running()
        and _EXAM_REALTIME_CONTEXT.get("mode") == "exam"
        and _EXAM_REALTIME_CONTEXT.get("exam_session_id") == exam_id
    )

    return jsonify({
        "running": running,
        "mode": _EXAM_REALTIME_CONTEXT.get("mode"),
        "exam_session_id": _EXAM_REALTIME_CONTEXT.get("exam_session_id"),
        "course_id": _EXAM_REALTIME_CONTEXT.get("course_id")
    }), 200