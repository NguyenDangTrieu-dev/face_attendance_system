from flask import Blueprint, jsonify
from db import get_db_connection
from psycopg2.extras import RealDictCursor

attendance_bp = Blueprint('attendance', __name__)


def already_attended_today(student_id, course_id):
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT 1 FROM attendance
                WHERE student_id=%s
                  AND course_id=%s
                  AND DATE(time)=CURRENT_DATE
                LIMIT 1
            """, (student_id, course_id))
            return cursor.fetchone() is not None
    finally:
        conn.close()
def is_student_in_course(student_id, course_id):
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT 1
                FROM course_students
                WHERE student_id = %s
                  AND course_id = %s
                LIMIT 1
            """, (student_id, course_id))

            return cursor.fetchone() is not None
    finally:
        conn.close()
        
def save_attendance(student_id, course_id, image_base64):
    conn = None
    cursor = None
    try:
        # Check duplicate
        if already_attended_today(student_id, course_id):

            return {"success": False, "message": "Already attended today"}

        # Check sinh viên thuộc lớp
        if not is_student_in_course(student_id, course_id):

            return {"success": False, "message": "Student not in course"}

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO attendance (
                student_id,
                course_id,
                image_base64,
                time
            )
            VALUES (%s, %s, %s, NOW())
        """, (
            student_id,
            course_id,
            image_base64
        ))
        conn.commit()
        
        return {"success": True}
    except Exception as e:
        if conn:
            conn.rollback()
        raise e
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@attendance_bp.route('/api/courses/<int:course_id>/attendance', methods=['GET'])
def get_attendance(course_id):

    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("""
        SELECT
            a.student_id,
            u.full_name,
            a.time,
            a.recognized,
            a.image_base64
        FROM attendance a
        JOIN users u
            ON a.student_id = u.id
        WHERE a.course_id = %s
        ORDER BY a.time DESC;
    """, (course_id,))

    data = cur.fetchall()

    cur.close()
    conn.close()

    return jsonify(data)

# Thống kê điểm danh
@attendance_bp.route("/api/courses/<int:course_id>/attendance/summary")
def get_attendance_summary(course_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        WITH total_sessions AS (
            SELECT COUNT(DISTINCT DATE(time)) AS session_count
            FROM attendance
            WHERE course_id = %s
        ),
        student_attendance AS (
            SELECT
                cs.student_id,
                u.full_name,
                COUNT(DISTINCT DATE(a.time)) FILTER (WHERE a.recognized = TRUE) AS attended
            FROM course_students cs
            JOIN users u ON u.id = cs.student_id
            LEFT JOIN attendance a ON a.student_id = u.id AND a.course_id = cs.course_id
            WHERE cs.course_id = %s
            GROUP BY cs.student_id, u.full_name
        )
        SELECT 
            sa.student_id,
            sa.full_name,
            sa.attended,
            ts.session_count - sa.attended AS absent
        FROM student_attendance sa, total_sessions ts
        ORDER BY sa.full_name;
    """, (course_id, course_id))
    
    rows = cursor.fetchall()
    conn.close()

    return jsonify([
        {
            "student_id": r[0],
            "full_name": r[1],
            "attended": r[2],
            "absent": r[3]
        } for r in rows
    ])