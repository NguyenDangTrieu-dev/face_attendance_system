from flask import Blueprint, jsonify
from db import get_db_connection
from auth import login_required
from psycopg2.extras import RealDictCursor

lecturer_bp = Blueprint('lecturer', __name__)


@lecturer_bp.route('/api/lecturer/<lecturer_id>/courses')
@login_required([2])
def get_courses(lecturer_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, name, semester FROM courses WHERE lecturer_id=%s", (lecturer_id,))
    rows = cur.fetchall()
    conn.close()

    return jsonify([{'id': r[0], 'name': r[1], 'semester': r[2]} for r in rows])
@lecturer_bp.route("/api/lecturers", methods=["GET"])
def get_lecturers():

    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("""
        SELECT
            u.id,
            u.full_name,
            u.role_id,
            COUNT(c.id) as total_courses
        FROM users u
        LEFT JOIN courses c ON c.lecturer_id = u.id
        WHERE u.role_id = 2
        GROUP BY u.id, u.full_name, u.role_id
        ORDER BY u.full_name
    """)

    lecturers = cur.fetchall()

    cur.close()
    conn.close()

    return jsonify(lecturers)