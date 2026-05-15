from flask import Blueprint, jsonify, request
from psycopg2.extras import RealDictCursor

from db import get_db_connection
from auth import role_required

admin_bp = Blueprint('admin', __name__, url_prefix='/api/admin')


# ================= USERS =================
@admin_bp.route("/users", methods=["GET"])
@role_required([1])
def get_users():
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("SELECT id, full_name, role_id FROM users ORDER BY id")
    users = cur.fetchall()

    conn.close()
    return jsonify(users)


@admin_bp.route("/users", methods=["POST"])
@role_required([1])
def create_user():
    data = request.json
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO users (id, full_name, password, role_id)
        VALUES (%s, %s, %s, %s)
    """, (
        data["id"],
        data["full_name"],
        data.get("password", "123456"),
        data["role_id"]
    ))

    conn.commit()
    conn.close()

    return jsonify({"message": "User created"})


# ================= COURSES =================
@admin_bp.route("/courses", methods=["GET"])
@role_required([1])
def get_courses():
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("""
        SELECT
            c.id,
            c.name,
            c.semester,
            c.lecturer_id,
            u.full_name AS lecturer_name
        FROM courses c
        LEFT JOIN users u
            ON c.lecturer_id = u.id
        ORDER BY c.id DESC
    """)
    data = cur.fetchall()

    conn.close()
    return jsonify(data)

@admin_bp.route("/attendance/recent", methods=["GET"])
@role_required([1])
def get_recent_attendance():

    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("""
        SELECT *
        FROM attendance
        ORDER BY time DESC
        LIMIT 100
    """)

    data = cur.fetchall()

    conn.close()

    return jsonify(data)

# ================= DELETE USER =================
@admin_bp.route("/users/<user_id>", methods=["DELETE"])
@role_required([1])
def delete_user(user_id):

    try:

        conn = get_db_connection()
        cur = conn.cursor()

        # xóa liên kết course trước
        cur.execute("""
            DELETE FROM course_students
            WHERE student_id = %s
        """, (user_id,))

        # xóa embedding
        cur.execute("""
            DELETE FROM embeddings
            WHERE user_id = %s
        """, (user_id,))

        # xóa user
        cur.execute("""
            DELETE FROM users
            WHERE id = %s
        """, (user_id,))

        conn.commit()

        cur.close()
        conn.close()

        return jsonify({
            "message": "Đã xóa người dùng"
        })

    except Exception as e:

        print("DELETE USER ERROR:", e)

        return jsonify({
            "error": str(e)
        }), 500
        
# ================= UPDATE USER =================
@admin_bp.route("/users/<user_id>", methods=["PUT"])
@role_required([1])
def update_user(user_id):

    try:

        data = request.json

        conn = get_db_connection()
        cur = conn.cursor()

        # 1. update user
        cur.execute("""
            UPDATE users
            SET full_name = %s,
                role_id = %s
            WHERE id = %s
        """, (
            data["full_name"],
            data["role_id"],
            user_id
        ))

        # 2. update password nếu có
        if data.get("password"):
            cur.execute("""
                UPDATE users
                SET password = %s
                WHERE id = %s
            """, (
                data["password"],
                user_id
            ))

        # 3. 🚨 nếu không còn là sinh viên → xóa khỏi tất cả học phần
        if int(data["role_id"]) != 3:

            cur.execute("""
                DELETE FROM course_students
                WHERE student_id = %s
            """, (user_id,))

        conn.commit()
        cur.close()
        conn.close()

        return jsonify({"message": "Đã cập nhật"})

    except Exception as e:

        print("UPDATE USER ERROR:", e)

        return jsonify({
            "error": str(e)
        }), 500