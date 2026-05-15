from flask import Blueprint, request, jsonify
from psycopg2.extras import RealDictCursor

from db import get_db_connection
from face_rec_SM_api import invalidate_embedding_cache
from auth import role_required

import pandas as pd
import logging

course_bp = Blueprint('course', __name__, url_prefix='/api/courses')

logger = logging.getLogger(__name__)

# =====================================================
# GET COURSES
# =====================================================
@course_bp.route('', methods=['GET'])
@role_required([1, 2])
def get_courses():

    lecturer_id = request.args.get('lecturer_id')

    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    sql = """
        SELECT
            c.id,
            c.name,
            c.semester,
            c.lecturer_id,
            c.room,
            c.period,
            c.credits,
            u.full_name AS lecturer_name
        FROM courses c
        LEFT JOIN users u
            ON c.lecturer_id = u.id
    """
    params = []

    if lecturer_id:
        sql += " WHERE c.lecturer_id = %s "
        params.append(lecturer_id)

    sql += " ORDER BY c.id DESC "

    cur.execute(sql, params)

    data = cur.fetchall()

    conn.close()

    return jsonify(data)


# =====================================================
# CREATE COURSE
# =====================================================
@course_bp.route('', methods=['POST'])
@role_required([1])
def create_course():

    data = request.json

    name = data.get('name')
    semester = data.get('semester')
    lecturer_id = data.get('lecturer_id')

    if not name or not semester or not lecturer_id:
        return jsonify({
            'error': 'Thiếu thông tin'
        }), 400

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO courses
        (
            name,
            semester,
            lecturer_id,
            room,
            period,
            credits
        )
        VALUES (%s,%s,%s,%s,%s,%s)
        RETURNING id
    """, (
        name,
        semester,
        lecturer_id,
        data.get('room'),
        data.get('period'),
        data.get('credits')
    ))

    course_id = cur.fetchone()[0]

    conn.commit()
    conn.close()

    return jsonify({
        'id': course_id,
        'message': 'created'
    })

# =====================================================
# UPDATE COURSE
# =====================================================
@course_bp.route('/<int:course_id>', methods=['PUT'])
@role_required([1])
def update_course(course_id):

    data = request.json

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        UPDATE courses
        SET
            name=%s,
            semester=%s,
            lecturer_id=%s,
            room=%s,
            period=%s,
            credits=%s
        WHERE id=%s
    """, (
        data['name'],
        data['semester'],
        data['lecturer_id'],
        data.get('room'),
        data.get('period'),
        data.get('credits'),
        course_id
    ))

    conn.commit()
    conn.close()

    return jsonify({
        'message': 'updated'
    })

# =====================================================
# DELETE COURSE
# =====================================================
@course_bp.route('/<int:course_id>', methods=['DELETE'])
@role_required([1])
def delete_course(course_id):

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM courses WHERE id=%s",
        (course_id,)
    )
    conn.commit()
    conn.close()

    return jsonify({
        'message': 'deleted'
    })

# =====================================================
# IMPORT STUDENTS
# =====================================================
@course_bp.route('/<int:course_id>/import_students', methods=['POST'])
@role_required([1])
def import_students(course_id):
    file = request.files['file']
    df = pd.read_excel(file)
    conn = get_db_connection()
    cur = conn.cursor()
    for _, row in df.iterrows():
        sid = str(row['student_id'])
        name = row['full_name']
        cur.execute(
            "SELECT id FROM users WHERE id=%s",
            (sid,)
        )
        if not cur.fetchone():
            cur.execute("""
                INSERT INTO users
                (
                    id,
                    full_name,
                    password,
                    role_id
                )
                VALUES (%s,%s,%s,3)
            """, (
                sid,
                name,
                '123456'
            ))
        cur.execute("""
            INSERT INTO course_students
            (
                course_id,
                student_id
            )
            VALUES (%s,%s)
            ON CONFLICT DO NOTHING
        """, (
            course_id,
            sid
        ))
    conn.commit()
    conn.close()
    invalidate_embedding_cache(course_id)
    return jsonify({
        'message': 'imported'
    })

@course_bp.route(
    '/<int:course_id>/students_with_embedding',
    methods=['GET'])
def get_students_with_embedding(course_id):

    conn = get_db_connection()

    cur = conn.cursor(
        cursor_factory=RealDictCursor
    )

    cur.execute("""
        SELECT
            u.id,
            u.full_name,

            CASE
                WHEN e.user_id IS NOT NULL
                THEN true
                ELSE false
            END AS has_embedding

        FROM users u

        JOIN course_students cs
            ON cs.student_id = u.id

        LEFT JOIN embeddings e
            ON e.user_id = u.id

        WHERE cs.course_id = %s

        ORDER BY u.full_name
    """, (course_id,))

    students = cur.fetchall()

    cur.close()
    conn.close()

    return jsonify(students)

# =========================
# GET STUDENTS IN COURSE
# =========================
@course_bp.route('/<int:course_id>/students', methods=['GET'])
def get_students_in_course(course_id):

    try:

        conn = get_db_connection()

        cur = conn.cursor(
            cursor_factory=RealDictCursor
        )

        cur.execute("""
            SELECT
                u.id,
                u.full_name,
                u.role_id

            FROM course_students cs

            JOIN users u
                ON u.id = cs.student_id

            WHERE cs.course_id = %s

            ORDER BY u.full_name
        """, (course_id,))

        students = cur.fetchall()

        cur.close()
        conn.close()

        return jsonify(students)

    except Exception as e:

        print("GET STUDENTS ERROR:", e)

        return jsonify({
            "error": str(e)
        }), 500