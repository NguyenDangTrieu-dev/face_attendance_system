from dotenv import load_dotenv
load_dotenv()

# Built-in & Standard Libraries
import os
import io
import socket
import logging
from functools import wraps
from datetime import datetime, timedelta
from io import BytesIO
import base64
# Third-party Libraries
import cv2
import jwt
import numpy as np
import pandas as pd
import qrcode
import psycopg2
import openpyxl
from flask import Flask, json, request, jsonify, render_template, redirect, url_for, send_file, Response
from werkzeug.utils import secure_filename
from psycopg2.extras import RealDictCursor

# Local Imports
from register import register_face
from face_rec_SM_api import recognize_face, invalidate_embedding_cache
from realtime_engine import RealtimeEngine
from src.api.student_import import import_from_zip

app = Flask(__name__)

# Realtime engine singleton
realtime_engine = RealtimeEngine(camera_src=int(os.getenv('CAMERA_SRC', '0')))

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Config
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

DB_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': os.getenv('POSTGRES_PORT', 5432),
    'database': os.getenv('POSTGRES_DB', 'face_db'),
    'user': os.getenv('POSTGRES_USER', 'postgres'),
    'password': os.getenv('POSTGRES_PASSWORD', '123456'),
}

SECRET_KEY = os.getenv('SECRET_KEY')

def role_required(allowed_roles):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            token = request.headers.get('Authorization', '').replace('Bearer ', '')
            try:
                data = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
                if data['role_id'] not in allowed_roles:
                    return jsonify({'error': 'Không có quyền truy cập'}), 403
                return func(*args, **kwargs)
            except jwt.ExpiredSignatureError:
                return jsonify({'error': 'Token hết hạn'}), 401
            except Exception:
                return jsonify({'error': 'Token không hợp lệ'}), 401
        return wrapper
    return decorator

def get_db_connection():
    return psycopg2.connect(
        host=DB_CONFIG['host'],
        port=DB_CONFIG['port'],
        dbname=DB_CONFIG['database'],
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password']
    )

# Giao diện chính
def login_required(roles=[]):
    def wrapper(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            token = request.cookies.get('token')
            if not token:
                return redirect(url_for('login'))
            try:
                data = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
                if roles and data['role_id'] not in roles:
                    return "Không có quyền truy cập", 403
                request.user = data
            except Exception:
                return redirect(url_for('login'))
            return f(*args, **kwargs)
        return decorated
    return wrapper

@app.route('/')
def index():
    return redirect('/login')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_id = request.form['id']
        password = request.form['password']
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM users WHERE id = %s AND password = %s", (user_id, password))
        user = cur.fetchone()
        conn.close()

        if user:
            token = jwt.encode({
                'user_id': user['id'],
                'role_id': user['role_id'],
                'full_name': user['full_name'],
                'exp': datetime.utcnow() + timedelta(hours=2)
            }, SECRET_KEY, algorithm='HS256')
            resp = redirect('/redirect-by-role')
            resp.set_cookie('token', token)
            return resp
        else:
            return render_template('login.html', error='Sai tài khoản hoặc mật khẩu')
    return render_template('login.html')

@app.route('/redirect-by-role')
def redirect_by_role():
    token = request.cookies.get('token')
    if not token:
        return redirect('/login')
    try:
        data = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        role_id = data['role_id']
        if role_id == 1:
            return redirect('/admin/dashboard')
        elif role_id == 2:
            return redirect('/lecturer/dashboard')
        elif role_id == 3:
            return redirect('/student/dashboard')
    except Exception:
        return redirect('/login')

@app.route('/logout')
def logout():
    resp = redirect('/login')
    resp.set_cookie('token', '', expires=0)
    return resp

@app.route('/admin/dashboard')
@login_required([1])
def admin_dashboard():
    return render_template('dashboard_admin.html', user=request.user)

@app.route('/lecturer/dashboard')
@login_required([2])
def lecturer_dashboard():
    return render_template('dashboard_lecturer.html', user=request.user)

@app.route('/student/dashboard')
@login_required([3])
def student_dashboard():
    return render_template('dashboard_student.html', user=request.user)

# Đăng ký sinh viên mới
@app.route("/register", methods=["POST"])
def register():
    student_id = request.form.get("student_id")
    name = request.form.get("name")
    files = request.files.getlist("files")

    if not student_id or not name or len(files) != 10:
        return jsonify({"success": False, "message": "Thiếu dữ liệu đăng ký"}), 400

    frames = []
    temp_files = []

    try:
        for f in files:
            if not allowed_file(f.filename):
                return jsonify({"success": False, "message": "Sai định dạng ảnh"}), 400

            path = os.path.join(UPLOAD_FOLDER, secure_filename(f.filename))
            f.save(path)
            temp_files.append(path)

            img = cv2.imread(path)
            if img is None:
                return jsonify({"success": False, "message": "Không đọc được ảnh"}), 400

            frames.append(img)

        result = register_face(student_id, name, frames)
        invalidate_embedding_cache()

        return jsonify({"success": True, "message": "Đăng ký thành công!"})

    finally:
        for p in temp_files:
            if os.path.exists(p):
                os.remove(p)

# Thông tin nhận diện
@app.route('/api/attendance', methods=['GET'])
def get_attendance():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        cursor.execute("""
            SELECT
            a.student_id,
            u.full_name,
            encode(a.image, 'base64') AS image_base64,
            a.time
            FROM attendance a
            JOIN users u ON a.student_id = u.id
            ORDER BY a.time DESC;

        """)
        attendance_data = cursor.fetchall()

        cursor.close()
        conn.close()

        return jsonify(attendance_data), 200

    except Exception as e:
        app.logger.error(f"Lỗi truy vấn DB PostgreSQL: {e}")
        return jsonify({'error': 'Không thể truy vấn dữ liệu điểm danh'}), 500

# Xóa sinh viên điểm danh
@app.route('/api/delete/<student_id>', methods=['DELETE'])
def delete_student_attendance(student_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM attendance WHERE student_id = %s", (student_id,))
        conn.commit()

        cursor.close()
        conn.close()

        return jsonify({"message": f"Đã xóa dữ liệu điểm danh của sinh viên {student_id}!"}), 200

    except Exception as e:
        app.logger.error(f"Lỗi khi xóa dữ liệu attendance: {e}")
        return jsonify({"error": "Không thể xóa dữ liệu"}), 500

# Route Giảng Viên
@app.route("/api/lecturer/<lecturer_id>/courses")
@login_required([2]) 
def get_lecturer_courses(lecturer_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, name, semester 
        FROM courses 
        WHERE lecturer_id = %s
    """, (lecturer_id,))
    rows = cursor.fetchall()
    conn.close()
    return jsonify([{"id": r[0], "name": r[1], "semester": r[2]} for r in rows])

# Lấy danh sách sinh viên trong khóa học
@app.route("/api/courses/<int:course_id>/students")
@login_required([2])
def get_course_students(course_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT u.id, u.full_name
        FROM users u
        JOIN course_students cs ON u.id = cs.student_id
        WHERE cs.course_id = %s
    """, (course_id,))
    rows = cursor.fetchall()
    conn.close()
    return jsonify([{"id": r[0], "full_name": r[1]} for r in rows])

# Lấy danh sách điểm danh của giảng viên (HOÀN THIỆN PHẦN TRUNCATED)
@app.route("/api/lecturer/<lecturer_id>/attendance")
@login_required([2])
def get_attendance_by_lecturer(lecturer_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT a.course_id, c.name AS course_name, u.full_name AS student_name, a.time, a.image_base64, a.recognized
        FROM attendance a
        JOIN courses c ON a.course_id = c.id
        JOIN users u ON a.student_id = u.id
        WHERE c.lecturer_id = %s
        ORDER BY a.time DESC
    """, (lecturer_id,))
    rows = cursor.fetchall()
    conn.close()
    attendance_list = [
        {
            "course_id": r[0],
            "course_name": r[1],
            "student_name": r[2],
            "time": r[3],
            "image": f"data:image/jpeg;base64,{r[4]}" if r[4] else None,
            "recognized": r[5]
        } for r in rows
    ]
    return jsonify(attendance_list)
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
def save_attendance(student_id, course_id, image_base64):
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO attendance (
                    student_id,
                    course_id,
                    time,
                    image_base64,
                    recognized
                )
                VALUES (%s, %s, NOW(), %s, TRUE)
            """, (student_id, course_id, image_base64))
        conn.commit()
    finally:
        conn.close()

@app.route("/recognize_siamese", methods=["POST"])
def recognize_siamese():
    if "file" not in request.files or "course_id" not in request.form:
        return jsonify({"error": "Thiếu hình ảnh hoặc course_id"}), 400

    file = request.files["file"]
    course_id = int(request.form["course_id"])

    file_bytes = file.read()
    if len(file_bytes) == 0:
        return jsonify({"error": "File rỗng"}), 400

    if len(file_bytes) > 5 * 1024 * 1024:
        return jsonify({"error": "Ảnh vượt quá 5MB"}), 400

    frame = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Không decode được ảnh"}), 400

    try:
        results = recognize_face(frame, course_id)

        recognized = []
        skipped = []

        h, w = frame.shape[:2]

        for r in results:
            if r["similarity"] < 0.75:
                skipped.append({
                    "student_id": r["student_id"],
                    "reason": "low_similarity"
                })
                continue

            student_id = r["student_id"]

            if already_attended_today(student_id, course_id):
                skipped.append({
                    "student_id": student_id,
                    "reason": "already_attended"
                })
                continue

            # clamp bbox
            x1, y1, x2, y2 = map(int, r["bbox"])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face_crop = frame[y1:y2, x1:x2]
            image_base64 = None

            if face_crop.size > 0:
                _, buf = cv2.imencode(".jpg", face_crop)
                image_base64 = base64.b64encode(buf).decode()

            save_attendance(student_id, course_id, image_base64)

            recognized.append({
                "student_id": student_id,
                "name": r["name"],
                "similarity": r["similarity"]
            })

        return jsonify({
            "status": "ok",
            "recognized": recognized,
            "skipped": skipped
        })

    except Exception as e:
        logger.exception("recognize_siamese error")
        return jsonify({"error": str(e)}), 500

# Có dữ liệu khuôn mặt hay không?
@app.route("/api/courses/<int:course_id>/students_with_embedding")
@login_required([2])
def get_students_with_embedding(course_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT u.id, u.full_name,
               EXISTS (
                   SELECT 1 FROM embeddings e WHERE e.user_id = u.id
               ) AS has_embedding
        FROM users u
        JOIN course_students cs ON u.id = cs.student_id
        WHERE cs.course_id = %s
    """, (course_id,))

    rows = cursor.fetchall()
    conn.close()

    students = [
        {
            "id": r[0],
            "full_name": r[1],
            "has_embedding": r[2]
        }
        for r in rows
    ]

    return jsonify(students)

# Tạo QR Code cho IP
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

@app.route('/api/get_ip', methods=['GET'])
def get_ip():
    try:
        ip = get_local_ip()
        return jsonify({'ip': ip}), 200
    except Exception as e:
        logger.error(f"Error getting local IP: {str(e)}")
        return jsonify({'error': 'Không thể lấy địa chỉ IP'}), 500

@app.route('/api/qr_ip', methods=['GET'])
def generate_qr_ip():
    try:
        ip = get_local_ip()
        course_id = request.args.get('course_id', '')
        course_name = request.args.get('course_name', '')

        data = {
            'ip': ip,
            'course_id': course_id,
            'course_name': course_name
        }
        qr_content = json.dumps(data, ensure_ascii=False)

        qr = qrcode.make(qr_content)
        img_io = io.BytesIO()
        qr.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        logger.error(f"Error creating QR: {str(e)}")
        return jsonify({'error': 'Không thể tạo mã QR'}), 500

# CRUD Khóa Học
@app.route('/api/courses', methods=['POST'])
def create_course():
    try:
        data = request.json
        name = data.get('name')
        semester = data.get('semester')
        lecturer_id = data.get('lecturer_id')
        if not name or not semester or not lecturer_id:
            return jsonify({'error': 'Thiếu thông tin cần thiết'}), 400
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO courses (name, semester, lecturer_id) VALUES (%s, %s, %s) RETURNING id",
                       (name, semester, lecturer_id))
        course_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({'id': course_id, 'name': name, 'semester': semester, 'lecturer_id': lecturer_id}), 201
    except Exception as e:
        logger.error(f"Lỗi khi tạo khóa học: {str(e)}")
        return jsonify({'error': 'Không thể tạo khóa học'}), 500
    
@app.route('/api/courses/<int:course_id>', methods=['PUT'])
def update_course(course_id):
    data = request.json
    name = data.get('name')
    semester = data.get('semester')
    lecturer_id = data.get('lecturer_id')
    if not name or not semester or not lecturer_id:
        return jsonify({'error': 'Thiếu thông tin cần thiết'}), 400
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE courses 
            SET name = %s, semester = %s, lecturer_id = %s 
            WHERE id = %s
        """, (name, semester, lecturer_id, course_id))
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({'message': 'Khóa học đã được cập nhật thành công'}), 200
    except Exception as e:
        logger.error(f"Lỗi khi cập nhật khóa học: {str(e)}")
        return jsonify({'error': 'Không thể cập nhật khóa học'}), 500
    
@app.route('/api/courses/<int:course_id>', methods=['DELETE'])
def delete_course(course_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM courses WHERE id = %s", (course_id,))
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({'message': 'Khóa học đã được xóa thành công'}), 200
    except Exception as e:
        logger.error(f"Lỗi khi xóa khóa học: {str(e)}")
        return jsonify({'error': 'Không thể xóa khóa học'}), 500

# Import data student from excel
@app.route("/api/courses/<int:course_id>/import_students", methods=["POST"])
def import_students_to_course(course_id):
    logger.info(f"Received request to import students for course_id: {course_id}")
    if "file" not in request.files:
        logger.error("No file part in request")
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        logger.error("No selected file")
        return jsonify({"error": "No selected file"}), 400
    filename = secure_filename(file.filename)
    logger.info(f"Processing file: {filename}")
    try:
        df = pd.read_excel(file)
        logger.info(f"Excel columns: {df.columns.tolist()}")
    except Exception as e:
        logger.error(f"Error reading Excel file: {str(e)}")
        return jsonify({"error": f"Error reading Excel file: {str(e)}"}), 400
    required_columns = {"student_id", "full_name"}
    if not required_columns.issubset(df.columns):
        logger.error(f"Missing required columns. Found: {df.columns.tolist()}")
        return jsonify({"error": f"Excel file must contain columns: {required_columns}"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()

    for _, row in df.iterrows():
        student_id = str(row["student_id"])
        full_name = row["full_name"]

        # Check if user exists
        cursor.execute("SELECT id FROM users WHERE id = %s", (student_id,))
        user_exists = cursor.fetchone()

        if not user_exists:
            cursor.execute("""
                INSERT INTO users (id, full_name, password, role_id)
                VALUES (%s, %s, %s, %s)
            """, (student_id, full_name, '123456', 3))  # Note: Nên hash password ở đây (ví dụ: bcrypt)

        cursor.execute("""
            SELECT 1 FROM course_students WHERE course_id = %s AND student_id = %s
        """, (course_id, student_id))
        in_course = cursor.fetchone()

        if not in_course:
            cursor.execute("""
                INSERT INTO course_students (course_id, student_id)
                VALUES (%s, %s)
            """, (course_id, student_id))

    conn.commit()
    conn.close()

    invalidate_embedding_cache(course_id)

    logger.info("Students imported successfully")
    return jsonify({"message": "Students imported successfully!"})

@app.route("/api/courses/<int:course_id>/attendance", methods=["GET"])
@login_required([2])
def get_course_attendance(course_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        cursor.execute("""
            SELECT
                a.student_id,
                u.full_name AS name,
                a.image_base64,
                a.time,
                a.recognized
            FROM attendance a
            JOIN users u ON u.id = a.student_id
            WHERE a.course_id = %s
            ORDER BY a.time DESC;
        """, (course_id,))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        return jsonify(rows), 200

    except Exception as e:
        logger.error(f"Lỗi lấy attendance course {course_id}: {e}")
        return jsonify({"error": "Không lấy được dữ liệu điểm danh"}), 500

# Thống kê điểm danh
@app.route("/api/courses/<int:course_id>/attendance/summary")
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

@app.route("/api/realtime/start", methods=["POST"])
def realtime_start():
    data = request.get_json() or {}
    course_id = data.get("course_id")
    camera_src = data.get("camera_src")

    if course_id is None:
        return jsonify({"error": "Thiếu course_id"}), 400

    realtime_engine.start(
        course_id=int(course_id),
        camera_src=int(camera_src) if camera_src is not None else None
    )
    return jsonify({"running": True})

@app.route("/api/realtime/stop", methods=["POST"])
def realtime_stop():
    realtime_engine.stop()
    return jsonify({"running": False})

@app.route("/api/realtime/status")
def realtime_status():
    return jsonify({
        "running": realtime_engine.is_running(),
        "results": realtime_engine.get_last_results()
    })

# Video stream
def mjpeg_generator():
    while True:
        frame = realtime_engine.get_latest_jpeg()
        if frame is None:
            import time
            time.sleep(0.02)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            frame +
            b"\r\n"
        )

@app.route("/video_feed")
def video_feed():
    return Response(
        mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )
@app.route("/api/students/register/camera", methods=["POST"])
@login_required([2])  # Chỉ giảng viên mới được thêm sinh viên
def register_student_camera():
    """
    API để THÊM MỚI sinh viên vào hệ thống bằng camera
    - Tạo user mới trong bảng users
    - Lưu ảnh vào thư mục
    - Tạo embedding khuôn mặt
    """
    student_id = request.form.get("student_id")
    name = request.form.get("name")
    images = request.files.getlist("images")

    if not student_id or not name:
        return jsonify({"error": "Thiếu MSSV hoặc họ tên"}), 400

    if not images or len(images) < 5:
        return jsonify({"error": "Cần ít nhất 5 ảnh"}), 400

    from student_camera import register_from_uploaded_images
    result, status = register_from_uploaded_images(student_id, name, images)

    return jsonify(result), status

@app.route("/api/students/register/zip", methods=["POST"])
def register_student_zip():
    zip_file = request.files.get("zip")
    if not zip_file:
        return jsonify({"error": "Chưa upload file ZIP"}), 400

    results = import_from_zip(zip_file)
    return jsonify(results), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)