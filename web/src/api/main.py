from dotenv import load_dotenv
load_dotenv()

# ==== Built-in & Standard Libraries ====
import os
import io
import socket
import logging
from functools import wraps
from datetime import datetime, timedelta
from io import BytesIO

# ==== Third-party Libraries ====
import cv2
import jwt
import numpy as np
import pandas as pd
import qrcode
import psycopg2
import openpyxl
from flask import Flask, json, request, jsonify, render_template, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from psycopg2.extras import RealDictCursor

# ==== Local Imports (Your own modules) ====
from register import register_face
from face_rec_SM_api import recognize_face

app = Flask(__name__)

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cấu hình
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Đảm bảo thư mục upload tồn tại
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# Đọc biến môi trường từ .env hoặc cấu hình trực tiếp
DB_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': os.getenv('POSTGRES_PORT', 5432),
    'database': os.getenv('POSTGRES_DB', 'face_db'),
    'user': os.getenv('POSTGRES_USER', 'postgres'),
    'password': os.getenv('POSTGRES_PASSWORD', '123456'),
}

SECRET_KEY= os.getenv('SECRET_KEY')

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
    """Tạo kết nối đến PostgreSQL."""
    return psycopg2.connect(
        host=DB_CONFIG['host'],
        port=DB_CONFIG['port'],
        dbname=DB_CONFIG['database'],
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password']
    )
#---------------------------------GIAO DIỆN CHÍNH-----------------------------------

#--------------------------
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


#---------------------------------ĐĂNG KÍ SINH VIENEJJ MỚI-----------------------------------
@app.route('/register', methods=['POST'])
def register():
    if 'student_id' not in request.form or 'name' not in request.form:
        logger.warning("Missing student_id or name in registration request")
        return jsonify({'success': False, 'message': 'Thiếu mã số sinh viên hoặc họ tên'}), 400
    
    student_id = request.form['student_id']
    name = request.form['name']

    if len(student_id) != 10 or not student_id.isdigit():
        logger.warning(f"Invalid student_id format: {student_id}")
        return jsonify({'success': False, 'message': 'Mã số sinh viên phải gồm 10 chữ số'}), 400

    if 'files' not in request.files:
        logger.warning("No files uploaded in registration request")
        return jsonify({'success': False, 'message': 'Không có hình ảnh được gửi. Vui lòng gửi đúng 10 ảnh.'}), 400

    files = request.files.getlist('files')
    if len(files) != 10:
        logger.warning(f"Expected 10 files, but received {len(files)}")
        return jsonify({'success': False, 'message': 'Vui lòng gửi đúng 10 ảnh để đăng ký.'}), 400

    frames = []
    temp_filepaths = []
    try:
        for file in files:
            if file.filename == '':
                logger.warning("No file selected in registration request")
                return jsonify({'success': False, 'message': 'Một trong các hình ảnh không được chọn. Vui lòng thử lại.'}), 400

            if not allowed_file(file.filename):
                logger.warning(f"Invalid file type in registration request: {file.filename}")
                return jsonify({'success': False, 'message': 'Định dạng hình ảnh không hợp lệ. Chỉ hỗ trợ PNG, JPG, JPEG.'}), 400

            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            if file_size > MAX_FILE_SIZE:
                logger.warning(f"File too large in registration request: {file_size} bytes")
                return jsonify({'success': False, 'message': 'Hình ảnh quá lớn. Kích thước tối đa là 5MB.'}), 400
            file.seek(0)

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            logger.info(f"Saving file to {filepath}")
            file.save(filepath)
            temp_filepaths.append(filepath)

            frame = cv2.imread(filepath)
            if frame is None:
                logger.error(f"Failed to read image from {filepath}")
                with open(filepath, 'rb') as f:
                    img_data = f.read()
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    return jsonify({'success': False, 'message': 'Không thể đọc hình ảnh. Vui lòng thử lại với hình ảnh hợp lệ.'}), 400
                cv2.imwrite(filepath, frame)

            frames.append(frame)

        logger.info(f"Registering face for student_id: {student_id}, name: {name}")
        result = register_face(student_id, name, frames)
        if isinstance(result, tuple):
            result, status_code = result
            if 'error' in result:
                return jsonify({'success': False, 'message': result['error']}), status_code
            return jsonify({'success': True, 'message': result.get('message', 'Đăng ký thành công!')}), status_code
        if 'error' in result:
            return jsonify({'success': False, 'message': result['error']}), 400
        return jsonify({'success': True, 'message': result.get('message', 'Đăng ký thành công!')}), 200

    except Exception as e:
        logger.error(f"Error during face registration: {str(e)}")
        return jsonify({'success': False, 'message': f'Đã xảy ra lỗi khi đăng ký: {str(e)}'}), 500
    finally:
        for filepath in temp_filepaths:
            if os.path.exists(filepath):
                logger.info(f"Removing temporary file: {filepath}")
                os.remove(filepath)

#---------------------------------NHẬN DIỆN SVM-----------------------------------
#đã xóa

#---------------------------------NHẬN DIỆN SIAMESE NETWORK-----------------------------------
current_course_id = None

@app.route('/start_attendance', methods=['POST'])
def start_attendance():
    global current_course_id
    course_id = request.json.get('course_id')
    if not course_id:
        return jsonify({'error': 'Thiếu mã khóa học'}), 400
    current_course_id = course_id
    logger.info(f"✅ Bắt đầu điểm danh cho khóa {course_id}")
    return jsonify({'message': f'Bắt đầu điểm danh cho khóa {course_id}'}), 200


@app.route('/recognize_siamese', methods=['POST'])
def recognize_siamese():
    if 'file' not in request.files or 'course_id' not in request.form:
        logger.warning("Thiếu file hoặc course_id")
        return jsonify({'error': 'Thiếu hình ảnh hoặc mã khóa học.'}), 400

    file = request.files['file']
    course_id = request.form['course_id']

    if file.filename == '':
        logger.warning("No file selected")
        return jsonify({'error': 'Không có hình ảnh được chọn. Vui lòng thử lại.'}), 400

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    if file_size > MAX_FILE_SIZE:
        logger.warning(f"File quá lớn: {file_size} bytes")
        return jsonify({'error': 'Hình ảnh quá lớn. Kích thước tối đa là 5MB.'}), 400
    file.seek(0)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            frame = cv2.imread(filepath)
            if frame is None:
                logger.error("Không đọc được hình ảnh")
                return jsonify({'error': 'Không thể đọc hình ảnh. Vui lòng thử lại.'}), 400

            # Log image dimensions
            height, width, _ = frame.shape
            logger.info(f"Image dimensions: {width}x{height}")

            # Save the image for debugging
            debug_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"debug_{filename}")
            cv2.imwrite(debug_filepath, frame)
            logger.info(f"Saved debug image to {debug_filepath}")

            logger.info("⏳ Đang nhận diện bằng Siamese Network...")
            results = recognize_face(frame, course_id)
            return jsonify(results)

        except Exception as e:
            logger.error(f"Lỗi khi nhận diện: {str(e)}")
            return jsonify({'error': f'Đã xảy ra lỗi khi nhận diện: {str(e)}'}), 500

        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    logger.warning("Sai định dạng file")
    return jsonify({'error': 'Định dạng hình ảnh không hợp lệ. Chỉ hỗ trợ PNG, JPG, JPEG.'}), 400

#--------------------- Điểm Danh Bang Ảnh ------------------------
MAX_FILE_SIZE = 5 * 1024 * 1024  

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@app.route('/recognize_siamese_photos', methods=['POST'])
def recognize_siamese_photos():
    if 'files' not in request.files or 'course_id' not in request.form:
        logger.warning("Thiếu files hoặc course_id")
        return jsonify({'error': 'Thiếu hình ảnh hoặc mã khóa học.'}), 400

    files = request.files.getlist('files')
    course_id = request.form['course_id']

    if not files or len(files) == 0:
        logger.warning("Không có file nào được chọn")
        return jsonify({'error': 'Không có hình ảnh được chọn. Vui lòng thử lại.'}), 400

    results = []
    temp_files = []

    try:
        for file in files:
            if not allowed_file(file.filename):
                logger.warning(f"Sai định dạng file: {file.filename}")
                continue

            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            if file_size > MAX_FILE_SIZE:
                logger.warning(f"File quá lớn: {file_size} bytes")
                continue
            file.seek(0)

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            temp_files.append(filepath)

            frame = cv2.imread(filepath)
            if frame is None:
                logger.error(f"Không đọc được hình ảnh: {filename}")
                continue

            height, width, _ = frame.shape
            logger.info(f"Image dimensions: {width}x{height}")

            # Lưu ảnh debug
            debug_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"debug_{filename}")
            cv2.imwrite(debug_filepath, frame)
            logger.info(f"Saved debug image to {debug_filepath}")

            logger.info(f"⏳ Đang nhận diện file {filename}...")
            recognition_results = recognize_face(frame, course_id)
            if isinstance(recognition_results, list):
                results.extend(recognition_results)

        # Lọc kết quả trùng lặp dựa trên tên
        unique_results = []
        seen_names = set()
        for result in results:
            name = result.get('name')
            if name and name != "Unknown" and name not in seen_names:
                unique_results.append(result)
                seen_names.add(name)
            elif name == "Unknown":
                unique_results.append(result)  # Vẫn giữ lại các kết quả Unknown

        logger.info(f"Found {len(unique_results)} unique faces")
        return jsonify(unique_results)

    except Exception as e:
        logger.error(f"Lỗi khi nhận diện: {str(e)}")
        return jsonify({'error': f'Đã xảy ra lỗi khi nhận diện: {str(e)}'}), 500

    finally:
        for filepath in temp_files:
            if os.path.exists(filepath):
                os.remove(filepath)
#---------------------------------THÔNG TIN NHẬN DIỆN-----------------------------------
@app.route('/api/attendance', methods=['GET'])
def get_attendance():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        cursor.execute("""
            SELECT a.student_id, u.full_name AS name, u.image_base64, a.timestamp
            FROM attendance a
            JOIN users u ON a.student_id = u.id
            ORDER BY a.timestamp DESC;
        """)
        attendance_data = cursor.fetchall()

        cursor.close()
        conn.close()

        return jsonify(attendance_data), 200

    except Exception as e:
        app.logger.error(f"Lỗi truy vấn DB PostgreSQL: {e}")
        return jsonify({'error': 'Không thể truy vấn dữ liệu điểm danh'}), 500



#---------------------------------DANH SÁCH LỚP ĐÃ ĐĂNG KÍ-----------------------------------
#đã Xóa
#---------------------------------XÓA SINH VIÊN TRONG DANH SÁCH-----------------------------------
#đã xóa
#---------------------------------XÓA SINH VIÊN ĐIỂM DANH-----------------------------------       
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
#---------------------Route Giảng Viên-------------------------
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
    return jsonify([{"id": r[0], "name": r[1], "semester": r[2]} for r in rows])

#-----------------------LẤY DANH SÁCH SINH VIÊN TRONG MỘT KHÓA HỌC-----------------------------------
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
    return jsonify([{"id": r[0], "full_name": r[1]} for r in rows])

#---------------------------------LẤY DANH SÁCH ĐIỂM DANH CỦA GIẢNG VIÊN-----------------------------------
@app.route("/api/lecturer/<lecturer_id>/attendance")
@login_required([2])
def get_attendance_by_lecturer(lecturer_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT a.course_id, c.name AS course_name, u.full_name AS student_name, a.time, a.recognized
        FROM attendance a
        JOIN courses c ON a.course_id = c.id
        JOIN users u ON a.student_id = u.id
        WHERE c.lecturer_id = %s
        ORDER BY a.time DESC
    """, (lecturer_id,))
    rows = cursor.fetchall()
    return jsonify([
        {
            "course_id": r[0],
            "course_name": r[1],
            "student_name": r[2],
            "time": r[3],
            "recognized": r[4]
        } for r in rows
    ])
#--------danh sách điểm danh của sinh viên trong khóa học-----
import base64
import os
@app.route("/api/courses/<int:course_id>/attendance")
def get_attendance_by_course(course_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT DISTINCT ON (a.student_id) 
        a.id, u.full_name AS student_name, a.time, a.recognized, a.image_base64
        FROM attendance a
        JOIN users u ON a.student_id = u.id
        WHERE a.course_id = %s
        ORDER BY a.student_id, a.time DESC
    """, (course_id,))

    rows = cursor.fetchall()
    conn.close()

    attendance_list = [
        {
            "attendance_id": r[0],
            "student_name": r[1],
            "time": r[2],
            "recognized": r[3],
            "image_base64": f"data:image/jpeg;base64,{r[4]}" if r[4] else None
        } for r in rows
    ]

    return jsonify(attendance_list)

#----------có dữ liệu khuôn mặt hay không?-------------------
@app.route("/api/courses/<int:course_id>/students_with_embedding")
@login_required([2])  # Chỉ giảng viên được phép
def get_students_with_embedding(course_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    # Truy vấn sinh viên và kiểm tra có embedding hay chưa
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
#--------------------------------- Tạo QR Code cho việc lấy ip điểm danh từ server -----------------------------------


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Tạo kết nối giả ra ngoài để lấy IP LAN thực tế
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

        # Gộp thông tin vào JSON
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
#---------------------------------CRUD  Khóa Học-----------------------------------
@app.route('/api/courses', methods = ['POST'])
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
    
@app.route('/api/courses/<int:course_id>', methods = ['DELETE'])
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

#---------------------------------import data student from excel to courses-----------------------------------


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
            """, (student_id, full_name, '123456', 3))

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

    logger.info("Students imported successfully")
    return jsonify({"message": "Students imported successfully!"})

#---------------------------------Thống Kê Điểm Danh-----------------------------------
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
#---------------------------------CHẠY CHƯƠNG TRÌNH-----------------------------------        
if __name__ == '__main__':
    if os.getenv('FLASK_ENV') == 'development':
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        app.run(host='0.0.0.0', port=5000)
# Chạy ứng dụng Flask