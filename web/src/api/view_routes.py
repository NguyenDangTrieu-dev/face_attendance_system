from functools import wraps
import io
import logging
import socket

from flask import Blueprint, json, jsonify, render_template, redirect, send_file, session, url_for, request
from datetime import datetime, timedelta

import jwt
import qrcode
from db import get_db_connection
from config import Config
from psycopg2.extras import RealDictCursor
from auth import login_required

view_bp = Blueprint('view', __name__)

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@view_bp.route('/')
def index():
    return redirect('/login')

@view_bp.route('/logout')
def logout():
    resp = redirect('/login')
    resp.set_cookie('token', '', expires=0)
    return resp

@view_bp.route('/login', methods=['GET', 'POST'])
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
            }, Config.SECRET_KEY, algorithm='HS256')
            resp = redirect('/redirect-by-role')
            resp.set_cookie('token', token)
            return resp
        else:
            return render_template('login.html', error='Sai tài khoản hoặc mật khẩu')
    return render_template('login.html')
    
@view_bp.route('/admin/dashboard')
@login_required([1])
def admin_dashboard():
    return render_template('dashboard_admin.html', user=request.user)

@view_bp.route('/lecturer/dashboard')
@login_required([2])
def lecturer_dashboard():
    return render_template('dashboard_lecturer.html', user=request.user)

@view_bp.route('/student/dashboard')
@login_required([3])
def student_dashboard():
    return render_template('dashboard_student.html', user=request.user)

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
@view_bp.route('/api/qr_ip', methods=['GET'])
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