import jwt
from functools import wraps
from flask import Blueprint, request, jsonify, redirect, session, url_for
from config import Config

auth = Blueprint('auth', __name__)

def get_token():
    # ưu tiên Authorization header
    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        return auth_header.replace('Bearer ', '')
    # fallback sang cookie
    return request.cookies.get('token')

def role_required(allowed_roles):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            token = get_token()
            if not token:
                return jsonify({
                    'error': 'Chưa đăng nhập'
                }), 401
            try:
                data = jwt.decode(
                    token,
                    Config.SECRET_KEY,
                    algorithms=['HS256']
                )
                if data['role_id'] not in allowed_roles:
                    return jsonify({
                        'error': 'Không có quyền truy cập'
                    }), 403
                request.user = data
                return func(*args, **kwargs)
            except jwt.ExpiredSignatureError:
                return jsonify({
                    'error': 'Token hết hạn'
                }), 401
            except Exception as e:
                print("JWT ERROR:", e)
                return jsonify({
                    'error': 'Token không hợp lệ'
                }), 401
        return wrapper
    return decorator

# Giao diện chính
def login_required(roles=[]):
    def wrapper(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            token = get_token()
            if not token:
                return redirect(url_for('view.login'))
            try:
                data = jwt.decode(
                    token,
                    Config.SECRET_KEY,
                    algorithms=['HS256']
                )
                if roles and data['role_id'] not in roles:
                    return "Không có quyền truy cập", 403
                request.user = data
            except Exception as e:
                print("LOGIN ERROR:", e)
                return redirect(url_for('view.login'))
            return f(*args, **kwargs)
        return decorated
    return wrapper

@auth.route('/redirect-by-role')
def redirect_by_role():
    token = request.cookies.get('token')
    if not token:
        return redirect('/login')
    try:
        data = jwt.decode(token, Config.SECRET_KEY, algorithms=['HS256'])
        role_id = data['role_id']
        if role_id == 1:
            return redirect('/admin/dashboard')
        elif role_id == 2:
            return redirect('/lecturer/dashboard')
        elif role_id == 3:
            return redirect('/student/dashboard')
    except Exception:
        return redirect('/login')