import jwt
from flask import request, jsonify
from functools import wraps
import os

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
