from flask import Blueprint, request, jsonify, current_app
import os, cv2, numpy as np
from werkzeug.utils import secure_filename
from register import register_face
import logging

register_bp = Blueprint('register', __name__)
logger = logging.getLogger(__name__)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 5 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@register_bp.route('/register', methods=['POST'])
def register():
    if 'student_id' not in request.form or 'name' not in request.form:
        logger.warning("Missing student_id or name in registration request")
        return jsonify({'success': False, 'message': 'Thiếu mã số sinh viên hoặc họ tên'}), 400
    
    student_id = request.form['student_id']
    name = request.form['name']

    if len(student_id) != 10 or not student_id.isdigit():
        logger.warning(f"Invalid student_id format: {student_id}")
        return jsonify({'success': False, 'message': 'Mã số sinh viên phải gồm 10 chữ số'}), 400

    # Kiểm tra danh sách 10 ảnh (thay vì 5 ảnh)
    if 'files' not in request.files:
        logger.warning("No files uploaded in registration request")
        return jsonify({'success': False, 'message': 'Không có hình ảnh được gửi. Vui lòng gửi đúng 10 ảnh.'}), 400

    files = request.files.getlist('files')
    if len(files) != 10:  # Thay đổi từ 5 thành 10
        logger.warning(f"Expected 10 files, but received {len(files)}")
        return jsonify({'success': False, 'message': 'Vui lòng gửi đúng 10 ảnh để đăng ký.'}), 400

    frames = []
    temp_filepaths = []
    try:
        # Xử lý từng file
        for file in files:
            if file.filename == '':
                logger.warning("No file selected in registration request")
                return jsonify({'success': False, 'message': 'Một trong các hình ảnh không được chọn. Vui lòng thử lại.'}), 400

            if not allowed_file(file.filename):
                logger.warning(f"Invalid file type in registration request: {file.filename}")
                return jsonify({'success': False, 'message': 'Định dạng hình ảnh không hợp lệ. Chỉ hỗ trợ PNG, JPG, JPEG.'}), 400

            # Kiểm tra kích thước tệp
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            if file_size > MAX_FILE_SIZE:
                logger.warning(f"File too large in registration request: {file_size} bytes, max allowed: {MAX_FILE_SIZE} bytes")
                return jsonify({'success': False, 'message': 'Hình ảnh quá lớn. Kích thước tối đa là 5MB.'}), 400
            file.seek(0)

            # Lưu ảnh tạm thời
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            logger.info(f"Saving file to {filepath}")
            file.save(filepath)
            temp_filepaths.append(filepath)

            # Đọc ảnh bằng OpenCV
            frame = cv2.imread(filepath)
            if frame is None:
                logger.error(f"Failed to read image from {filepath}")
                # Thử đọc ảnh dưới dạng binary và chuyển đổi
                with open(filepath, 'rb') as f:
                    img_data = f.read()
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    return jsonify({'success': False, 'message': 'Không thể đọc hình ảnh. Vui lòng thử lại với hình ảnh hợp lệ.'}), 400
                # Lưu lại ảnh dưới định dạng đúng
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
        # Xóa các file tạm thời
        for filepath in temp_filepaths:
            if os.path.exists(filepath):
                logger.info(f"Removing temporary file: {filepath}")
                os.remove(filepath)


