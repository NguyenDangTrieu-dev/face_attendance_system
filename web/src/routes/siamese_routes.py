from flask import Blueprint, request, jsonify, current_app
import os, cv2
from werkzeug.utils import secure_filename
from face_rec_SM_api import recognize_face
import logging

siamese_bp = Blueprint('siamese', __name__)
logger = logging.getLogger(__name__)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 5 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@siamese_bp.route('/recognize_siamese', methods=['POST'])
def recognize_siamese():
    if 'file' not in request.files:
        logger.warning("No file uploaded in request")
        return jsonify({'error': 'Không có hình ảnh được gửi. Vui lòng thử lại.'}), 400

    file = request.files['file']
    if file.filename == '':
        logger.warning("No file selected in request")
        return jsonify({'error': 'Không có hình ảnh được chọn. Vui lòng thử lại.'}), 400

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    if file_size > MAX_FILE_SIZE:
        logger.warning(f"File too large: {file_size} bytes, max allowed: {MAX_FILE_SIZE} bytes")
        return jsonify({'error': 'Hình ảnh quá lớn. Kích thước tối đa là 5MB.'}), 400
    file.seek(0)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logger.info(f"Saving file to {filepath}")
        file.save(filepath)

        try:
            frame = cv2.imread(filepath)
            if frame is None:
                logger.error(f"Failed to read image from {filepath}")
                return jsonify({'error': 'Không thể đọc hình ảnh. Vui lòng thử lại với tệp hợp lệ.'}), 400

            logger.info("Recognizing faces in image using Siamese Network")
            results = recognize_face(frame)
            logger.info(f"Recognition results (Siamese): {results}")
            return jsonify(results)
        except Exception as e:
            logger.error(f"Error during face recognition (Siamese): {str(e)}")
            return jsonify({'error': f'Đã xảy ra lỗi khi nhận diện: {str(e)}'}), 500
        finally:
            if os.path.exists(filepath):
                logger.info(f"Removing temporary file: {filepath}")
                os.remove(filepath)

    logger.warning(f"Invalid file type: {file.filename}")
    return jsonify({'error': 'Định dạng hình ảnh không hợp lệ. Chỉ hỗ trợ PNG, JPG, JPEG.'}), 400

