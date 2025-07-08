import os
import base64
import sys
import cv2
import psycopg2
import numpy as np
import logging
import tensorflow as tf
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.align import detect_face
from src.facenet import load_model, prewhiten

load_dotenv()

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Đọc biến môi trường
POSTGRES_CONFIG = {
    'dbname': os.getenv('POSTGRES_DB'),
    'user': os.getenv('POSTGRES_USER'),
    'password': os.getenv('POSTGRES_PASSWORD'),
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': int(os.getenv('POSTGRES_PORT', 5432))
}

FACENET_MODEL_PATH = 'Models/20180402-114759.pb'
INPUT_IMAGE_SIZE = 160

graph = tf.Graph()
sess = tf.compat.v1.Session(graph=graph)

with graph.as_default():
    load_model(FACENET_MODEL_PATH)
    images_placeholder = graph.get_tensor_by_name("input:0")
    embeddings_tensor = graph.get_tensor_by_name("embeddings:0")
    phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")
    pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

def preprocess_face(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    cropped = frame[y1:y2, x1:x2]
    if cropped.size == 0:
        return None
    scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE))
    return prewhiten(scaled)


def register_face(student_id, name, frames):
    try:
        if not student_id or not name:
            return {"error": "Thiếu mã số sinh viên hoặc họ tên"}, 400

        if len(student_id) != 10 or not student_id.isdigit():
            return {"error": "Mã số sinh viên phải gồm 10 chữ số"}, 400

        embeddings = []
        first_valid_image = None

        with graph.as_default():
            with sess.as_default():
                for frame in frames:
                    bounding_boxes, _ = detect_face.detect_face(
                        frame, 20, pnet, rnet, onet,
                        threshold=[0.5, 0.6, 0.6], factor=0.709
                    )
                    if bounding_boxes.shape[0] == 0:
                        logger.warning(f"No face detected in frame for student_id {student_id}")
                        continue

                    det = bounding_boxes[:, 0:4]
                    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in det]
                    largest_face_idx = np.argmax(areas)
                    bbox = det[largest_face_idx]

                    if bbox[2] - bbox[0] < 50 or bbox[3] - bbox[1] < 50:
                        logger.warning(f"Face too small for student_id {student_id}")
                        continue

                    scaled = preprocess_face(frame, bbox)
                    if scaled is None:
                        logger.warning(f"Invalid face crop for student_id {student_id}")
                        continue

                    scaled = np.expand_dims(scaled, axis=0)
                    feed_dict = {images_placeholder: scaled, phase_train_placeholder: False}
                    emb = sess.run(embeddings_tensor, feed_dict=feed_dict)[0]
                    embeddings.append(emb)

                    if first_valid_image is None:
                        _, buffer = cv2.imencode('.jpg', frame)
                        first_valid_image = base64.b64encode(buffer).decode('utf-8')

                    if len(embeddings) >= 10:
                        break

        if len(embeddings) < 5:
            logger.error(f"Not enough valid faces detected for student_id {student_id}: {len(embeddings)}")
            return {"error": "Cần ít nhất 5 ảnh hợp lệ để đăng ký"}, 400

        # Tính trung bình embedding
        avg_embedding = np.mean(embeddings, axis=0).astype(np.float32)
        # Chuyển thành nhị phân
        emb_bytes = avg_embedding.tobytes()

        conn = psycopg2.connect(**POSTGRES_CONFIG)
        cur = conn.cursor()

        # Cập nhật ảnh đại diện trong bảng users
        if first_valid_image:
            cur.execute(
                "UPDATE users SET image = %s WHERE id = %s",
                (first_valid_image, student_id)
            )

        # Lưu embedding vào bảng embeddings
        cur.execute(
            """
            INSERT INTO embeddings (user_id, embedding)
            VALUES (%s, %s)
            ON CONFLICT (user_id) DO UPDATE SET embedding = %s
            """,
            (student_id, emb_bytes, emb_bytes)
        )

        conn.commit()
        cur.close()
        conn.close()

        logger.info(f"✅ Đăng ký thành công cho {student_id} - {name}")
        return {"message": "Đăng ký thành công!", "id": student_id, "name": name}, 200

    except Exception as e:
        logger.error(f"❌ Lỗi đăng ký: {e}")
        if 'conn' in locals():
            conn.rollback()
            cur.close()
            conn.close()
        return {"error": f"Lỗi hệ thống: {str(e)}"}, 500
