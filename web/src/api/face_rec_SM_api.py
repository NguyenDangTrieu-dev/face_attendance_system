import os
import cv2
import base64
import logging
import numpy as np
import psycopg2
import tensorflow as tf
from datetime import datetime
from keras.models import load_model
from src.align import detect_face
from src.facenet import load_model as load_facenet, prewhiten

import tensorflow as tf
if not tf.executing_eagerly():
    tf.compat.v1.enable_eager_execution()
tf.config.run_functions_eagerly(True)
# --- Config ---
FACENET_MODEL_PATH = 'Models/20180402-114759.pb'
SIAMESE_MODEL_PATH = 'Models/finetuned_siamese_model.keras'
INPUT_IMAGE_SIZE = 160
THRESHOLD = 0.9

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Database ---
POSTGRES_CONFIG = {
    'dbname': os.getenv('POSTGRES_DB'),
    'user': os.getenv('POSTGRES_USER'),
    'password': os.getenv('POSTGRES_PASSWORD'),
    'host': os.getenv('POSTGRES_HOST'),
    'port': os.getenv('POSTGRES_PORT')
}

def get_connection():
    return psycopg2.connect(**POSTGRES_CONFIG)

# --- Cosine similarity ---
def cosine_numpy(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)

# --- L2 normalization ---
def l2_normalize(x, axis=-1, epsilon=1e-10):
    return x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))

# --- Siamese cosine layer ---
def cosine_similarity(vects):
    x, y = vects
    x = tf.nn.l2_normalize(x, axis=-1)
    y = tf.nn.l2_normalize(y, axis=-1)
    return tf.reduce_sum(x * y, axis=-1, keepdims=True)

# --- Image to base64 ---
def image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

# --- Load models ---
graph = tf.Graph()
sess = tf.compat.v1.Session(graph=graph)

def init_models():
    global pnet, rnet, onet, images_placeholder, embeddings_tensor, phase_train_placeholder
    with graph.as_default():
        load_facenet(FACENET_MODEL_PATH)
        images_placeholder = graph.get_tensor_by_name("input:0")
        embeddings_tensor = graph.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")
        pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
        logger.info("✅ Đã tải MTCNN và FaceNet.")

siamese_model = load_model(
    SIAMESE_MODEL_PATH,
    custom_objects={'cosine_similarity': cosine_similarity},
    compile=False
)

# --- Preprocess face ---
def preprocess_face(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    cropped = frame[y1:y2, x1:x2]
    if cropped.size == 0:
        return None
    scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE))
    return prewhiten(scaled)

# --- Get FaceNet embeddings ---
def get_embeddings(frames):
    feed_dict = {images_placeholder: frames, phase_train_placeholder: False}
    emb = sess.run(embeddings_tensor, feed_dict=feed_dict)
    return l2_normalize(emb)

# --- Load embeddings from PostgreSQL ---
import psycopg2
import numpy as np

def load_embeddings_from_db(course_id):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT u.id, e.embedding
            FROM users u
            JOIN course_students cs ON u.id = cs.student_id
            JOIN embeddings e ON u.id = e.user_id
            WHERE cs.course_id = %s
        """, (course_id,))
        rows = cur.fetchall()
        cur.close()
        conn.close()

        emb_dict = {}
        for user_id, emb_bytes in rows:
            try:
                if len(emb_bytes) != 2048:  # 512 float32 = 2048 bytes
                    logger.error(f"Invalid embedding size for user_id {user_id}: {len(emb_bytes)} bytes")
                    continue
                emb_np = np.frombuffer(emb_bytes, dtype=np.float32)
                if emb_np.shape != (512,):
                    logger.error(f"Invalid embedding shape for user_id {user_id}: {emb_np.shape}")
                    continue
                emb_np = l2_normalize(emb_np)
                emb_dict[user_id] = emb_np
                logger.debug(f"Loaded embedding for user_id {user_id}, shape: {emb_np.shape}")
            except Exception as e:
                logger.error(f"Failed to parse embedding for user_id {user_id}: {str(e)}")
                continue
        logger.info(f"Loaded {len(emb_dict)} embeddings for course_id {course_id}")
        return emb_dict
    except Exception as e:
        logger.error(f"Error loading embeddings for course_id {course_id}: {str(e)}")
        return {}
# --- Get user info ---
def get_user_info(user_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT full_name FROM users WHERE id = %s", (user_id,))
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    return row[0] if row else None

# --- Save attendance ---
def save_attendance(student_id, course_id, base64_image=None):
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO attendance (student_id, course_id, image_base64, time)
            VALUES (%s, %s, %s, %s)
            """,
            (student_id, course_id, base64_image, timestamp)
        )

        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f"✅ Đã lưu điểm danh cho {student_id} trong khóa học {course_id}")

    except Exception as e:
        logger.error(f"❌ Lỗi lưu điểm danh: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
            cursor.close()
            conn.close()

# --- Main recognition logic ---
def is_user_in_course(user_id, course_id):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT EXISTS (
                SELECT 1 FROM course_students WHERE student_id = %s AND course_id = %s
            )
        """, (user_id, course_id))
        exists = cur.fetchone()[0]
        cur.close()
        conn.close()
        return exists
    except Exception as e:
        logger.error(f"Error checking user in course: {str(e)}")
        return False

def recognize_face(frame, course_id):
    try:
        logger.info(f"🔍 Nhận diện khuôn mặt cho course_id: {course_id}")
        results = []

        with graph.as_default():
            with sess.as_default():
                bounding_boxes, _ = detect_face.detect_face(
                    frame, 30, pnet, rnet, onet,
                    [0.6, 0.7, 0.7], 0.709
                )

                faces_found = bounding_boxes.shape[0]
                if faces_found == 0:
                    logger.info("No faces detected in frame")
                    return []

                det = bounding_boxes[:, 0:4]
                faces = []
                bboxes = []

                for i in range(faces_found):
                    face = preprocess_face(frame, det[i])
                    if face is not None:
                        faces.append(face)
                        bboxes.append(det[i])

                if not faces:
                    logger.info("No valid faces after preprocessing")
                    return []

                faces_np = np.stack(faces)
                embeddings = get_embeddings(faces_np)
                if embeddings.shape[1] != 512:
                    logger.error(f"Invalid embedding dimension: {embeddings.shape[1]}")
                    return []

        db_embeddings = load_embeddings_from_db(course_id)
        if not db_embeddings:
            logger.warning(f"No embeddings loaded for course_id {course_id}")
            return []

        for i, emb in enumerate(embeddings):
            logger.info(f"\n🔎 Đang xử lý khuôn mặt {i + 1}...")
            if emb.shape != (512,):
                logger.error(f"Invalid embedding shape for face {i + 1}: {emb.shape}")
                continue

            cosine_scores = []
            for user_id, db_emb in db_embeddings.items():
                score = cosine_numpy(emb, db_emb)
                cosine_scores.append((user_id, db_emb, score))

            top_candidates = sorted(cosine_scores, key=lambda x: x[2], reverse=True)[:3]
            logger.debug(f"Top 3 candidates: {[cand[0] for cand in top_candidates]}")

            best_name = "Unknown"
            best_score = -1
            best_user_id = None

            for user_id, db_emb, basic_score in top_candidates:
                emb1 = np.expand_dims(emb, axis=0).astype(np.float32)
                emb2 = np.expand_dims(db_emb, axis=0).astype(np.float32)
                if emb1.shape != (1, 512) or emb2.shape != (1, 512):
                    logger.error(f"Invalid input shapes: emb1={emb1.shape}, emb2={emb2.shape}")
                    continue

                try:
                    siamese_score = siamese_model.predict([emb1, emb2], verbose=0)[0][0]
                except Exception as e:
                    logger.error(f"Siamese predict error for user_id {user_id}: {str(e)}")
                    continue

                combined_score = 0.7 * siamese_score + 0.3 * basic_score
                logger.debug(f"User {user_id}: siamese_score={siamese_score:.4f}, basic_score={basic_score:.4f}, combined_score={combined_score:.4f}")

                if combined_score > best_score and combined_score > THRESHOLD:
                    best_score = combined_score
                    best_name = get_user_info(user_id)
                    best_user_id = user_id

            x1, y1, x2, y2 = map(int, bboxes[i])
            result = {
                "name": best_name,
                "bbox": [x1, y1, x2, y2]
            }

            if best_name != "Unknown" and best_user_id:
                result["similarity"] = round(float(best_score), 4)
                face_img = frame[y1:y2, x1:x2]
                if face_img.size > 0:
                    base64_image = image_to_base64(face_img)
                    # Kiểm tra trước khi lưu điểm danh
                    if is_user_in_course(best_user_id, course_id):
                        save_attendance(best_user_id, course_id, base64_image)
                    else:
                        logger.warning(f"User {best_user_id} not in course {course_id}, skipping attendance")
                else:
                    logger.warning(f"Empty face image for user_id {best_user_id}")

            results.append(result)

        return results

    except Exception as e:
        logger.error(f"❌ Lỗi nhận diện khuôn mặt: {str(e)}")
        return []

init_models()