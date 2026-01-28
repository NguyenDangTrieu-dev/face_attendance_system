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

# ========= TensorFlow runtime tweaks =========
# Note: TF1 graph/session + Facenet pb require compat.v1 session.
# Do NOT enable eager here; it can slow graph mode and cause subtle issues.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# --- Config ---
FACENET_MODEL_PATH = 'Models/20180402-114759.pb'
SIAMESE_MODEL_PATH = 'Models/finetuned_siamese_model.keras'
INPUT_IMAGE_SIZE = 160
THRESHOLD = 0.87

# Cache embeddings per course to avoid DB lag each request
EMBEDDING_CACHE_TTL_SEC = float(os.getenv("EMBEDDING_CACHE_TTL_SEC", "300"))  # 5 minutes
_EMBEDDING_CACHE = {}  # course_id -> {"ts": float, "data": {user_id: emb}}

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
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))

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
    ok, buffer = cv2.imencode('.jpg', image)
    if not ok:
        return None
    return base64.b64encode(buffer).decode('utf-8')

# --- Load models (TF1 graph/session for Facenet pb) ---
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
logger.info("✅ Đã tải Siamese model.")

# --- Preprocess face ---
def preprocess_face(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
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

# --- Load embeddings from PostgreSQL (with cache) ---
def load_embeddings_from_db(course_id):
    # TTL cache to avoid re-query each request
    now = datetime.now().timestamp()
    cached = _EMBEDDING_CACHE.get(str(course_id))
    if cached and (now - cached["ts"]) < EMBEDDING_CACHE_TTL_SEC:
        return cached["data"]

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
                if emb_bytes is None:
                    continue
                if len(emb_bytes) != 2048:  # 512 float32 = 2048 bytes
                    logger.warning(f"Invalid embedding size for user_id {user_id}: {len(emb_bytes)} bytes")
                    continue
                emb_np = np.frombuffer(emb_bytes, dtype=np.float32)
                if emb_np.shape != (512,):
                    logger.warning(f"Invalid embedding shape for user_id {user_id}: {emb_np.shape}")
                    continue
                emb_np = l2_normalize(emb_np)
                emb_dict[user_id] = emb_np
            except Exception as e:
                logger.warning(f"Failed to parse embedding for user_id {user_id}: {str(e)}")
                continue

        _EMBEDDING_CACHE[str(course_id)] = {"ts": now, "data": emb_dict}
        logger.info(f"Loaded {len(emb_dict)} embeddings for course_id {course_id} (cached)")
        return emb_dict

    except Exception as e:
        logger.error(f"Error loading embeddings for course_id {course_id}: {str(e)}")
        return {}

def invalidate_embedding_cache(course_id=None):
    """Call this after import_students or register to refresh cache."""
    if course_id is None:
        _EMBEDDING_CACHE.clear()
    else:
        _EMBEDDING_CACHE.pop(str(course_id), None)

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
            "DELETE FROM attendance WHERE student_id = %s AND course_id = %s",
            (student_id, course_id)
        )

        cursor.execute(
            """
            INSERT INTO attendance (student_id, course_id, image_base64, time, recognized)
            VALUES (%s, %s, %s, %s, TRUE)
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
        return bool(exists)
    except Exception as e:
        logger.error(f"Error checking user in course: {str(e)}")
        return False

def _resize_keep_aspect(frame, target_width: int):
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame, 1.0
    scale = target_width / float(w)
    new_h = int(h * scale)
    resized = cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale

# --- Main recognition logic ---
def recognize_face(frame, course_id, resize_for_speed: bool = False, target_width: int = 960):
    """
    Nhận diện khuôn mặt:
    - MTCNN detect face
    - FaceNet embedding
    - Cosine filter + Siamese verify (batch)
    """
    try:
        results = []

        # Optional resize for speed (use in realtime mode). If you resize, bbox is returned in resized coords.
        if resize_for_speed:
            frame, _ = _resize_keep_aspect(frame, target_width)

        with graph.as_default():
            with sess.as_default():
                bounding_boxes, _ = detect_face.detect_face(
                    frame, 30, pnet, rnet, onet,
                    [0.6, 0.7, 0.7], 0.709
                )

                faces_found = bounding_boxes.shape[0]
                if faces_found == 0:
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
                    return []

                faces_np = np.stack(faces)
                embeddings = get_embeddings(faces_np)
                if embeddings.shape[1] != 512:
                    logger.error(f"Invalid embedding dimension: {embeddings.shape[1]}")
                    return []

        db_embeddings = load_embeddings_from_db(course_id)
        if not db_embeddings:
            return []

        # For speed, materialize db arrays once
        db_user_ids = list(db_embeddings.keys())
        db_matrix = np.stack([db_embeddings[uid] for uid in db_user_ids]).astype(np.float32)  # (N,512)

        for i, emb in enumerate(embeddings):
            emb = emb.astype(np.float32)

            # Cosine against all (vectorized)
            # db_matrix already l2-normalized; emb is l2-normalized from get_embeddings()
            scores = db_matrix @ emb  # (N,)
            # Top-K candidates
            K = 3
            if scores.shape[0] < K:
                K = scores.shape[0]
            top_idx = np.argpartition(-scores, K-1)[:K]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            top_users = [db_user_ids[j] for j in top_idx]
            top_scores = [float(scores[j]) for j in top_idx]

            best_name = "Unknown"
            best_score = -1.0
            best_user_id = None

            # Early skip: if best cosine too low, skip Siamese (giữ mượt, Siamese vẫn dùng khi cần)
            if top_scores and top_scores[0] >= 0.75:
                emb1_batch = np.repeat(emb[np.newaxis, :], len(top_users), axis=0)
                emb2_batch = np.stack([db_embeddings[uid] for uid in top_users]).astype(np.float32)
                try:
                    siamese_scores = siamese_model.predict([emb1_batch, emb2_batch], verbose=0).reshape(-1)
                except Exception as e:
                    logger.error(f"Siamese batch predict error: {str(e)}")
                    siamese_scores = np.zeros((len(top_users),), dtype=np.float32)

                for uid, s_score, c_score in zip(top_users, siamese_scores, top_scores):
                    combined = 0.7 * float(s_score) + 0.3 * float(c_score)
                    if combined > best_score and combined > THRESHOLD:
                        best_score = combined
                        best_user_id = uid
                        best_name = get_user_info(uid) or "Unknown"

            x1, y1, x2, y2 = map(int, bboxes[i])
            result = {"name": best_name, "bbox": [x1, y1, x2, y2]}

            if best_name != "Unknown" and best_user_id:
                result["similarity"] = round(float(best_score), 4)

                face_img = frame[y1:y2, x1:x2]
                if face_img.size > 0:
                    base64_image = image_to_base64(face_img)
                    if is_user_in_course(best_user_id, course_id):
                        save_attendance(best_user_id, course_id, base64_image)
                results.append(result)
            else:
                results.append(result)

        return results

    except Exception as e:
        logger.error(f"❌ Lỗi nhận diện khuôn mặt: {str(e)}")
        return []

init_models()
