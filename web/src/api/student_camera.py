# src/student_camera.py
import os
import cv2
import numpy as np
import unicodedata
import psycopg2
from register import register_face

BASE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "data",
    "students"
)

# Database config
POSTGRES_CONFIG = {
    'dbname': os.getenv('POSTGRES_DB'),
    'user': os.getenv('POSTGRES_USER'),
    'password': os.getenv('POSTGRES_PASSWORD'),
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': int(os.getenv('POSTGRES_PORT', 5432))
}


def normalize_name(text: str) -> str:
    """
    Chuyển tên có dấu → không dấu, an toàn cho path
    """
    text = unicodedata.normalize("NFD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")
    return text.replace(" ", "_")


def register_from_uploaded_images(student_id: str, name: str, images):
    """
    Đăng ký sinh viên MỚI vào hệ thống:
    1. Tạo user trong bảng users (nếu chưa tồn tại)
    2. Lưu ảnh vào thư mục
    3. Tạo embedding khuôn mặt
    
    images: list[FileStorage] từ request.files.getlist("images")
    """

    # Validate input
    if not student_id or not name:
        return {"error": "Thiếu MSSV hoặc họ tên"}, 400
    
    if len(student_id) != 10 or not student_id.isdigit():
        return {"error": "MSSV phải gồm 10 chữ số"}, 400

    # 1. KIỂM TRA VÀ TẠO USER TRONG DATABASE
    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        cur = conn.cursor()
        
        # Kiểm tra user đã tồn tại chưa
        cur.execute("SELECT id FROM users WHERE id = %s", (student_id,))
        existing = cur.fetchone()
        
        if existing:
            cur.close()
            conn.close()
            return {
                "error": f"Sinh viên {student_id} đã tồn tại trong hệ thống"
            }, 400
        
        # Tạo user mới với role_id = 3 (sinh viên)
        cur.execute("""
            INSERT INTO users (id, full_name, password, role_id)
            VALUES (%s, %s, %s, %s)
        """, (student_id, name, '123456', 3))  # Mật khẩu mặc định: 123456
        
        conn.commit()
        cur.close()
        conn.close()
        
        print(f"✅ Đã tạo user mới: {student_id} - {name}")
        
    except Exception as e:
        print(f"❌ Lỗi tạo user: {e}")
        if 'conn' in locals():
            conn.rollback()
            cur.close()
            conn.close()
        return {"error": f"Lỗi tạo user: {str(e)}"}, 500

    # 2. LƯU ẢNH VÀO THƯ MỤC
    safe_name = normalize_name(name)
    folder_name = f"{student_id}-{safe_name}"
    save_dir = os.path.abspath(os.path.join(BASE_DIR, folder_name))
    os.makedirs(save_dir, exist_ok=True)

    frames = []

    for idx, file in enumerate(images):
        try:
            file.seek(0)  # ⚠️ QUAN TRỌNG
            img_bytes = file.read()

            npimg = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

            if img is None:
                continue

            img_path = os.path.join(save_dir, f"{idx + 1}.jpg")
            cv2.imwrite(img_path, img)

            frames.append(img)

        except Exception as e:
            print(f"[WARN] Lỗi ảnh {idx}: {e}")
            continue

    if len(frames) < 5:
        # XÓA USER ĐÃ TẠO vì không đủ ảnh
        try:
            conn = psycopg2.connect(**POSTGRES_CONFIG)
            cur = conn.cursor()
            cur.execute("DELETE FROM users WHERE id = %s", (student_id,))
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            print(f"[WARN] Không xóa được user: {e}")
        
        return {
            "error": "Không đủ ảnh hợp lệ (cần ít nhất 5 ảnh)",
            "saved": len(frames)
        }, 400

    # 3. TẠO EMBEDDING KHUÔN MẶT
    result, status = register_face(student_id, name, frames)
    
    if status != 200:
        # XÓA USER nếu đăng ký khuôn mặt thất bại
        try:
            conn = psycopg2.connect(**POSTGRES_CONFIG)
            cur = conn.cursor()
            cur.execute("DELETE FROM users WHERE id = %s", (student_id,))
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            print(f"[WARN] Không xóa được user: {e}")
        
        return result, status

    # 4. INVALIDATE CACHE (nếu cần)
    from face_rec_SM_api import invalidate_embedding_cache
    invalidate_embedding_cache()

    return {
        "message": "✅ Tạo sinh viên mới và đăng ký khuôn mặt thành công!",
        "student_id": student_id,
        "name": name,
        "saved_images": len(frames),
        "folder": folder_name,
        "password": "123456"  # Thông báo mật khẩu mặc định
    }, 200