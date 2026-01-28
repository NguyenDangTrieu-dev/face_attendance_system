# src/student_import.py
import os
import cv2
import zipfile
import tempfile
import numpy as np
from register import register_face


def import_from_images(student_id: str, name: str, image_files):
    """
    image_files: list[file-like] hoặc list[path]
    """
    frames = []

    for f in image_files:
        if isinstance(f, str):
            img = cv2.imread(f)
        else:
            img_bytes = f.read()
            npimg = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is not None:
            frames.append(img)

    return register_face(student_id, name, frames)


def import_from_zip(zip_file):
    results = []

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "students.zip")
        zip_file.save(zip_path)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)

        # ===== FIX: phát hiện thư mục gốc =====
        items = os.listdir(tmpdir)
        root_dir = tmpdir

        if len(items) == 1:
            possible_root = os.path.join(tmpdir, items[0])
            if os.path.isdir(possible_root):
                root_dir = possible_root

        # ===== Duyệt MSSV-HoTen =====
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)

            if not os.path.isdir(folder_path):
                continue

            if "-" not in folder:
                results.append({
                    "folder": folder,
                    "status": "❌ Sai format thư mục (MSSV-HoTen)"
                })
                continue

            student_id, name = folder.split("-", 1)

            frames = []
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)

                # chỉ nhận ảnh
                if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".avif")):
                    continue

                img = cv2.imread(img_path)
                if img is not None:
                    frames.append(img)

            if len(frames) < 5:
                results.append({
                    "student_id": student_id,
                    "name": name,
                    "status": "❌ Không đủ ảnh (>=5)"
                })
                continue

            res, status = register_face(student_id, name, frames)

            results.append({
                "student_id": student_id,
                "name": name,
                "status": res.get("message", "Lỗi")
            })

    return results

