import os
import cv2
import zipfile
import tempfile
import numpy as np
import re
from register import register_face,insert_user_if_not_exists


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

        # =========================
        # 🔥 PHASE 1: FIND STUDENTS
        # =========================
        student_folders = {}

        for root, dirs, files in os.walk(tmpdir):
            for folder in dirs:
                match = re.match(r"^(\d{5,})[-_](.+)", folder)
                if not match:
                    continue

                student_id = match.group(1)
                name = match.group(2)
                folder_path = os.path.join(root, folder)

                if student_id not in student_folders:
                    student_folders[student_id] = {
                        "name": name,
                        "path": folder_path
                    }

        print(f"[DEBUG] Found {len(student_folders)} students")

        # =========================
        # 🔥 PHASE 2: PROCESS
        # =========================
        for student_id, info in student_folders.items():
            name = info["name"]
            folder_path = info["path"]

            # ✅ FIX DB FIRST
            insert_user_if_not_exists(student_id, name)

            frames = []
            valid_faces = 0

            for root_img, _, files in os.walk(folder_path):
                for img_name in files:

                    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue

                    img_path = os.path.join(root_img, img_name)

                    try:
                        data = np.fromfile(img_path, dtype=np.uint8)
                        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                    except:
                        img = None

                    if img is not None:
                        frames.append(img)

            print(f"[DEBUG] {student_id} - {name} -> {len(frames)} images")

            # ❌ Not enough images
            if len(frames) < 3:
                results.append({
                    "student_id": student_id,
                    "name": name,
                    "status": "❌ Not enough images (>=3 required)"
                })
                continue

            # ✅ Register face
            try:
                res, st = register_face(student_id, name, frames)

                if st == 200:
                    message = res.get("message", "Đăng ký thành công!")
                else:
                    message = f"{res.get('error', '❌ Đăng ký thất bại')} (code {st})"

                results.append({
                    "student_id": student_id,
                    "name": name,
                    "status": message
                })

            except Exception as e:
                results.append({
                    "student_id": student_id,
                    "name": name,
                    "status": f"❌ Error: {str(e)}"
                })

    return results