from venv import logger

from flask import Blueprint, request, jsonify
import cv2, base64, numpy as np

from face_rec_SM_api import recognize_face
from attendance_api import already_attended_today, is_student_in_course, save_attendance

recognition_bp = Blueprint('recognition', __name__)

from realtime_state import GLOBAL_STATE


@recognition_bp.route('/recognize_siamese', methods=['POST'])
def recognize_siamese():
    if "file" not in request.files or "course_id" not in request.form:
        return jsonify({"error": "Thiếu hình ảnh hoặc course_id"}), 400

    file = request.files["file"]
    course_id = int(request.form["course_id"])

    file_bytes = file.read()
    if len(file_bytes) == 0:
        return jsonify({"error": "File rỗng"}), 400

    if len(file_bytes) > 5 * 1024 * 1024:
        return jsonify({"error": "Ảnh vượt quá 5MB"}), 400

    frame = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Không decode được ảnh"}), 400
    
    try:
        results = recognize_face(frame, course_id)

        recognized = []
        skipped = []

        h, w = frame.shape[:2]

        for r in results:

            student_id = r["student_id"]
            similarity = r["similarity"]

            print("SIMILARITY:", student_id, similarity)

            # 1. threshold hợp lý
            if similarity < 0.75:
                skipped.append({
                    "student_id": student_id,
                    "reason": "low_similarity"
                })
                continue

            # 2. check thuộc lớp (PHẢI LÀM TRƯỚC SAVE)
            if not is_student_in_course(student_id, course_id):
                print("BLOCKED NOT IN COURSE:", student_id)
                skipped.append({
                    "student_id": student_id,
                    "reason": "not_in_course"
                })
                continue

            # 3. check điểm danh hôm nay
            if already_attended_today(student_id, course_id):
                print("BLOCKED DUPLICATE:", student_id)
                skipped.append({
                    "student_id": student_id,
                    "reason": "already_attended"
                })
                continue

            # 4. crop face
            x1, y1, x2, y2 = map(int, r["bbox"])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face_crop = frame[y1:y2, x1:x2]
            image_base64 = None

            if face_crop.size > 0:
                _, buf = cv2.imencode(".jpg", face_crop)
                image_base64 = base64.b64encode(buf).decode()

            # 5. INSERT (CHỈ Ở CUỐI)
            ok = save_attendance(student_id, course_id, image_base64)

            if ok:
                print("INSERT SUCCESS:", student_id)

                recognized.append({
                    "student_id": student_id,
                    "name": r["name"],
                    "similarity": similarity
                })
            else:
                print("INSERT FAILED:", student_id)

        return jsonify({
            "status": "ok",
            "recognized": recognized,
            "skipped": skipped
        })

    except Exception as e:
        logger.exception("recognize_siamese error")
        return jsonify({"error": str(e)}), 500
