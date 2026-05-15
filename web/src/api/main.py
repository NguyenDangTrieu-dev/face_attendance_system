import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from flask import Flask
from course_api import course_bp
from attendance_api import attendance_bp
from src.api.lecture.lecture_api import lecturer_bp
from recognition_api import recognition_bp
from realtime_api import create_realtime_api
from src.api.student.student_api import student_bp
from src.api.admin.admin_api import admin_bp
from view_routes import view_bp
from backup_api import backup_bp
from exam_api import exam_bp

from auth import auth
from realtime_engine import RealtimeEngine

app = Flask(__name__)

engine = RealtimeEngine()

app.register_blueprint(view_bp)
app.register_blueprint(admin_bp)
app.register_blueprint(auth)
app.register_blueprint(course_bp)
app.register_blueprint(attendance_bp)
app.register_blueprint(lecturer_bp)
app.register_blueprint(recognition_bp)
app.register_blueprint(create_realtime_api(engine))
app.register_blueprint(student_bp)
app.register_blueprint(backup_bp)
app.register_blueprint(exam_bp)

view_bp.engine = engine

def mjpeg_generator():
    while True:
        frame = engine.get_latest_jpeg()
        if frame is None:
            import time
            time.sleep(0.02)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            frame +
            b"\r\n"
        )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)