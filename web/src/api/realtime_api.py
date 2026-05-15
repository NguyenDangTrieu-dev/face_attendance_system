import time

from flask import Blueprint, jsonify, request, Response

from src.api import realtime_state


def create_realtime_api(engine):
    bp = Blueprint('realtime', __name__)

    @bp.route('/api/realtime/start', methods=['POST'])
    def start_realtime():
        data = request.get_json()
        course_id = data.get("course_id")

        if not course_id:
            return jsonify({"running": False, "error": "missing course_id"}), 400

        try:
            engine.start(int(course_id))
            return jsonify({"running": True})
        except Exception as e:
            return jsonify({"running": False, "error": str(e)}), 500

    @bp.route('/api/realtime/stop', methods=['POST'])
    def stop_realtime():
        engine.stop()
        return jsonify({"ok": True})

    @bp.route('/api/realtime/status')
    def status():
        return jsonify({'running': engine.is_running()})

    @bp.route('/video_feed')
    def video():
        def gen():
            try:
                while engine.is_running():
                    frame = engine.get_latest_jpeg()

                    if frame:
                        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    else:
                        time.sleep(0.01)

            except GeneratorExit:
                print("Client disconnected")

        return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @bp.route('/api/realtime/events', methods=['GET'])
    def realtime_events():
        return jsonify({"success": True, "events": engine.get_realtime_events()})
    
    return bp