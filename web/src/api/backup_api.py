import os
import subprocess
from datetime import datetime
from pathlib import Path

from flask import Blueprint, jsonify, send_file, request, after_this_request
from werkzeug.utils import secure_filename

from config import Config
from auth import role_required


backup_bp = Blueprint("backup", __name__)

BASE_DIR = Path(__file__).resolve().parent
BACKUP_DIR = BASE_DIR / "backups"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".dump"}


def db_env():
    env = os.environ.copy()
    env["PGPASSWORD"] = str(Config.DB_CONFIG["password"])
    return env


def db_args():
    return [
        "-h", str(Config.DB_CONFIG["host"]),
        "-p", str(Config.DB_CONFIG["port"]),
        "-U", str(Config.DB_CONFIG["user"]),
        "-d", str(Config.DB_CONFIG["database"]),
    ]


def run_command(command):
    result = subprocess.run(
        command,
        env=db_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout or "Command failed")

    return result


def make_backup_file(prefix="face_db_backup"):
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.dump"
    path = BACKUP_DIR / filename

    command = [
        "pg_dump",
        *db_args(),
        "-Fc",
        "-f", str(path)
    ]

    run_command(command)

    return path


def terminate_db_connections():
    database = Config.DB_CONFIG["database"]

    command = [
        "psql",
        "-h", str(Config.DB_CONFIG["host"]),
        "-p", str(Config.DB_CONFIG["port"]),
        "-U", str(Config.DB_CONFIG["user"]),
        "-d", "postgres",
        "-c",
        f"""
        SELECT pg_terminate_backend(pid)
        FROM pg_stat_activity
        WHERE datname = '{database}'
          AND pid <> pg_backend_pid();
        """
    ]

    run_command(command)


@backup_bp.route("/api/admin/backup/download", methods=["GET"])
@role_required([1])
def download_backup():
    try:
        backup_path = make_backup_file()

        @after_this_request
        def cleanup(response):
            try:
                if backup_path.exists():
                    backup_path.unlink()
            except Exception:
                pass
            return response

        return send_file(
            backup_path,
            as_attachment=True,
            download_name=backup_path.name,
            mimetype="application/octet-stream"
        )

    except Exception as e:
        return jsonify({
            "error": "Backup thất bại",
            "detail": str(e)
        }), 500


@backup_bp.route("/api/admin/backup/restore", methods=["POST"])
@role_required([1])
def restore_backup():
    if "file" not in request.files:
        return jsonify({"error": "Chưa chọn file backup"}), 400

    file = request.files["file"]

    if not file.filename:
        return jsonify({"error": "Tên file không hợp lệ"}), 400

    original_name = secure_filename(file.filename)
    ext = Path(original_name).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({
            "error": "Chỉ cho phép restore file .dump"
        }), 400

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    upload_path = BACKUP_DIR / f"restore_upload_{timestamp}.dump"

    try:
        file.save(upload_path)

        safety_backup = make_backup_file(prefix="before_restore")

        terminate_db_connections()

        command = [
            "pg_restore",
            *db_args(),
            "--clean",
            "--if-exists",
            "--no-owner",
            "--no-privileges",
            str(upload_path)
        ]

        run_command(command)

        return jsonify({
            "message": "Restore thành công",
            "safety_backup": safety_backup.name
        })

    except Exception as e:
        return jsonify({
            "error": "Restore thất bại",
            "detail": str(e)
        }), 500

    finally:
        try:
            if upload_path.exists():
                upload_path.unlink()
        except Exception:
            pass