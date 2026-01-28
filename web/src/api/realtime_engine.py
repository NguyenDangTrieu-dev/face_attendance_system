from __future__ import annotations

import os
import time
import threading
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from face_rec_SM_api import recognize_face

logger = logging.getLogger(__name__)

# Utils
def _clamp_bbox(x1, y1, x2, y2, w, h):
    x1 = int(max(0, min(x1, w - 1)))
    y1 = int(max(0, min(y1, h - 1)))
    x2 = int(max(0, min(x2, w - 1)))
    y2 = int(max(0, min(y2, h - 1)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2

def iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)

def ema_bbox(prev: Tuple[int, int, int, int], cur: Tuple[int, int, int, int], alpha=0.65):
    px1, py1, px2, py2 = prev
    cx1, cy1, cx2, cy2 = cur
    x1 = int(alpha * cx1 + (1 - alpha) * px1)
    y1 = int(alpha * cy1 + (1 - alpha) * py1)
    x2 = int(alpha * cx2 + (1 - alpha) * px2)
    y2 = int(alpha * cy2 + (1 - alpha) * py2)
    return x1, y1, x2, y2

def encode_jpeg(frame_bgr: np.ndarray, quality: int = 80) -> Optional[bytes]:
    if frame_bgr is None:
        return None
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    return buf.tobytes()

# Data classes
@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]  # xyxy
    last_seen: float
    created_at: float
    hits: int = 1
    miss: int = 0
    confirmed: bool = False
    name: str = "Unknown"
    student_id: Optional[str] = None
    similarity: Optional[float] = None

    # AI scheduling
    last_ai_ts: float = 0.0
    ai_inflight: bool = False

    pending_crop: Optional[np.ndarray] = None

    def to_result(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "bbox": [int(self.bbox[0]), int(self.bbox[1]), int(self.bbox[2]), int(self.bbox[3])],
            "name": self.name or "Unknown",
            "student_id": self.student_id,
            "similarity": self.similarity,
        }
    

# RealtimeEngine
class RealtimeEngine:
    def __init__(
        self,
        camera_src: int = 0,
        jpeg_quality: int = 80,
        detect_every_n: int = 2,
        confirm_frames: int = 2,
        ttl_seconds: float = 0.9,
        iou_threshold: float = 0.35,
        max_tracks: int = 20,
        ai_interval_sec: float = 0.8,
        attendance_cooldown_sec: float = 25.0,
    ):
        self.camera_src = camera_src
        self.jpeg_quality = jpeg_quality

        self.detect_every_n = max(1, int(detect_every_n))
        self.confirm_frames = max(1, int(confirm_frames))
        self.ttl_seconds = float(ttl_seconds)
        self.iou_threshold = float(iou_threshold)
        self.max_tracks = int(max_tracks)

        self.ai_interval_sec = float(ai_interval_sec)
        self.attendance_cooldown_sec = float(attendance_cooldown_sec)

        self._lock = threading.Lock()
        self._running = False
        self._course_id: Optional[int] = None

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None

        self._frame_idx = 0
        self._tracks: Dict[int, Track] = {}
        self._next_track_id = 1

        self._latest_jpeg: Optional[bytes] = None
        self._last_results: List[Dict[str, Any]] = []

        self._last_marked: Dict[str, float] = {}

        self.on_attendance: Optional[callable] = None

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    def get_last_results(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._last_results)

    def start(self, course_id: int, camera_src: Optional[int] = None):
        with self._lock:
            if self._running:
                self._course_id = int(course_id)
                return

            self._running = True
            self._course_id = int(course_id)
            if camera_src is not None:
                self.camera_src = int(camera_src)

            self._tracks = {}
            self._last_results = []
            self._frame_idx = 0
            self._next_track_id = 1
            self._last_marked = {}

        self._cap = cv2.VideoCapture(self.camera_src)
        if not self._cap.isOpened():
            with self._lock:
                self._running = False
            raise RuntimeError(f"Cannot open camera_src={self.camera_src}")

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("RealtimeEngine started course_id=%s camera_src=%s", course_id, self.camera_src)

    def stop(self):
        with self._lock:
            self._running = False

        if self._thread:
            self._thread.join(timeout=1.0)

        if self._cap:
            self._cap.release()

        with self._lock:
            self._thread = None
            self._cap = None
            self._tracks = {}
            self._last_results = []
            self._latest_jpeg = None

        logger.info("RealtimeEngine stopped")

    def process_frame(self, frame_bgr: np.ndarray, course_id: int) -> List[Dict[str, Any]]:
        now = time.time()
        dets = self._detect_faces(frame_bgr, course_id)
        self._update_tracks(frame_bgr, dets, now, course_id)
        results = self._build_results(now)
        return results

    def _loop(self):
        while True:
            with self._lock:
                if not self._running:
                    break
                course_id = self._course_id

            ok, frame = self._cap.read() if self._cap else (False, None)
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            self._frame_idx += 1
            now = time.time()

            dets: List[Dict[str, Any]] = []
            if (self._frame_idx % self.detect_every_n) == 0:
                try:
                    dets = self._detect_faces(frame, int(course_id))
                except Exception:
                    logger.exception("detect_faces failed")
                    dets = []

            self._update_tracks(frame, dets, now, int(course_id))
            results = self._build_results(now)

            overlay = self._draw_overlay(frame.copy(), results)

            jpg = encode_jpeg(overlay, quality=self.jpeg_quality)
            with self._lock:
                self._latest_jpeg = jpg
                self._last_results = results

            time.sleep(0.001)

    def _detect_faces(self, frame_bgr: np.ndarray, course_id: int) -> List[Dict[str, Any]]:
        raw = recognize_face(frame_bgr, course_id)

        dets: List[Dict[str, Any]] = []

        if isinstance(raw, list):
            for item in raw:
                bbox = item.get("bbox") or item.get("box") or item.get("bboxes")
                if bbox is None:
                    if all(k in item for k in ("x", "y", "w", "h")):
                        x1 = int(item["x"]); y1 = int(item["y"])
                        x2 = x1 + int(item["w"]); y2 = y1 + int(item["h"])
                        bbox = [x1, y1, x2, y2]
                if bbox is None:
                    continue
                dets.append({
                    "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                    "name": item.get("name", "Unknown"),
                    "student_id": item.get("student_id") or item.get("id"),
                    "similarity": item.get("similarity") or item.get("score"),
                })

        elif isinstance(raw, dict):
            faces = raw.get("faces") or raw.get("results") or raw.get("detections") or []
            if isinstance(faces, list):
                for item in faces:
                    bbox = item.get("bbox") or item.get("box")
                    if bbox is None:
                        continue
                    dets.append({
                        "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                        "name": item.get("name", "Unknown"),
                        "student_id": item.get("student_id") or item.get("id"),
                        "similarity": item.get("similarity") or item.get("score"),
                    })
        else:
            dets = []

        h, w = frame_bgr.shape[:2]
        out: List[Dict[str, Any]] = []
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            x1, y1, x2, y2 = _clamp_bbox(x1, y1, x2, y2, w, h)
            out.append({**d, "bbox": [x1, y1, x2, y2]})

        return out

    def _update_tracks(self, frame_bgr: np.ndarray, dets: List[Dict[str, Any]], now: float, course_id: int):
        h, w = frame_bgr.shape[:2]

        with self._lock:
            track_ids = list(self._tracks.keys())
            tracks = {tid: self._tracks[tid] for tid in track_ids}

        det_bboxes = [tuple(d["bbox"]) for d in dets]
        track_bboxes = [tracks[tid].bbox for tid in track_ids]

        matches: List[Tuple[int, int, float]] = []
        for ti, tb in enumerate(track_bboxes):
            for di, db in enumerate(det_bboxes):
                s = iou_xyxy(tb, db)
                if s >= self.iou_threshold:
                    matches.append((ti, di, s))

        matches.sort(key=lambda x: x[2], reverse=True)
        used_t = set()
        used_d = set()
        assign: List[Tuple[int, int]] = []

        for ti, di, s in matches:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            assign.append((ti, di))

        matched_track_ids = set()
        matched_det_ids = set()

        for ti, di in assign:
            tid = track_ids[ti]
            det = dets[di]
            matched_track_ids.add(tid)
            matched_det_ids.add(di)

            new_bbox = tuple(det["bbox"])
            tr = tracks[tid]

            sm = ema_bbox(tr.bbox, new_bbox, alpha=0.70)
            sm = _clamp_bbox(sm[0], sm[1], sm[2], sm[3], w, h)

            tr.bbox = sm
            tr.last_seen = now
            tr.miss = 0
            tr.hits += 1
            if (not tr.confirmed) and tr.hits >= self.confirm_frames:
                tr.confirmed = True

            det_name = det.get("name")
            det_sid = det.get("student_id")
            det_sim = det.get("similarity")

            if det_name and det_name != "Unknown":
                tr.name = det_name
            if det_sid:
                tr.student_id = str(det_sid)
            if det_sim is not None:
                try:
                    tr.similarity = float(det_sim)
                except Exception:
                    tr.similarity = None

            self._maybe_mark_attendance(tr, course_id, frame_bgr, now)

        for tid, tr in tracks.items():
            if tid in matched_track_ids:
                continue
            tr.miss += 1

        for di, det in enumerate(dets):
            if di in matched_det_ids:
                continue
            if len(tracks) >= self.max_tracks:
                continue

            bbox = tuple(det["bbox"])
            x1, y1, x2, y2 = _clamp_bbox(bbox[0], bbox[1], bbox[2], bbox[3], w, h)

            tid = self._next_track_id
            self._next_track_id += 1

            tr = Track(
                track_id=tid,
                bbox=(x1, y1, x2, y2),
                last_seen=now,
                created_at=now,
                hits=1,
                miss=0,
                confirmed=False,
                name=det.get("name") or "Unknown",
                student_id=str(det["student_id"]) if det.get("student_id") else None,
                similarity=float(det["similarity"]) if det.get("similarity") is not None else None,
                last_ai_ts=0.0,
            )

            tracks[tid] = tr

            self._maybe_mark_attendance(tr, course_id, frame_bgr, now)

        expired = [tid for tid, tr in tracks.items() if (now - tr.last_seen) > self.ttl_seconds]

        for tid in expired:
            tracks.pop(tid, None)

        with self._lock:
            self._tracks = tracks

    def crop_face(frame: np.ndarray, bbox, padding=0.15):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        bw = x2 - x1
        bh = y2 - y1

        px = int(bw * padding)
        py = int(bh * padding)

        x1 = max(0, x1 - px)
        y1 = max(0, y1 - py)
        x2 = min(w, x2 + px)
        y2 = min(h, y2 + py)

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return None

        return face

    def _maybe_mark_attendance(self, tr: Track, course_id: int, frame_bgr: np.ndarray, now: float):
        if not tr.student_id or not tr.confirmed:
            return

        sid = str(tr.student_id)

        last = self._last_marked.get(sid)
        if last and (now - last) < self.attendance_cooldown_sec:
            return

        face = self.crop_face(frame_bgr, tr.bbox)
        if face is None:
            return

        self._last_marked[sid] = now

        if self.on_attendance:
            try:
                self.on_attendance(
                    track=tr,
                    course_id=course_id,
                    face_img=face,   # ⬅️ CHỈ face
                    timestamp=now
                )
            except Exception:
                logger.exception("on_attendance failed")

    def _build_results(self, now: float) -> List[Dict[str, Any]]:
        with self._lock:
            tracks = list(self._tracks.values())

        results = []
        for tr in tracks:
            if not tr.confirmed:
                continue
            if (now - tr.last_seen) > self.ttl_seconds:
                continue
            results.append(tr.to_result())
        return results

    def _draw_overlay(self, frame_bgr: np.ndarray, results: List[Dict[str, Any]]) -> np.ndarray:
        for r in results:
            bbox = r.get("bbox") or []
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            name = r.get("name") or "Unknown"
            tid = r.get("track_id", 0)
            sim = r.get("similarity", None)

            is_known = name != "Unknown"
            color = (0, 255, 0) if is_known else (0, 165, 255)

            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
            label = f"#{tid} {name}"
            if sim is not None and is_known:
                try:
                    label += f" ({float(sim):.2f})"
                except Exception:
                    pass

            cv2.putText(
                frame_bgr,
                label,
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
        return frame_bgr