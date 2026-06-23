"""
Headless stop-line and red-light video processing helper.
Provides process_video_headless(...) to run detections on a video file without GUI.
Writes annotated video and a JSON results file into the Config.EVIDENCE_FOLDER and
returns the results dictionary.
"""

import os
import time
import uuid
import json
import math
from typing import List, Tuple, Dict, Optional
import cv2
import numpy as np

from .config import Config
from .models import ModelManager

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]

COLOR_GREEN = (0, 255, 0)
COLOR_ORANGE = (0, 165, 255)
COLOR_RED = (0, 0, 255)
COLOR_STOP_LINE = (0, 255, 0)
COLOR_RED_LIGHT_LINE = (255, 0, 0)
COLOR_TEXT = (255, 255, 255)


def _transcode_to_h264(src_path: str) -> bool:
    """
    Re-encode a video to browser-compatible H.264 (yuv420p) + faststart in-place.

    OpenCV's VideoWriter with the 'mp4v' fourcc produces MPEG-4 Part 2 video, which
    HTML5 <video> elements cannot decode (the player loads but stays black). We use
    the static ffmpeg binary bundled with imageio-ffmpeg so no system ffmpeg/codec
    install is required. Returns True on success, False if transcoding was skipped.
    """
    try:
        import subprocess
        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        tmp_out = src_path + ".h264.mp4"

        cmd = [
            ffmpeg_exe, "-y",
            "-i", src_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "veryfast",
            "-movflags", "+faststart",
            "-an",
            tmp_out,
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode == 0 and os.path.exists(tmp_out) and os.path.getsize(tmp_out) > 0:
            os.replace(tmp_out, src_path)
            return True

        if os.path.exists(tmp_out):
            os.remove(tmp_out)
        print(f"[stopline] ffmpeg transcode failed (rc={proc.returncode}): {proc.stderr.decode(errors='ignore')[:500]}")
        return False
    except Exception as e:
        print(f"[stopline] H.264 transcode skipped: {e}")
        return False


def bottom_center(bbox: BBox) -> Point:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)


def orientation(a: Point, b: Point, c: Point) -> int:
    val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    if abs(val) < 1e-9:
        return 0
    return 1 if val > 0 else 2


def on_segment(a: Point, b: Point, c: Point) -> bool:
    if min(a[0], c[0]) - 1e-9 <= b[0] <= max(a[0], c[0]) + 1e-9 and \
       min(a[1], c[1]) - 1e-9 <= b[1] <= max(a[1], c[1]) + 1e-9:
        return True
    return False


def segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and on_segment(p1, q1, p2):
        return True
    if o2 == 0 and on_segment(p1, q2, p2):
        return True
    if o3 == 0 and on_segment(q1, p1, q2):
        return True
    if o4 == 0 and on_segment(q1, p2, q2):
        return True
    return False


def point_side_of_line(pt: Point, a: Point, b: Point) -> float:
    return (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])


def _line_crossed(prev_pt: Point, curr_pt: Point, line_p1: Point, line_p2: Point) -> bool:
    crossed = segments_intersect(prev_pt, curr_pt, line_p1, line_p2)
    prev_side = point_side_of_line(prev_pt, line_p1, line_p2)
    curr_side = point_side_of_line(curr_pt, line_p1, line_p2)
    side_changed = prev_side * curr_side < 0
    return crossed or side_changed


def draw_info(
    frame,
    stop_line: Optional[Tuple[Point, Point]],
    red_light_line: Optional[Tuple[Point, Point]],
    light_state: str,
    violations_on_frame: List[Dict],
):
    if stop_line is not None:
        p1, p2 = stop_line
        cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), COLOR_STOP_LINE, 2)
        cv2.putText(
            frame, "Stop Line",
            (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_STOP_LINE, 2,
        )

    if red_light_line is not None:
        p1, p2 = red_light_line
        cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), COLOR_RED_LIGHT_LINE, 2)
        cv2.putText(
            frame, "Red Light Line",
            (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED_LIGHT_LINE, 2,
        )

    cv2.putText(
        frame, f"Light: {(light_state or 'red').upper()}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_TEXT, 2, cv2.LINE_AA,
    )

    for violation in violations_on_frame:
        if violation['type'] == 'stop_line_violation':
            cv2.putText(frame, "STOP LINE VIOLATION!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_ORANGE, 3)
        elif violation['type'] == 'red_light_violation':
            cv2.putText(frame, "RED LIGHT VIOLATION!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_RED, 3)


class SimpleTracker:
    def __init__(self, max_distance=80, max_missed=5):
        self.next_id = 1
        self.tracks = {}
        self.max_distance = max_distance
        self.max_missed = max_missed

    def _distance(self, p1: Point, p2: Point) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def update(self, detections: List[Dict], frame_idx: int) -> Dict[int, Dict]:
        assigned = {}
        used_track_ids = set()
        det_centroids = []
        for det in detections:
            cx, _ = bottom_center(det['bbox'])
            cy = (det['bbox'][1] + det['bbox'][3]) / 2.0
            det_centroids.append((det, (cx, cy)))

        remaining = []
        for det, cent in det_centroids:
            if 'id' in det and det['id'] in self.tracks:
                tid = det['id']
                self.tracks[tid]['bbox'] = det['bbox']
                self.tracks[tid]['centroid'] = cent
                self.tracks[tid]['last_frame'] = frame_idx
                self.tracks[tid]['missed'] = 0
                assigned[tid] = det
                used_track_ids.add(tid)
            else:
                remaining.append((det, cent))

        for det, cent in remaining:
            best_tid = None
            best_dist = float('inf')
            for tid, tr in self.tracks.items():
                if tid in used_track_ids:
                    continue
                d = self._distance(cent, tr['centroid'])
                if d < best_dist:
                    best_dist = d
                    best_tid = tid
            if best_tid is not None and best_dist <= self.max_distance:
                self.tracks[best_tid]['bbox'] = det['bbox']
                self.tracks[best_tid]['centroid'] = cent
                self.tracks[best_tid]['last_frame'] = frame_idx
                self.tracks[best_tid]['missed'] = 0
                assigned[best_tid] = det
                used_track_ids.add(best_tid)
            else:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    'bbox': det['bbox'],
                    'centroid': cent,
                    'last_frame': frame_idx,
                    'missed': 0,
                    'violated': False,
                    'violation_type': None,
                    'stop_line_crossed': False,
                    'red_light_crossed': False,
                    'history': [],
                }
                assigned[tid] = det
                used_track_ids.add(tid)

        for tid, tr in list(self.tracks.items()):
            if tr['last_frame'] != frame_idx:
                tr['missed'] += 1
                if tr['missed'] > self.max_missed:
                    del self.tracks[tid]

        for tid, det in assigned.items():
            det_cent = self.tracks[tid]['centroid']
            self.tracks[tid]['history'].append((frame_idx, det['bbox'], det_cent))

        out = {}
        for tid, det in assigned.items():
            det_copy = dict(det)
            det_copy['assigned_id'] = tid
            out[tid] = det_copy

        return out


class StopLineViolationDetector:
    def __init__(
        self,
        stop_line: Tuple[Point, Point],
        red_light_line: Tuple[Point, Point],
        max_track_distance=80,
    ):
        self.stop_line_p1, self.stop_line_p2 = stop_line
        self.red_light_line_p1, self.red_light_line_p2 = red_light_line
        self.tracker = SimpleTracker(max_distance=max_track_distance)
        self.violations = {}

    def process_frame(self, detections: List[Dict], light_state: str, frame_idx: int) -> List[Dict]:
        state = (light_state or '').strip().lower()
        assigned = self.tracker.update(detections, frame_idx)
        new_violations = []

        for tid, det in assigned.items():
            track = self.tracker.tracks[tid]
            hist = track['history']
            if len(hist) < 2:
                continue

            _, prev_bbox, _ = hist[-2]
            prev_pt = bottom_center(prev_bbox)
            curr_pt = bottom_center(hist[-1][1])

            stop_line_crossed = _line_crossed(prev_pt, curr_pt, self.stop_line_p1, self.stop_line_p2)
            red_line_crossed = _line_crossed(prev_pt, curr_pt, self.red_light_line_p1, self.red_light_line_p2)

            if stop_line_crossed and not track.get('stop_line_crossed', False):
                track['stop_line_crossed'] = True
                ev = {
                    'track_id': tid,
                    'frame_idx': frame_idx,
                    'bbox': det['bbox'],
                    'prev_point': prev_pt,
                    'curr_point': curr_pt,
                    'light_state': state,
                    'type': 'stop_line_violation',
                }
                self.violations[tid] = ev
                track['violated'] = True
                track['violation_type'] = 'stop_line_violation'
                new_violations.append(ev)

            elif track.get('stop_line_crossed', False) and red_line_crossed and not track.get('red_light_crossed', False):
                track['red_light_crossed'] = True
                ev = {
                    'track_id': tid,
                    'frame_idx': frame_idx,
                    'bbox': det['bbox'],
                    'prev_point': prev_pt,
                    'curr_point': curr_pt,
                    'light_state': state,
                    'type': 'red_light_violation',
                }
                self.violations[tid] = ev
                track['violated'] = True
                track['violation_type'] = 'red_light_violation'
                new_violations.append(ev)

        return new_violations


def _norm_line_to_abs(
    line_norm: Tuple[float, float, float, float],
    width: int,
    height: int,
) -> Tuple[Point, Point]:
    x1 = int(line_norm[0] * width)
    y1 = int(line_norm[1] * height)
    x2 = int(line_norm[2] * width)
    y2 = int(line_norm[3] * height)
    return ((x1, y1), (x2, y2))


def process_video_headless(
    video_path: str,
    model_path: Optional[str] = None,
    stop_line_norm: Optional[Tuple[float, float, float, float]] = None,
    red_light_line_norm: Optional[Tuple[float, float, float, float]] = None,
    initial_light_state: str = 'red',
    conf_thres: float = 0.3,
    output_basename: Optional[str] = None,
) -> Dict:
    """
    Process a video file headlessly: run object detection per frame, track objects,
    detect stop-line and red-light crossings, and write an annotated output video plus JSON.

    stop_line_norm / red_light_line_norm: normalized [x1,y1,x2,y2] in 0..1 relative to frame.
    """
    os.makedirs(Config.EVIDENCE_FOLDER, exist_ok=True)
    model_manager = ModelManager()

    if model_path:
        try:
            from ultralytics import YOLO as _YOLO
            model = _YOLO(model_path)
        except Exception as e:
            return {"success": False, "error": f"Failed to load model: {e}"}
    else:
        model = model_manager.load_uvh()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"success": False, "error": f"Failed to open video: {video_path}"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if stop_line_norm is not None:
        stop_line = _norm_line_to_abs(stop_line_norm, width, height)
    else:
        stop_line = ((0, int(height * 0.7)), (width, int(height * 0.7)))

    if red_light_line_norm is not None:
        red_light_line = _norm_line_to_abs(red_light_line_norm, width, height)
    else:
        red_light_line = ((0, int(height * 0.5)), (width, int(height * 0.5)))

    detector = StopLineViolationDetector(stop_line, red_light_line, max_track_distance=80)

    if not output_basename:
        output_basename = time.strftime("stopline_output_%Y%m%dT%H%M%S") + "_" + str(uuid.uuid4())[:8]
    output_video_path = os.path.join(Config.EVIDENCE_FOLDER, f"{output_basename}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_idx = 0
    violations_log = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                results = model(frame)
                bboxes = []
                if results and len(results) > 0 and hasattr(results[0], 'boxes'):
                    r = results[0]
                    boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes.xyxy, 'cpu') else np.array(r.boxes.xyxy)
                    confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes.conf, 'cpu') else np.array(r.boxes.conf)
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = box.astype(float).tolist()
                        conf = float(confs[i]) if i < len(confs) else 1.0
                        if conf < conf_thres:
                            continue
                        bboxes.append((x1, y1, x2, y2))
            except Exception:
                bboxes = []

            detections = [{'bbox': tuple(map(float, b))} for b in bboxes]

            violations_on_frame = []
            new_violations = detector.process_frame(detections, initial_light_state, frame_idx)
            if new_violations:
                violations_log.extend(new_violations)
                violations_on_frame = new_violations

            vis = frame.copy()

            for tid, tr in detector.tracker.tracks.items():
                if not tr.get('history'):
                    continue
                _, last_bbox, _ = tr['history'][-1]
                x1, y1, x2, y2 = map(int, last_bbox)

                if tr.get('violation_type') == 'red_light_violation':
                    color = COLOR_RED
                elif tr.get('violation_type') == 'stop_line_violation':
                    color = COLOR_ORANGE
                else:
                    color = COLOR_GREEN

                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis, f"ID:{tid}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)
                bx, by = int((x1 + x2) / 2), int(y2)
                cv2.circle(vis, (bx, by), 3, color, -1)

            for ev in violations_on_frame:
                x1, y1, x2, y2 = map(int, ev['bbox'])
                if ev['type'] == 'stop_line_violation':
                    cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_ORANGE, 3)
                    cv2.putText(
                        vis, f"STOP VIOLATION ID:{ev['track_id']}", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_ORANGE, 2,
                    )
                elif ev['type'] == 'red_light_violation':
                    cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_RED, 3)
                    cv2.putText(
                        vis, f"RED LIGHT VIOLATION ID:{ev['track_id']}", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2,
                    )

            draw_info(vis, stop_line, red_light_line, initial_light_state, violations_on_frame)

            writer.write(vis)
            frame_idx += 1

        writer.release()
        cap.release()

        _transcode_to_h264(output_video_path)

        results = {
            "success": True,
            "annotated_video_path": os.path.abspath(output_video_path),
            "annotated_video_url": f"/evidence/{os.path.basename(output_video_path)}",
            "violations": [],
        }

        for v in violations_log:
            results["violations"].append({
                "track_id": int(v["track_id"]),
                "frame_idx": int(v["frame_idx"]),
                "bbox": [float(x) for x in v["bbox"]],
                "prev_point": [float(x) for x in v["prev_point"]],
                "curr_point": [float(x) for x in v["curr_point"]],
                "light_state": v.get("light_state", "red"),
                "type": v.get("type", "stop_line_violation"),
            })

        json_name = f"{output_basename}.json"
        json_path = os.path.join(Config.EVIDENCE_FOLDER, json_name)
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

        results["results_json_path"] = os.path.abspath(json_path)
        results["results_json_url"] = f"/evidence/{json_name}"

        return results

    except Exception as e:
        try:
            writer.release()
        except Exception:
            pass
        try:
            cap.release()
        except Exception:
            pass
        return {"success": False, "error": str(e)}
