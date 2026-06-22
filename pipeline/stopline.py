"""
Headless stop-line video processing helper.
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

        # transcode failed; clean up and keep original
        if os.path.exists(tmp_out):
            os.remove(tmp_out)
        print(f"[stopline] ffmpeg transcode failed (rc={proc.returncode}): {proc.stderr.decode(errors='ignore')[:500]}")
        return False
    except Exception as e:
        print(f"[stopline] H.264 transcode skipped: {e}")
        return False

# --- Minimal geometry / tracker / detector (adapted from script) ---

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
                    'history': []
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
    def __init__(self, stop_line: Tuple[Point, Point], max_track_distance=80):
        self.line_p1, self.line_p2 = stop_line
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
            if track.get('violated', False):
                continue
            crossed_segment = segments_intersect(prev_pt, curr_pt, self.line_p1, self.line_p2)
            prev_side = point_side_of_line(prev_pt, self.line_p1, self.line_p2)
            curr_side = point_side_of_line(curr_pt, self.line_p1, self.line_p2)
            side_changed = prev_side * curr_side < 0
            if crossed_segment or side_changed:
                if state == 'red':
                    ev = {
                        'track_id': tid,
                        'frame_idx': frame_idx,
                        'bbox': det['bbox'],
                        'prev_point': prev_pt,
                        'curr_point': curr_pt,
                        'light_state': state,
                        'type': 'stop_line_violation'
                    }
                    self.violations[tid] = ev
                    track['violated'] = True
                    new_violations.append(ev)
                else:
                    track['violated'] = False
        return new_violations


# --- Video processing function ---

def process_video_headless(
    video_path: str,
    model_path: Optional[str] = None,
    stop_line_norm: Optional[Tuple[float, float, float, float]] = None,
    initial_light_state: str = 'red',
    conf_thres: float = 0.3,
    output_basename: Optional[str] = None,
) -> Dict:
    """
    Process a video file headlessly: run object detection per frame, track objects,
    detect stop-line crossings, and write an annotated output video plus a JSON results file.

    stop_line_norm: normalized coords [x1,y1,x2,y2] in 0..1 relative to frame width/height.
    Returns a dict with keys: success, annotated_video_path (absolute), violations (list), results_json_path
    """
    os.makedirs(Config.EVIDENCE_FOLDER, exist_ok=True)
    model_manager = ModelManager()

    # Load model (use provided path or config 'uvh' as default)
    if model_path:
        try:
            model = YOLO = None
            # Use Ultralytics YOLO directly
            from ultralytics import YOLO as _YOLO
            model = _YOLO(model_path)
        except Exception as e:
            return {"success": False, "error": f"Failed to load model: {e}"}
    else:
        # default to uvh
        model = model_manager.load_uvh()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"success": False, "error": f"Failed to open video: {video_path}"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # compute absolute stop line if provided
    if stop_line_norm is not None:
        sx1 = int(stop_line_norm[0] * width)
        sy1 = int(stop_line_norm[1] * height)
        sx2 = int(stop_line_norm[2] * width)
        sy2 = int(stop_line_norm[3] * height)
        stop_line = ((sx1, sy1), (sx2, sy2))
    else:
        # default horizontal line near 70% height
        stop_line = ((0, int(height * 0.7)), (width, int(height * 0.7)))

    detector = StopLineViolationDetector(stop_line, max_track_distance=80)

    # prepare output video writer
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

            # run model on frame
            try:
                results = model(frame)
                # extract boxes with confidence
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
                else:
                    bboxes = []
            except Exception as e:
                # model inference failed for this frame; continue
                bboxes = []

            detections = [{'bbox': tuple(map(float, b))} for b in bboxes]

            # process frame through detector (constant light state)
            new_violations = detector.process_frame(detections, initial_light_state, frame_idx)
            if new_violations:
                violations_log.extend(new_violations)

            # annotate frame: draw track boxes with colors
            vis = frame.copy()
            # draw stop line
            cv2.line(vis, (int(stop_line[0][0]), int(stop_line[0][1])), (int(stop_line[1][0]), int(stop_line[1][1])), (0,255,0), 2)

            for tid, tr in detector.tracker.tracks.items():
                if not tr.get('history'):
                    continue
                _, last_bbox, _ = tr['history'][-1]
                x1, y1, x2, y2 = map(int, last_bbox)
                color = (0,0,255) if tr.get('violated', False) else (0,200,255)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis, f"ID:{tid}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # write frame
            writer.write(vis)
            frame_idx += 1

        writer.release()
        cap.release()

        # Re-encode the mp4v output to browser-playable H.264 so the annotated
        # video can be streamed directly into an HTML5 <video> element.
        _transcode_to_h264(output_video_path)

        # prepare results json
        results = {
            "success": True,
            "annotated_video_path": os.path.abspath(output_video_path),
            "annotated_video_url": f"/evidence/{os.path.basename(output_video_path)}",
            "violations": []
        }

        for v in violations_log:
            # keep bbox as list of floats
            results["violations"].append({
                "track_id": int(v["track_id"]),
                "frame_idx": int(v["frame_idx"]),
                "bbox": [float(x) for x in v["bbox"]],
                "prev_point": [float(x) for x in v["prev_point"]],
                "curr_point": [float(x) for x in v["curr_point"]],
                "light_state": v.get("light_state", "red")
            })

        # save results json
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
