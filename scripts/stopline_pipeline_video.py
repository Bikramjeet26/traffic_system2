"""
Stop-line violation pipeline (video input):

Usage:
    python stopline_pipeline_video.py [--video "path/to/video.mp4"] [--model "path/to/model.pt"]

Controls while window is open:
 - Left-click twice to set the stop line (pause the video with 'p' if needed).
 - 'p' : pause / resume playback (pause recommended to place the two clicks accurately).
 - 'r' : set light = RED
 - 'g' : set light = GREEN
 - 's' : save snapshot
 - 'c' : clear the stop line (clicks)
 - 'q' : quit

Notes:
 - The default model path is set to the path you provided earlier; override using --model.
 - Requires ultralytics and opencv-python: pip install ultralytics opencv-python
"""

import os
import time
import math
import argparse
from typing import List, Tuple, Dict
import cv2
import numpy as np
import sys

# Try to import Ultralytics YOLO (recommended). If not present, the script will error.
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    ULTRALYTICS_AVAILABLE = False
    YOLO = None

# ------------------ Geometry + simple tracker + detector (same logic as earlier) ------------------
Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]  # x1, y1, x2, y2

def bottom_center(bbox: BBox) -> Point:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    by = y2
    return (cx, by)

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
    if o1 == 0 and on_segment(p1, q1, p2): return True
    if o2 == 0 and on_segment(p1, q2, p2): return True
    if o3 == 0 and on_segment(q1, p1, q2): return True
    if o4 == 0 and on_segment(q1, p2, q2): return True
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

# ------------------ Inference + visualization pipeline ------------------

# Default model path (your provided path). You can override via --model.
DEFAULT_MODEL_PATH = r"C:\Users\satvi\Desktop\Flipkart_GridLock\Round_2\vehicle pedistrain model\vehicle pedistrain model\UVH-26-MV-YOLOv11-S.pt"

# Default video path (change this variable to infer on a different video,
# or pass --video on the command line to override).
DEFAULT_VIDEO_PATH = r"C:\Users\satvi\Videos\Screen Recordings\Screen Recording 2026-06-21 192716.mp4"

# Visualization colours
COLOR_YELLOW = (0, 200, 255)   # non-violator
COLOR_RED = (0, 0, 255)        # violator
COLOR_LINE = (0, 255, 0)
COLOR_TEXT = (255, 255, 255)

def load_detector_model(path):
    if ULTRALYTICS_AVAILABLE:
        print("Loading model with ultralytics.YOLO:", path)
        return YOLO(path)
    else:
        raise RuntimeError("Ultralytics is not installed. Install with: pip install ultralytics")

def extract_bboxes_from_results(results, conf_thres=0.3) -> List[Tuple[float, float, float, float]]:
    dets = []
    if not results:
        return dets
    r = results[0]
    if hasattr(r, 'boxes'):
        boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes.xyxy, 'cpu') else np.array(r.boxes.xyxy)
        confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes.conf, 'cpu') else np.array(r.boxes.conf)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(float).tolist()
            conf = float(confs[i]) if i < len(confs) else 1.0
            if conf < conf_thres:
                continue
            dets.append((x1, y1, x2, y2))
    return dets

# mouse callback to collect two clicks for stop line
clicked_points = []
def mouse_cb(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) >= 2:
            clicked_points = []
        clicked_points.append((x, y))
        print("Clicked:", x, y)

def draw_info(frame, stop_line, light_state, violations_on_frame, paused):
    # draw stop line
    if stop_line is not None:
        p1, p2 = stop_line
        cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), COLOR_LINE, 2)
    # light state
    cv2.putText(frame, f"Light: {light_state.upper()}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_TEXT, 2, cv2.LINE_AA)
    if violations_on_frame:
        cv2.putText(frame, f"VIOLATION!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_RED, 3)
    status = "PAUSED" if paused else "PLAY"
    cv2.putText(frame, f"Status: {status}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2)

def parse_args():
    p = argparse.ArgumentParser(description="Stop-line violation detector on video")
    # video is optional now; default set to DEFAULT_VIDEO_PATH variable
    p.add_argument("--video", default=DEFAULT_VIDEO_PATH, help="Path to input video file (default = DEFAULT_VIDEO_PATH variable)")
    p.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to YOLO .pt model (Ultralytics)")
    p.add_argument("--conf-thres", type=float, default=0.3, help="Detection confidence threshold")
    # Use parse_known_args so this also runs inside notebooks (ignores unknown ipykernel args)
    args, unknown = p.parse_known_args()
    return args

def iou(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    areaA = (ax2 - ax1) * (ay2 - ay1)
    areaB = (bx2 - bx1) * (by2 - by1)
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0

def main():
    global clicked_points
    args = parse_args()
    video_path = args.video
    model_path = args.model
    conf_thres = args.conf_thres

    # Print a startup banner
    print("=" * 60)
    print("Stop-line detector starting")
    print(f"Video: {video_path}")
    print(f"Model: {model_path}")
    print("=" * 60)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = load_detector_model(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video source: " + str(video_path))

    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", mouse_cb)

    print("Controls: 'p' pause/resume, Left-click twice to set stop line (pause recommended), 'r'/'g' to set light, 'c' clear line, 'q' quit")

    stop_line = None
    light_state = 'red'
    detector = None

    frame_idx = 0
    violations_log = []
    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video or cannot read frame.")
                break
        else:
            try:
                frame
            except NameError:
                ret, frame = cap.read()
                if not ret:
                    break

        vis = frame.copy()

        # build stop_line if clicks available
        if len(clicked_points) >= 2:
            stop_line = (clicked_points[0], clicked_points[1])
            if detector is None:
                detector = StopLineViolationDetector(stop_line, max_track_distance=80)
        elif stop_line is None:
            cv2.putText(vis, "Click two points to set stop line (pause with 'p')", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)

        # run model inference and obtain bboxes
        bboxes = []
        try:
            results = model(frame)  # synchronous inference
            all_boxes = extract_bboxes_from_results(results, conf_thres=conf_thres)
            bboxes = all_boxes
        except Exception as e:
            cv2.putText(vis, f"Model inference error: {e}", (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            bboxes = []

        detections = [{'bbox': tuple(map(float, b))} for b in bboxes]

        violations_on_frame = []
        if detector is not None:
            new_violations = detector.process_frame(detections, light_state, frame_idx)
            if new_violations:
                for ev in new_violations:
                    print("Violation:", ev)
                    violations_log.append(ev)
                violations_on_frame = new_violations

        # --- Visualization: persistent coloring based on tracker ---
        # Draw tracked boxes (persist color: red for violator, yellow otherwise)
        track_boxes = []
        if detector is not None:
            for tid, tr in detector.tracker.tracks.items():
                if not tr.get('history'):
                    continue
                _, last_bbox, _ = tr['history'][-1]
                x1, y1, x2, y2 = map(int, last_bbox)
                color = COLOR_RED if tr.get('violated', False) else COLOR_YELLOW
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis, f"ID:{tid}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)
                bx, by = int((x1 + x2) / 2), int(y2)
                cv2.circle(vis, (bx, by), 3, color, -1)
                track_boxes.append(((x1, y1, x2, y2), tid))

        # Draw model detections that are not matched to any existing track in yellow
        for bbox in bboxes:
            bx1, by1, bx2, by2 = map(int, bbox)
            matched = False
            for tb, tid in track_boxes:
                if iou((bx1, by1, bx2, by2), tb) > 0.2:
                    matched = True
                    break
            if not matched:
                cv2.rectangle(vis, (bx1, by1), (bx2, by2), COLOR_YELLOW, 2)
                cx, cy = int((bx1 + bx2) / 2), int(by2)
                cv2.circle(vis, (cx, cy), 3, COLOR_YELLOW, -1)

        # emphasize violations detected on this frame
        for ev in violations_on_frame:
            x1, y1, x2, y2 = map(int, ev['bbox'])
            cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_RED, 3)
            cv2.putText(vis, f"VIOLATION ID:{ev['track_id']}", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)

        draw_info(vis, stop_line, light_state, bool(violations_on_frame), paused)

        cv2.imshow("Frame", vis)
        key = cv2.waitKey(0 if paused else 1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            light_state = 'red'
            print("Light set to RED")
        elif key == ord('g'):
            light_state = 'green'
            print("Light set to GREEN")
        elif key == ord('s'):
            cv2.imwrite(f"snapshot_{int(time.time())}.jpg", vis)
            print("Snapshot saved")
        elif key == ord('p'):
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == ord('c'):
            clicked_points = []
            stop_line = None
            detector = None
            print("Cleared stop line and detector (reinitialize after drawing new line)")

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    # print summary of violators (starred banner if any)
    if detector is not None and detector.violations:
        violator_ids = sorted(detector.violations.keys())
        stars = "*" * 60
        print("\n" + stars)
        print("*** VIOLATORS (IDs):", ", ".join(map(str, violator_ids)), "***")
        print(stars + "\n")
        for tid, ev in detector.violations.items():
            print(f"*** VIOLATOR ID {tid} -> frame {ev['frame_idx']} bbox {ev['bbox']} ***")
    else:
        print("No violations logged.")

if __name__ == "__main__":
    main()
