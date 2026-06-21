"""Utility Functions for Traffic Violation Detection System"""

import cv2
import json
import base64
import numpy as np
from datetime import datetime
from pathlib import Path
from .config import Config


# ============ IMAGE CONVERSIONS ============

def image_bytes_to_numpy(image_bytes):
    """
    Convert image bytes (from API upload) to numpy array (BGR).
    
    Args:
        image_bytes: Image data as bytes
    
    Returns:
        Numpy array (BGR) or None if conversion fails
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error converting image bytes: {e}")
        return None


def numpy_to_jpeg_bytes(image_array):
    """
    Convert numpy array to JPEG bytes.
    
    Args:
        image_array: Numpy array (BGR)
    
    Returns:
        JPEG bytes or None if conversion fails
    """
    try:
        _, buffer = cv2.imencode('.jpg', image_array)
        return buffer.tobytes()
    except Exception as e:
        print(f"Error converting numpy to JPEG: {e}")
        return None


def numpy_to_base64(image_array):
    """
    Convert numpy array to base64 encoded JPEG string.
    
    Args:
        image_array: Numpy array (BGR)
    
    Returns:
        Base64 string
    """
    try:
        jpeg_bytes = numpy_to_jpeg_bytes(image_array)
        if jpeg_bytes:
            return base64.b64encode(jpeg_bytes).decode('utf-8')
    except Exception as e:
        print(f"Error converting to base64: {e}")
    return None


# ============ BOUNDING BOX SCALING ============

def denormalize_bboxes(bboxes, original_shape, model_shape):
    """
    Scale bounding box coordinates from model input size back to original image size.
    
    Models run on 1088x1088, but original images may be different sizes.
    This function scales bbox coordinates accordingly.
    
    Args:
        bboxes: List of bboxes [[x1, y1, x2, y2], ...]
        original_shape: Original image shape (H, W, C)
        model_shape: Model input shape (default 1088)
    
    Returns:
        Denormalized bboxes in original image coordinates
    """
    try:
        orig_h, orig_w = original_shape[:2]
        
        denormalized = []
        for bbox in bboxes:
            if len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]
                
                # Scale coordinates
                x1_scaled = int((x1 / model_shape) * orig_w)
                y1_scaled = int((y1 / model_shape) * orig_h)
                x2_scaled = int((x2 / model_shape) * orig_w)
                y2_scaled = int((y2 / model_shape) * orig_h)
                
                # Clamp to image bounds
                x1_scaled = max(0, min(x1_scaled, orig_w))
                y1_scaled = max(0, min(y1_scaled, orig_h))
                x2_scaled = max(0, min(x2_scaled, orig_w))
                y2_scaled = max(0, min(y2_scaled, orig_h))
                
                denormalized.append([x1_scaled, y1_scaled, x2_scaled, y2_scaled])
            else:
                denormalized.append(bbox)
        
        return denormalized
    except Exception as e:
        print(f"Error denormalizing bboxes: {e}")
        return bboxes


# ============ BOUNDING BOX DRAWING ============

def draw_bounding_boxes(image, detections, detection_type="vehicle"):
    """
    Draw bounding boxes on image with labels.
    
    Args:
        image: Input image (BGR)
        detections: List of detection dicts with 'box', 'label', 'confidence'
        detection_type: Type of detection ("vehicle", "plate", "violation")
    
    Returns:
        Image with drawn boxes
    """
    try:
        annotated = image.copy()
        
        # Select color based on detection type
        color = Config.ANNOTATION_COLORS.get(detection_type, (255, 255, 255))
        
        # Set text color to black (0, 0, 0) for high contrast against light backgrounds
        # Or switch dynamically if the background color is dark
        text_color = (0, 0, 0) if sum(color) > 382 else (255, 255, 255)
        
        for detection in detections:
            try:
                box = detection.get("box")
                if not box or len(box) < 4:
                    continue
                
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                
                # Draw rectangle
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # Prepare label
                label = detection.get("label", "")
                confidence = detection.get("confidence", 0)
                if confidence:
                    label += f" {confidence:.2f}"
                
                if label:
                    # Draw text background
                    text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    
                    # Correctly calculate background bounds to prevent overlap issues
                    cv2.rectangle(
                        annotated,
                        (x1, y1 - text_size[1] - baseline - 5),
                        (x1 + text_size[0], y1),
                        color,
                        -1
                    )
                    # Draw high-contrast text on top of the background block
                    cv2.putText(
                        annotated,
                        label,
                        (x1, y1 - baseline),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        text_color,
                        1,
                        cv2.LINE_AA
                    )
            except Exception as e:
                continue
        
        return annotated
    except Exception as e:
        print(f"Error drawing bboxes: {e}")
        return image


# ============ 3-WHEELER DETECTION ============

def detect_3wheeler(vehicle_boxes, class_labels):
    """
    Check if 3-wheeler is detected in vehicle boxes.
    
    Args:
        vehicle_boxes: List of vehicle detections
        class_labels: List of class name strings for each box
    
    Returns:
        True if 3-wheeler detected, False otherwise
    """
    try:
        for class_label in class_labels:
            if class_label and class_label.lower() in Config.THREEWHEELER_CLASSES:
                return True
        return False
    except Exception as e:
        return False


# ============ INPUT VALIDATION ============

def validate_gps_timestamp(timestamp_str, gps_tuple):
    """
    Validate GPS coordinates and ISO timestamp format.
    
    Args:
        timestamp_str: ISO format timestamp string (e.g., "2025-06-18T14:30:22.123Z")
        gps_tuple: Tuple of (latitude, longitude)
    
    Returns:
        Tuple of (is_valid: bool, error_message: str or None)
    
    Raises:
        ValueError: If validation fails
    """
    try:
        # Validate timestamp format
        if not isinstance(timestamp_str, str):
            raise ValueError("Timestamp must be a string")
        
        try:
            datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except Exception:
            raise ValueError(f"Invalid ISO timestamp format: {timestamp_str}")
        
        # Validate GPS coordinates
        if not isinstance(gps_tuple, (list, tuple)) or len(gps_tuple) != 2:
            raise ValueError("GPS must be a tuple/list of (latitude, longitude)")
        
        lat, lon = gps_tuple
        
        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            raise ValueError("GPS coordinates must be numeric")
        
        if not (Config.GPS_LAT_MIN <= lat <= Config.GPS_LAT_MAX):
            raise ValueError(f"Latitude must be between {Config.GPS_LAT_MIN} and {Config.GPS_LAT_MAX}, got {lat}")
        
        if not (Config.GPS_LON_MIN <= lon <= Config.GPS_LON_MAX):
            raise ValueError(f"Longitude must be between {Config.GPS_LON_MIN} and {Config.GPS_LON_MAX}, got {lon}")
        
        return True, None
    
    except ValueError as e:
        return False, str(e)


def box_center_inside(inner_box, outer_box):
    """True when the center of inner_box lies inside outer_box."""
    if not inner_box or not outer_box or len(inner_box) < 4 or len(outer_box) < 4:
        return False
    cx = (inner_box[0] + inner_box[2]) / 2.0
    cy = (inner_box[1] + inner_box[3]) / 2.0
    return outer_box[0] <= cx <= outer_box[2] and outer_box[1] <= cy <= outer_box[3]


def count_boxes_inside(outer_box, detections):
    """Count detection boxes whose center falls inside outer_box."""
    count = 0
    for detection in detections:
        box = detection.get("box") if isinstance(detection, dict) else detection
        if box and box_center_inside(box, outer_box):
            count += 1
    return count


# ============ EVIDENCE FILE HANDLING ============

def get_evidence_filename(timestamp_str):
    """
    Generate evidence filename from timestamp with millisecond precision.
    
    Args:
        timestamp_str: ISO format timestamp
    
    Returns:
        Filename string (without extension)
    """
    try:
        # Parse ISO timestamp and convert to filename format
        # 2025-06-18T14:30:22.123Z → 2025-06-18T14-30-22.123Z
        filename = timestamp_str.replace(':', '-').replace('Z', 'Z')
        return filename
    except Exception:
        # Fallback: use current timestamp with milliseconds
        return datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S.%f")[:-3] + "Z"


def save_json_evidence(evidence_dict, evidence_folder=None):
    """
    Save evidence JSON to file with timestamp-based filename.
    
    Args:
        evidence_dict: Evidence data dictionary
        evidence_folder: Folder path (default from config)
    
    Returns:
        Tuple of (success: bool, file_path: str)
    """
    try:
        if evidence_folder is None:
            evidence_folder = Config.EVIDENCE_FOLDER
        
        Path(evidence_folder).mkdir(parents=True, exist_ok=True)
        
        timestamp = evidence_dict.get("timestamp", datetime.utcnow().isoformat() + "Z")
        filename = get_evidence_filename(timestamp) + ".json"
        file_path = Path(evidence_folder) / filename
        
        with open(file_path, 'w') as f:
            json.dump(evidence_dict, f, indent=2)
        
        return True, str(file_path)
    
    except Exception as e:
        print(f"Error saving evidence JSON: {e}")
        return False, None


def save_annotated_image(image_array, timestamp_str, evidence_folder=None):
    """
    Save annotated image to file with timestamp-based filename.
    
    Args:
        image_array: Numpy array (BGR)
        timestamp_str: ISO format timestamp
        evidence_folder: Folder path (default from config)
    
    Returns:
        Tuple of (success: bool, file_path: str)
    """
    try:
        if evidence_folder is None:
            evidence_folder = Config.EVIDENCE_FOLDER
        
        Path(evidence_folder).mkdir(parents=True, exist_ok=True)
        
        filename = get_evidence_filename(timestamp_str) + ".jpg"
        file_path = Path(evidence_folder) / filename
        
        cv2.imwrite(str(file_path), image_array)
        
        return True, str(file_path)
    
    except Exception as e:
        print(f"Error saving annotated image: {e}")
        return False, None


def load_all_evidence(evidence_folder=None):
    """
    Load all evidence JSON files from folder.
    
    Args:
        evidence_folder: Folder path (default from config)
    
    Returns:
        List of evidence dictionaries
    """
    try:
        if evidence_folder is None:
            evidence_folder = Config.EVIDENCE_FOLDER
        
        evidence_list = []
        evidence_path = Path(evidence_folder)
        
        if not evidence_path.exists():
            return evidence_list
        
        for json_file in sorted(evidence_path.glob("*.json"), reverse=True):
            try:
                with open(json_file, 'r') as f:
                    evidence = json.load(f)
                    evidence_list.append(evidence)
            except Exception as e:
                continue
        
        return evidence_list
    
    except Exception as e:
        print(f"Error loading evidence: {e}")
        return []
