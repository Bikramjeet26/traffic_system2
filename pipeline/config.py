"""Configuration for Traffic Violation Detection System"""

import os

class Config:
    """Centralized configuration for all models, paths, and thresholds"""
    
    # ============ MODEL PATHS ============
    BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    MODEL_PATHS = {
        "bdd100k": os.path.join(BASE_PATH, "bdd100k_opensource.pt"),
        "uvh": os.path.join(BASE_PATH, "UVH-26-MV-YOLOv11-S.pt"),
        "license_plate": os.path.join(BASE_PATH, "License_plate_bb.pt"),
        "helmet": os.path.join(BASE_PATH, "helmet_detection.pt"),
        "seatbelt": os.path.join(BASE_PATH, "seatbelt_detection.pt"),
    }
    
    # ============ FOLDER PATHS ============
    EVIDENCE_FOLDER = os.path.join(BASE_PATH, "evidence")
    
    
    # ============ MODEL INFERENCE PARAMETERS ============
    MODEL_INPUT_SIZE = 1088  # YOLO input size (1088x1088)
    
    # Confidence thresholds per model
    CONFIDENCE_THRESHOLDS = {
        "bdd100k": 0.45,
        "uvh": 0.45,
        "license_plate": 0.4,
        "helmet": 0.4,
        "seatbelt": 0.4,
    }
    
    # ============ VEHICLE CLASS MAPPING (BDD100K) ============
    VEHICLE_CLASSES = {
        "car": 0,
        "bike": 1,
        "truck": 2,
        "pedestrian": 3,
        "rider": 4,
        "bus": 5,
        "train": 6
    }
    
    # Map for detecting 3-wheelers (if BDD100K has this class)
    THREEWHEELER_CLASSES = ["three_wheeler"]

    UVH_TRIGGER_VEHICLES = ["mini_bus", "tempo_traveller"]
    
    # ============ VIOLATION CLASS FILTERS ============
    VIOLATION_CLASSES = {
        "helmet": "Without Helmet",           # Helmet model: filter for 'no-helmet'
        "seatbelt": "person-noseatbelt",  # Seatbelt model: filter for 'person-noseatbelt'
    }
    
    # ============ VEHICLE TYPE -> VIOLATION MAPPING ============
    VEHICLE_VIOLATIONS = {
        "motorcycle": ["helmet"],           # Check helmet for 2-wheelers
        "two_wheeler": ["helmet"],
        "car": ["seatbelt"],                # Check seatbelt for cars
        "truck": ["seatbelt"],
    }
    
    # ============ GPS VALIDATION BOUNDS ============
    GPS_LAT_MIN = -90.0
    GPS_LAT_MAX = 90.0
    GPS_LON_MIN = -180.0
    GPS_LON_MAX = 180.0
    
    # ============ PREPROCESSING PARAMETERS ============
    PREPROCESSING_PARAMS = {
        "undistort": {
            "alpha": 0,  # Retain original frame size
        },
        "enhance_lighting": {
            "clipLimit": 1.3,  # CLAHE clip limit
            "tileGridSize": (8, 8),
        },
        "reduce_motion_blur": {
            "kernel": (5, 5),
            "sigma": 1.0,
        },
        "dehaze": {
            "omega": 0.95,
            "t0": 0.1,
        },
        "roi_mask": {
            "top_percentage": 0.2,  # Mask top 20% of frame
        }
    }
    
    # ============ OCR PARAMETERS ============
    OCR_PARAMS = {
        "white_border_px": 12,  # White border padding for OCR
        "text_uppercase": True,
        "text_alphanumeric_only": True,
    }
    
    # ============ ANNOTATION PARAMETERS ============
    ANNOTATION_COLORS = {
        "vehicle": (0, 255, 0),      # Green (BGR)
        "plate": (0, 255, 255),      # Yellow (BGR)
        "violation": (0, 0, 255),    # Red (BGR)
    }
    
    # ============ API PARAMETERS ============
    API_HOST = "0.0.0.0"
    API_PORT = int(os.getenv("PORT", os.getenv("API_PORT", "7860")))
    API_WORKERS = 1

    # Comma-separated list of allowed browser origins (Vercel frontend, local dev)
    CORS_ORIGINS = [
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000",
        ).split(",")
        if origin.strip()
    ]
    # Allow any *.vercel.app preview/production deployment
    CORS_ORIGIN_REGEX = os.getenv(
        "CORS_ORIGIN_REGEX",
        r"https://.*\.vercel\.app",
    )

    @classmethod
    def ensure_folders_exist(cls):
        """Ensure all required folders exist"""
        os.makedirs(cls.EVIDENCE_FOLDER, exist_ok=True)
