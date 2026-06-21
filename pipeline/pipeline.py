"""Main Traffic Violation Pipeline Class"""

import cv2
import numpy as np
from datetime import datetime
from .config import Config
from .models import ModelManager
from .preprocessing import preprocess_frame
from .ocr_processor import OCRProcessor

from .utils import (
    image_bytes_to_numpy, numpy_to_jpeg_bytes, numpy_to_base64,
    denormalize_bboxes, draw_bounding_boxes, detect_3wheeler,
    validate_gps_timestamp, get_evidence_filename,
    save_json_evidence, save_annotated_image, load_all_evidence,
    count_boxes_inside,
)


class TrafficViolationPipeline:
    """
    Complete traffic violation detection pipeline with 9 methods.
    Detects vehicles, license plates, violations (helmet, seatbelt), and optional facial recognition.
    """
    
    def __init__(self):
        """Initialize pipeline components"""
        Config.ensure_folders_exist()
        self.model_manager = ModelManager()
        self.ocr_processor = OCRProcessor()
        
        self.original_shape = None
    
    # ============ METHOD 1: INPUT VALIDATION ============
    
    def _validate_input(self, timestamp, gps):
        """Validate GPS and timestamp before processing"""
        is_valid, error_msg = validate_gps_timestamp(timestamp, gps)
        if not is_valid:
            raise ValueError(f"Input validation failed: {error_msg}")
    
    # ============ METHOD 2: PREPROCESS ============
    
    def preprocess(self, image):
        """
        Preprocess image for model inference.
        
        Args:
            image: Input image (BGR, numpy array)
        
        Returns:
            Tuple of (preprocessed_image, original_image, original_shape)
        """
        try:
            self.original_shape = image.shape
            preprocessed, _ = preprocess_frame(image, apply_dehaze=False)
            return preprocessed, image, self.original_shape
        except Exception as e:
            print(f"Error in preprocess: {e}")
            return image, image, image.shape
    
    # ============ METHOD 3: DETECT VEHICLES ============
    
    def detect_vehicles(self, image):
        """
            Detect vehicles using BDD100K or UVH model.
            Switches to UVH if specific vehicles (3-wheeler, mini-bus, tempo traveller) are detected.

            Args:
                image: Preprocessed image

            Returns:
                List of vehicle detections [{"box": [x1,y1,x2,y2], "class": "car", "confidence": 0.9}, ...]
        """
        try:
            # Run BDD100K first
            model = self.model_manager.load_bdd100k()
            results = model.predict(
                image,
                conf=Config.CONFIDENCE_THRESHOLDS["bdd100k"],
                imgsz=Config.MODEL_INPUT_SIZE,
                verbose=False
            )

            detections = []
            class_labels = []

            if results and len(results) > 0:
                for result in results:
                    if result.boxes is not None and len(result.boxes) > 0:
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            class_id = int(box.cls.cpu().numpy()[0])
                            confidence = float(box.conf.cpu().numpy()[0])

                            # Get class name from model
                            class_name = result.names.get(class_id, f"class_{class_id}")
                            class_labels.append(class_name)

                            detections.append({
                                "box": [float(x1), float(y1), float(x2), float(y2)],
                                "class": class_name,
                                "class_id": class_id,
                                "confidence": confidence
                            })

            # # Check if specific vehicles are detected that require the UVH model
            # switch_to_uvh = False
            # # First, check if a 3-wheeler (or similar) is detected using the utility function
            # if detect_3wheeler(detections, class_labels): # This relies on Config.THREEWHEELER_CLASSES in utils.py
            #     switch_to_uvh = True
            #     print("3-wheeler detected, switching to UVH model...")
            # else:
            #     # Then, check for mini-bus and tempo traveller directly from BDD100K detections
            #     for detection in detections:
            #         if detection.get("class", "").lower() in Config.UVH_TRIGGER_VEHICLES:
            #             switch_to_uvh = True
            #             print(f"'{detection.get('class')}' detected, switching to UVH model...")
            #             break # Exit loop once a trigger is found

            # # If any of the trigger conditions are met, reload and run the UVH model
            # if switch_to_uvh:
            #     model = self.model_manager.load_uvh()
            #     results = model.predict(
            #         image,
            #         conf=Config.CONFIDENCE_THRESHOLDS["uvh"],
            #         imgsz=Config.MODEL_INPUT_SIZE,
            #         verbose=False
            #     )

            #     detections = [] # Reset detections to store UVH model results
            #     if results and len(results) > 0:
            #         for result in results:
            #             if result.boxes is not None and len(result.boxes) > 0:
            #                 for box in result.boxes:
            #                     x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            #                     class_id = int(box.cls.cpu().numpy()[0])
            #                     confidence = float(box.conf.cpu().numpy()[0])

            #                     class_name = result.names.get(class_id, f"class_{class_id}")

            #                     detections.append({
            #                         "box": [float(x1), float(y1), float(x2), float(y2)],
            #                         "class": class_name,
            #                         "class_id": class_id,
            #                         "confidence": confidence
            #                     })

            
            return detections

        except Exception as e:
            print(f"Error detecting vehicles: {e}")
            return []
    
    # ============ METHOD 4: DETECT LICENSE PLATES ============
    
    def detect_license_plates(self, image, vehicle_boxes=None):
        """
        Detect license plates directly on the entire original frame.
        Cropping and vehicle-class prerequisites have been removed.
        """
        try:
            plate_detections = []
            model = self.model_manager.load_license_plate()
            
            # Run plate detection across the full global frame dimensions
            print(Config.MODEL_INPUT_SIZE)
            results = model.predict(
                image,
                conf=Config.CONFIDENCE_THRESHOLDS["license_plate"],
                # imgzs=Config.MODEL_INPUT_SIZE,
                verbose=False
            )
            # results = model(image)[0]
            # print("printing no of license plates recognized: ")
            # print(len(results))
            
            if results and len(results) > 0:
                for result in results:
                    if result.boxes is not None and len(result.boxes) > 0:
                        for box_obj in result.boxes:
                            px1, py1, px2, py2 = box_obj.xyxy[0].cpu().numpy()
                            confidence = float(box_obj.conf.cpu().numpy()[0])
                            print([float(px1), float(py1), float(px2), float(py2)])
                            plate_detections.append({
                                "box": [float(px1), float(py1), float(px2), float(py2)],
                                "confidence": confidence,
                                "vehicle_idx": None  # Global detection mode unlinked from standard indexes
                            })
            
            return plate_detections
        except Exception as e:
            print(f"Error detecting license plates globally: {e}")
            
            return []
    
    # ============ METHOD 5: EXTRACT PLATE TEXT ============
    
    def extract_plate_text(self, image, plate_boxes):
        """
        Extract text from detected license plates using RapidOCR.
        
        Args:
            image: Original image
            plate_boxes: List of plate bounding boxes
        
        Returns:
            List of plate texts [{"box": [...], "text": "AB1234CD", "confidence": 0.92}, ...]
        """
        try:
            
            plate_text = self.ocr_processor.extract_plate_text(image, plate_boxes)
            
            return plate_text
        except Exception as e:
            print(f"Error extracting plate text: {e}")
            return []
    
    # # ============ METHOD 6: DETECT VIOLATIONS ============
    
    # def detect_violations(self, image, vehicle_boxes):
    #     """
    #     Detect traffic violations: helmet non-compliance, seatbelt non-compliance.
    #     Filters for violation classes only.
        
    #     Args:
    #         image: Original image
    #         vehicle_boxes: List of vehicle bounding boxes with class info
        
    #     Returns:
    #         List of violations [{"type": "no_helmet", "box": [...], "confidence": 0.87, "vehicle_class": "motorcycle"}, ...]
    #     """
    #     violations = []
        
    #     for vehicle_idx, vehicle_box in enumerate(vehicle_boxes):
    #         try:
    #             vehicle_class = vehicle_box.get("class", "").lower()
    #             box = vehicle_box.get("box")
                
    #             if not box or len(box) < 4:
    #                 continue
                
    #             x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                
    #             # Crop vehicle region
    #             vehicle_crop = image[y1:y2, x1:x2]
                
    #             if vehicle_crop.size == 0:
    #                 continue
                
    #             # ===== CHECK HELMET FOR 2-WHEELERS =====
    #             if vehicle_class in ["motorcycle", "two_wheeler", "bike"]:
    #                 try:
    #                     model = self.model_manager.load_helmet()
    #                     results = model.predict(
    #                         vehicle_crop,
    #                         conf=Config.CONFIDENCE_THRESHOLDS["helmet"],
    #                         # imgsz=Config.MODEL_INPUT_SIZE,
    #                         verbose=False
    #                     )
                        
    #                     if results and len(results) > 0:
    #                         for result in results:
    #                             if result.boxes is not None and len(result.boxes) > 0:
    #                                 for box_obj in result.boxes:
    #                                     class_id = int(box_obj.cls.cpu().numpy()[0])
    #                                     class_name = result.names.get(class_id, f"class_{class_id}")
    #                                     confidence = float(box_obj.conf.cpu().numpy()[0])
                                        
    #                                     # Filter for violation class only
    #                                     if class_name.lower() == Config.VIOLATION_CLASSES["helmet"].lower():
    #                                         hx1, hy1, hx2, hy2 = box_obj.xyxy[0].cpu().numpy()
                                            
    #                                         # Denormalize back to original image
    #                                         abs_hx1 = x1 + int(hx1)
    #                                         abs_hy1 = y1 + int(hy1)
    #                                         abs_hx2 = x1 + int(hx2)
    #                                         abs_hy2 = y1 + int(hy2)
                                            
    #                                         violations.append({
    #                                             "type": "no_helmet",
    #                                             "box": [float(abs_hx1), float(abs_hy1), float(abs_hx2), float(abs_hy2)],
    #                                             "confidence": confidence,
    #                                             "vehicle_class": vehicle_class,
    #                                             "vehicle_idx": vehicle_idx
    #                                         })
                    
    #                 except Exception as e:
    #                     continue
                
    #             # ===== CHECK SEATBELT FOR CARS/TRUCKS =====
    #             if vehicle_class in ["car", "truck"]:
    #                 try:
    #                     model = self.model_manager.load_seatbelt()
    #                     results = model.predict(
    #                         vehicle_crop,
    #                         conf=Config.CONFIDENCE_THRESHOLDS["seatbelt"],
    #                         # imgsz=Config.MODEL_INPUT_SIZE,
    #                         verbose=False
    #                     )
                        
    #                     if results and len(results) > 0:
    #                         for result in results:
    #                             if result.boxes is not None and len(result.boxes) > 0:
    #                                 for box_obj in result.boxes:
    #                                     class_id = int(box_obj.cls.cpu().numpy()[0])
    #                                     class_name = result.names.get(class_id, f"class_{class_id}")
    #                                     confidence = float(box_obj.conf.cpu().numpy()[0])
                                        
    #                                     # Filter for violation class only
    #                                     if class_name.lower() == Config.VIOLATION_CLASSES["seatbelt"].lower():
    #                                         sx1, sy1, sx2, sy2 = box_obj.xyxy[0].cpu().numpy()
                                            
    #                                         # Denormalize back to original image
    #                                         abs_sx1 = x1 + int(sx1)
    #                                         abs_sy1 = y1 + int(sy1)
    #                                         abs_sx2 = x1 + int(sx2)
    #                                         abs_sy2 = y1 + int(sy2)
                                            
    #                                         violations.append({
    #                                             "type": "no_seatbelt",
    #                                             "box": [float(abs_sx1), float(abs_sy1), float(abs_sx2), float(abs_sy2)],
    #                                             "confidence": confidence,
    #                                             "vehicle_class": vehicle_class,
    #                                             "vehicle_idx": vehicle_idx
    #                                         })
                    
    #                 except Exception as e:
    #                     continue
            
    #         except Exception as e:
    #             continue
    #     print("violations is working")
    #     print(f"detected violation: {violations}")
    #     return violations
    def detect_violations(self, image, vehicle_boxes=None):
        """
        Detect traffic violations globally across the entire original frame.
        Cropping and vehicle-class prerequisites have been removed.
        """
        violations = []
        
        # ===== GLOBAL HELMET DETECTION =====
        try:
            helmet_model = self.model_manager.load_helmet()
            # Run prediction on the FULL original image
            helmet_results = helmet_model.predict(
                image,
                conf=Config.CONFIDENCE_THRESHOLDS["helmet"],
                verbose=False
            )
            
            if helmet_results and len(helmet_results) > 0 and helmet_results[0].boxes is not None:
                for box_obj in helmet_results[0].boxes:
                    class_id = int(box_obj.cls.cpu().numpy()[0])
                    class_name = helmet_results[0].names.get(class_id, f"class_{class_id}")
                    conf = float(box_obj.conf.cpu().numpy()[0])
                    
                    # Filter for violation class only
                    if class_name.lower() == Config.VIOLATION_CLASSES["helmet"].lower():
                        # Coordinates are already absolute to the full image!
                        hx1, hy1, hx2, hy2 = box_obj.xyxy[0].cpu().numpy()
                        
                        violations.append({
                            "type": "no_helmet",
                            "box": [float(hx1), float(hy1), float(hx2), float(hy2)],
                            "confidence": conf,
                            "vehicle_class": "unknown", # Global detection unlinked from specific vehicles
                            "vehicle_idx": None
                        })
        except Exception as e:
            print(f"Error in global helmet detection: {e}")

        # ===== GLOBAL SEATBELT DETECTION =====
        try:
            seatbelt_model = self.model_manager.load_seatbelt()
            # Run prediction on the FULL original image
            seatbelt_results = seatbelt_model.predict(
                image,
                conf=Config.CONFIDENCE_THRESHOLDS["seatbelt"],
                verbose=False
            )
            
            if seatbelt_results and len(seatbelt_results) > 0 and seatbelt_results[0].boxes is not None:
                for box_obj in seatbelt_results[0].boxes:
                    class_id = int(box_obj.cls.cpu().numpy()[0])
                    class_name = seatbelt_results[0].names.get(class_id, f"class_{class_id}")
                    conf = float(box_obj.conf.cpu().numpy()[0])
                    
                    # Filter for violation class only
                    if class_name.lower() == Config.VIOLATION_CLASSES["seatbelt"].lower():
                        # Coordinates are already absolute to the full image!
                        sx1, sy1, sx2, sy2 = box_obj.xyxy[0].cpu().numpy()
                        
                        violations.append({
                            "type": "no_seatbelt",
                            "box": [float(sx1), float(sy1), float(sx2), float(sy2)],
                            "confidence": conf,
                            "vehicle_class": "unknown", # Global detection unlinked from specific vehicles
                            "vehicle_idx": None
                        })
        except Exception as e:
            print(f"Error in global seatbelt detection: {e}")

        # ===== TRIPLE RIDING ON TWO-WHEELERS =====
        violations.extend(self.detect_triple_riding(image, vehicle_boxes))

        return violations

    def detect_triple_riding(self, image, vehicle_boxes=None):
        """
        Flag triple riding when 3+ persons are detected inside a two-wheeler bounding box.
        Uses all helmet-model detections (any class) as person proxies on the vehicle.
        """
        if not vehicle_boxes:
            return []

        two_wheelers = [
            v for v in vehicle_boxes
            if v.get("class", "").lower() in Config.TWO_WHEELER_CLASSES
        ]
        if not two_wheelers:
            return []

        person_detections = []
        try:
            helmet_model = self.model_manager.load_helmet()
            results = helmet_model.predict(
                image,
                conf=Config.CONFIDENCE_THRESHOLDS["helmet"],
                verbose=False,
            )
            if results and len(results) > 0 and results[0].boxes is not None:
                for box_obj in results[0].boxes:
                    x1, y1, x2, y2 = box_obj.xyxy[0].cpu().numpy()
                    person_detections.append({
                        "box": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": float(box_obj.conf.cpu().numpy()[0]),
                    })
        except Exception as e:
            print(f"Error detecting persons for triple riding: {e}")
            return []

        violations = []
        for vehicle in two_wheelers:
            vbox = vehicle.get("box")
            if not vbox or len(vbox) < 4:
                continue

            person_count = count_boxes_inside(vbox, person_detections)
            if person_count >= Config.TRIPLE_RIDING_MIN_PERSONS:
                violations.append({
                    "type": "triple_riding",
                    "box": [float(x) for x in vbox],
                    "confidence": float(vehicle.get("confidence", 0.0)),
                    "vehicle_class": vehicle.get("class", "unknown"),
                    "vehicle_idx": None,
                    "person_count": person_count,
                })

        return violations
    
    # ============ METHOD 7: ANNOTATE IMAGE ============
    
    def annotate_image(self, original_image, vehicle_boxes, plate_results, violations):
        """
        Draw bounding boxes and labels on image.
        Color-coded: green (vehicles), yellow (plates), red (violations).
        
        Args:
            original_image: Original image (BGR)
            vehicle_boxes: Vehicle detections
            plate_results: License plate text results
            violations: Traffic violations
        
        Returns:
            Annotated image (numpy array)
        """
        try:
            annotated = original_image.copy()
            
            # Draw vehicles (green)
            vehicle_detections = [
                {
                    "box": v["box"],
                    "label": f"{v['class']}",
                    "confidence": v["confidence"]
                }
                for v in vehicle_boxes
            ]
            annotated = draw_bounding_boxes(annotated, vehicle_detections, "vehicle")
            
            # Draw license plates (yellow)
            plate_detections = [
                {
                    "box": p["box"],
                    "label": f"Plate: {p['text']}",
                    "confidence": p["confidence"]
                }
                for p in plate_results
            ]
            annotated = draw_bounding_boxes(annotated, plate_detections, "plate")
            
            # Draw violations (red)
            violation_detections = [
                {
                    "box": v["box"],
                    "label": f"{v['type'].upper()}",
                    "confidence": v["confidence"]
                }
                for v in violations
            ]
            annotated = draw_bounding_boxes(annotated, violation_detections, "violation")
            
            return annotated
        
        except Exception as e:
            print(f"Error annotating image: {e}")
            return original_image
    
    # ============ METHOD 8: GENERATE EVIDENCE ============
    
    def generate_evidence(self, annotated_image, plate_results, all_violations, timestamp, gps):
        """
        Build evidence JSON record and save annotated image.
        
        Args:
            annotated_image: Annotated image (numpy array)
            plate_results: License plate text results
            all_violations: Traffic violations detected
            timestamp: ISO timestamp string
            gps: GPS tuple (lat, lon)
        
        Returns:
            Tuple of (evidence_json, image_path)
        """
        try:
            # Save annotated image
            success, image_path = save_annotated_image(annotated_image, timestamp)
            
            if not success:
                image_path = "error_saving_image"
            
            # Build evidence JSON
            evidence = {
                "timestamp": timestamp,
                "gps": list(gps) if isinstance(gps, tuple) else gps,
                "plates": [
                    {
                        "text": p["text"],
                        "confidence": float(p["confidence"]),
                        "box": [float(x) for x in p["box"]]
                    }
                    for p in plate_results
                ],
                "violations": [
                    {
                        "type": v["type"],
                        "vehicle_class": v.get("vehicle_class", "unknown"),
                        "confidence": float(v["confidence"]),
                        "box": [float(x) for x in v["box"]],
                        **({"person_count": int(v["person_count"])} if "person_count" in v else {}),
                    }
                    for v in all_violations
                ],
                "annotated_image_path": image_path
            }
            
            # Save JSON evidence
            success, json_path = save_json_evidence(evidence)
            
            return evidence, image_path
        
        except Exception as e:
            print(f"Error generating evidence: {e}")
            return {}, None
    
    # ============ METHOD 9: RUN (MASTER ORCHESTRATOR) ============
    
    def run(self, image_bytes, timestamp, gps):
        """
        Master pipeline method: orchestrates all 9 steps.
        
        Args:
            image_bytes: Image data as bytes
            timestamp: ISO format timestamp (e.g., "2025-06-18T14:30:22.123Z")
            gps: Tuple of (latitude, longitude)
        
        Returns:
            Dictionary with evidence and annotated image bytes
            {
                "success": True/False,
                "evidence": {...},
                "annotated_image_bytes": b"...",
                "error": "..." (if any)
            }
        """
        try:
            # Step 0: Validate input
            self._validate_input(timestamp, gps)
            
            # Step 1: Convert image bytes to numpy
            image = image_bytes_to_numpy(image_bytes)
            if image is None:
                return {
                    "success": False,
                    "error": "Failed to decode image bytes",
                    "evidence": None,
                    "annotated_image_bytes": None
                }
            
            # Step 2: Preprocess
            preprocessed_img, original_img, original_shape = self.preprocess(image)
            
            # Step 3: Detect vehicles
            vehicle_boxes = self.detect_vehicles(original_img)
            
            # Step 4: Detect license plates (on cropped regions)
            plate_boxes = self.detect_license_plates(original_img, vehicle_boxes=None)
            
            # Step 5: Extract plate text
            plate_results = self.extract_plate_text(original_img, plate_boxes)
            
            # Step 6: Detect violations
            violations = self.detect_violations(original_img, vehicle_boxes)
            
            # Step 7: Annotate image
            annotated_img = self.annotate_image(original_img, vehicle_boxes, plate_results, violations)
            
            # Step 8: Generate evidence
            evidence, image_path = self.generate_evidence(annotated_img, plate_results, violations, timestamp, gps)
            
            # Convert annotated image to bytes
            annotated_bytes = numpy_to_jpeg_bytes(annotated_img)
            annotated_base64 = numpy_to_base64(annotated_img)
            
            return {
                "success": True,
                "evidence": evidence,
                "annotated_image_bytes": annotated_bytes,
                "annotated_image_base64": annotated_base64
            }
        
        except ValueError as e:
            return {
                "success": False,
                "error": str(e),
                "evidence": None,
                "annotated_image_bytes": None
            }
        except Exception as e:
            print(f"Error in pipeline.run(): {e}")
            return {
                "success": False,
                "error": f"Pipeline error: {str(e)}",
                "evidence": None,
                "annotated_image_bytes": None
            }    
