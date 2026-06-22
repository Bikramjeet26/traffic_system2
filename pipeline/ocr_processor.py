"""OCR Processing Module - License Plate Text Extraction with Strict Validation"""

import cv2
import numpy as np
import re
from rapidocr_onnxruntime import RapidOCR
from pydantic import BaseModel, field_validator, ValidationError
from typing import Optional
from .config import Config

# Valid Indian state codes (29 states + 8 UTs + special categories)
INDIAN_STATE_CODES = {
    "AP", "AR", "AS", "BR", "CT", "GA", "GJ", "HR", "HP", "JH", "JK", "KA", "KL", "LD", "MH",
    "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TG", "TN", "TR", "UP", "WB",
    "AN", "CH", "DD", "DN", "DL", "LA", "TS"
}

# Bidirectional maps for OCR misreads context-aware substitution
ALPHA_TO_DIGIT = {"O": "0", "I": "1", "S": "5", "B": "8", "Z": "2", "G": "6", "T": "1"}
DIGIT_TO_ALPHA = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "G"}


class IndianLicensePlate(BaseModel):
    """Pydantic model for Indian License Plate validation and fuzzy correction"""
    plate_number: str

    @field_validator('plate_number')
    @classmethod
    def validate_plate_format(cls, v: str) -> str:
        """
        Validate and correct Indian license plate format.
        
        Format: State(2) + District(2) + Optional Vehicle Class(1-2 Alphas) + Registration Number(1-4 Digits) + Optional(Alphas)
        
        Features:
        - Sliding window detection for state codes (handles leading noise)
        - OCR misread correction for district digits (O->0, I->1, etc.)
        - Structural regex validation
        """
        # 1. Base Cleanup
        v = "".join([c for c in v if c.isalnum()]).upper()
        
        # 2. Sliding Window Matching for Leading Noise (e.g., 'RKA02...' -> 'KA02...')
        matched_state = None
        start_idx = 0
        
        for i in range(min(3, len(v) - 1)):
            candidate = v[i:i+2]
            # Try applying correction mapping to the candidate characters
            corrected_candidate = "".join([DIGIT_TO_ALPHA.get(c, c) for c in candidate])
            
            if corrected_candidate in INDIAN_STATE_CODES:
                matched_state = corrected_candidate
                start_idx = i
                break
            elif candidate in INDIAN_STATE_CODES:
                matched_state = candidate
                start_idx = i
                break
        
        if matched_state:
            v = v[start_idx:]
            v = matched_state + v[2:]
        else:
            raise ValueError(f"No valid Indian state code found in sequence: {v}")

        if len(v) < 6:
            raise ValueError("License plate is too short. Expected at least 6 characters.")

        # 3. Contextual Correction for District Code (Characters 3 and 4 must be digits)
        district_part = list(v[2:4])
        for idx, char in enumerate(district_part):
            if not char.isdigit() and char in ALPHA_TO_DIGIT:
                district_part[idx] = ALPHA_TO_DIGIT[char]
        v = v[:2] + "".join(district_part) + v[4:]

        if not v[2:4].isdigit():
            raise ValueError(f"Characters 3-4 must be numeric district codes. Got: {v[2:4]}")

        # 4. Final Format Check using Regex
        pattern = r"^[A-Z]{2}[0-9]{2}[A-Z]{0,2}[0-9]{1,4}[A-Z]{0,2}$"
        if not re.match(pattern, v):
            raise ValueError(f"Plate failed structural pattern validation: {v}")

        return v


class OCRProcessor:
    """Handles license plate text extraction using RapidOCR with strict validation"""
    
    def __init__(self):
        """Initialize RapidOCR engine"""
        self.ocr_engine = RapidOCR()
    
    def extract_plate_text(self, image, plate_boxes):
        """
        Extract text from detected license plates with STRICT validation.
        
        ONLY valid plates are added to results.
        Invalid plates are NOT stored (not added to results list).
        
        Args:
            image: Original image (BGR, not preprocessed)
            plate_boxes: List of plate bounding boxes [{"box": [x1,y1,x2,y2], "confidence": 0.88}, ...]
        
        Returns:
            List of VALIDATED extracted texts only
            [{"box": [x1,y1,x2,y2], "text": "KA02ME3547", "confidence": 0.92, "validated": True}, ...]
            
            Invalid detections are silently dropped (not returned).
        """
        results = []
        
        for plate_box in plate_boxes:
            try:
                box = plate_box["box"]
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                
                # Crop plate region from image
                plate_crop = image[y1:y2, x1:x2]
                
                if plate_crop.size == 0:
                    continue
                
                # Add white border padding (prevents edge characters from being cut off)
                border_px = Config.OCR_PARAMS["white_border_px"]
                plate_with_border = cv2.copyMakeBorder(
                    plate_crop,
                    border_px, border_px, border_px, border_px,
                    cv2.BORDER_CONSTANT,
                    value=(255, 255, 255)
                )
                
                # Run RapidOCR on padded crop
                ocr_output, _ = self.ocr_engine(
                    plate_with_border,
                    use_det=False,
                    use_cls=False,
                    use_rec=True,
                )

                if ocr_output:
                    extracted_texts = []

                    for detection in ocr_output:
                        if len(detection) >= 2:
                            if isinstance(detection[0], str):
                                text = detection[0]
                                confidence = float(detection[1])
                            else:
                                text = detection[1]
                                confidence = float(detection[2]) if len(detection) > 2 else 0.0
                            extracted_texts.append((text, confidence))
                    
                    if extracted_texts:
                        # Combine all detected texts
                        combined_text = "".join([text for text, _ in extracted_texts])
                        avg_confidence = np.mean([conf for _, conf in extracted_texts])
                        
                        # Clean text: uppercase, alphanumeric only
                        cleaned_text = self._clean_plate_text(combined_text)
                        
                        if cleaned_text:
                            # STRICT VALIDATION: Only store if validation succeeds
                            validated_plate = self._validate_indian_license_plate(cleaned_text)
                            
                            # ✓ ONLY add to results if validation PASSED
                            # ✗ If validation FAILED, do NOT add to results (flagged as NOT_DETECTED implicitly)
                            if validated_plate is not None:
                                results.append({
                                    "box": box,
                                    "text": validated_plate,
                                    "confidence": avg_confidence,
                                    "validated": True,
                                    "validation_status": "VALID"
                                })
                            # else: Invalid plate — NOT stored, NOT added to results
            
            except Exception as e:
                # Lenient error handling: skip failed plates, continue
                # (No logging needed — invalid plates are expected)
                continue
        
        return results
    
    def _clean_plate_text(self, text):
        """
        Clean extracted license plate text.
        
        Args:
            text: Raw extracted text
        
        Returns:
            Cleaned text (uppercase, alphanumeric only)
        """
        if not text:
            return ""
        
        # Convert to uppercase
        text = text.upper()
        
        # Keep only alphanumeric
        cleaned = "".join(c for c in text if c.isalnum())
        
        return cleaned.strip()
    
    def _validate_indian_license_plate(self, plate_text: str) -> Optional[str]:
        """
        Cross-validate Indian license plate format and state codes.
        
        Args:
            plate_text: Extracted plate text to validate
        
        Returns:
            Validated plate number if VALID, None if INVALID.
            When None is returned, the plate is NOT stored in results.
        """
        try:
            plate = IndianLicensePlate(plate_number=plate_text)
            return plate.plate_number
        except ValidationError as e:
            # Validation failed — return None
            # Caller will NOT add this plate to results
            return None