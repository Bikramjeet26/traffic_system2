"""Face match orchestration — violation DB lookup and compare modes."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2

from .config import Config
from .face_preprocessor import FaceRecognitionProcessor
from .utils import load_all_evidence


class FaceMatchService:
    """Independent service for /face-match endpoints."""

    def __init__(self):
        self.processor = FaceRecognitionProcessor()

    @staticmethod
    def _resolve_violation_image_path(evidence: Dict[str, Any]) -> Optional[str]:
        """Resolve stored violation image path from an evidence record."""
        annotated_path = evidence.get("annotated_image_path")
        if annotated_path and os.path.isfile(annotated_path):
            return annotated_path

        timestamp = evidence.get("timestamp")
        if not timestamp:
            return None

        filename = timestamp.replace(":", "-").replace("Z", "Z") + ".jpg"
        candidate = os.path.join(Config.EVIDENCE_FOLDER, filename)
        if os.path.isfile(candidate):
            return candidate

        evidence_path = Path(Config.EVIDENCE_FOLDER)
        stem = Path(filename).stem
        for jpg in evidence_path.glob("*.jpg"):
            if jpg.stem == stem or stem in jpg.stem:
                return str(jpg)
        return None

    @staticmethod
    def _find_evidence_by_id(violation_id: str) -> Optional[Dict[str, Any]]:
        """Find a stored violation record by timestamp stem or partial id."""
        violation_id = violation_id.strip()
        if not violation_id:
            return None

        normalized_id = violation_id.replace(":", "-")
        for evidence in load_all_evidence():
            timestamp = evidence.get("timestamp", "")
            stem = timestamp.replace(":", "-").replace("Z", "Z")
            json_stem = Path(stem + ".json").stem if not stem.endswith(".json") else Path(stem).stem

            if (
                timestamp == violation_id
                or stem == normalized_id
                or json_stem == normalized_id
                or normalized_id in stem
                or violation_id in timestamp
            ):
                return evidence
        return None

    def _load_violation_image(self, evidence: Dict[str, Any]) -> Optional[Any]:
        image_path = self._resolve_violation_image_path(evidence)
        if not image_path:
            return None
        image = cv2.imread(image_path)
        return image

    def compare_images(
        self,
        person_image: Union[bytes, Any],
        target_image: Union[bytes, Any],
        threshold: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Mode 1: compare uploaded person image against a second uploaded image."""
        result = self.processor.compare_two_images(person_image, target_image, threshold)
        return {**result, "mode": "image_to_image"}

    def compare_with_violation(
        self,
        person_image: Union[bytes, Any],
        violation_id: str,
        threshold: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Mode 2: compare person image against one stored violation record."""
        evidence = self._find_evidence_by_id(violation_id)
        if not evidence:
            return {
                "success": False,
                "error": f"Violation record not found: {violation_id}",
                "mode": "violation_record",
            }

        violation_image = self._load_violation_image(evidence)
        if violation_image is None:
            return {
                "success": False,
                "error": "Violation image file not found for the given record",
                "violation": self._evidence_summary(evidence),
                "mode": "violation_record",
            }

        comparison = self.processor.compare_two_images(person_image, violation_image, threshold)
        return {
            **comparison,
            "mode": "violation_record",
            "violation": self._evidence_summary(evidence),
        }

    def scan_violation_database(
        self,
        person_image: Union[bytes, Any],
        threshold: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Mode 3: scan all stored violations and return best match(es)."""
        evidence_list = load_all_evidence()
        if limit is not None and limit > 0:
            evidence_list = evidence_list[:limit]

        ref_embedding = self.processor.get_primary_embedding(person_image, enforce_detection=True)
        if ref_embedding is None:
            return {
                "success": False,
                "error": "No face detected in reference (person) image",
                "mode": "violation_database_scan",
            }

        threshold = threshold if threshold is not None else self.processor.similarity_threshold
        matches: List[Dict[str, Any]] = []
        scanned = 0
        skipped = 0

        for evidence in evidence_list:
            violation_image = self._load_violation_image(evidence)
            if violation_image is None:
                skipped += 1
                continue

            scanned += 1
            comparison = self.processor.compare_two_images(person_image, violation_image, threshold)
            if not comparison.get("success"):
                continue

            best = comparison.get("best_match") or {}
            matches.append(
                {
                    "violation": self._evidence_summary(evidence),
                    "is_match": comparison.get("is_match", False),
                    "similarity": best.get("similarity", 0.0),
                    "distance": best.get("distance", 1.0),
                    "face_index": best.get("face_index"),
                    "box": best.get("box"),
                    "faces_in_violation": comparison.get("faces_in_target", 0),
                }
            )

        matches.sort(key=lambda m: m.get("similarity", 0.0), reverse=True)
        positive_matches = [m for m in matches if m.get("is_match")]

        return {
            "success": True,
            "mode": "violation_database_scan",
            "is_match": len(positive_matches) > 0,
            "total_violations_scanned": scanned,
            "total_violations_skipped": skipped,
            "matches_found": len(positive_matches),
            "best_match": matches[0] if matches else None,
            "all_results": matches,
            "positive_matches": positive_matches,
            "threshold": threshold,
        }

    @staticmethod
    def _evidence_summary(evidence: Dict[str, Any]) -> Dict[str, Any]:
        gps = evidence.get("gps", [])
        return {
            "timestamp": evidence.get("timestamp"),
            "gps": {
                "latitude": gps[0] if isinstance(gps, (list, tuple)) and len(gps) > 0 else None,
                "longitude": gps[1] if isinstance(gps, (list, tuple)) and len(gps) > 1 else None,
            },
            "violations_count": len(evidence.get("violations", [])),
            "plates_count": len(evidence.get("plates", [])),
            "annotated_image_path": evidence.get("annotated_image_path"),
        }
