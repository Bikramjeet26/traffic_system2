"""Facial recognition utilities — optional, only used by /face-match endpoints."""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .config import Config


def _deepface():
    from deepface import DeepFace

    return DeepFace


class FaceRecognitionProcessor:
    """
    Face embedding extraction and cosine-similarity matching (Facenet512 + RetinaFace).
    Independent from the main traffic-violation pipeline.
    """

    def __init__(self, similarity_threshold: Optional[float] = None):
        self.face_db: Optional[Dict[str, List[float]]] = None
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else Config.FACE_SIMILARITY_THRESHOLD
        )

    def load_face_database(self) -> Dict[str, List[float]]:
        """Load optional named face embeddings from face_database/."""
        try:
            db_path = Config.FACE_DATABASE_FOLDER
            if not os.path.exists(db_path):
                self.face_db = {}
                return self.face_db

            embeddings_file = os.path.join(db_path, "embeddings.json")
            if os.path.exists(embeddings_file):
                import json

                with open(embeddings_file, "r", encoding="utf-8") as f:
                    self.face_db = json.load(f)
            else:
                self.face_db = {}
                for filename in os.listdir(db_path):
                    if filename.endswith(".npy"):
                        name = filename.replace(".npy", "")
                        embedding = np.load(os.path.join(db_path, filename))
                        self.face_db[name] = embedding.tolist()

            return self.face_db
        except Exception as e:
            print(f"Error loading face database: {e}")
            self.face_db = {}
            return self.face_db

    @staticmethod
    def _prepare_image_input(
        image: Union[str, np.ndarray],
    ) -> Union[str, np.ndarray]:
        """DeepFace accepts file paths or numpy arrays (BGR/RGB)."""
        if isinstance(image, np.ndarray):
            return image
        if isinstance(image, (bytes, bytearray)):
            arr = np.frombuffer(image, np.uint8)
            decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if decoded is None:
                raise ValueError("Failed to decode image bytes")
            return decoded
        return image

    # def extract_faces(
    #     self,
    #     image: Union[str, np.ndarray, bytes],
    #     enforce_detection: bool = False,
    # ) -> List[Dict[str, Any]]:
    #     """
    #     Detect all faces and return bounding boxes with 512-d embeddings.

    #     Returns:
    #         [{"box": [x1,y1,x2,y2], "embedding": [...], "face_index": 0}, ...]
    #     """
    #     try:
    #         img = self._prepare_image_input(image)
    #         face_objs = _deepface().extract_faces(
    #             img_path=img,
    #             detector_backend=Config.FACE_DETECTOR,
    #             enforce_detection=enforce_detection,
    #             align=True,
    #         )
    #     except Exception as e:
    #         print(f"Face extraction failed: {e}")
    #         return []

    #     results: List[Dict[str, Any]] = []
    #     for idx, face_obj in enumerate(face_objs):
    #         try:
    #             facial_area = face_obj.get("facial_area", {})
    #             x = facial_area.get("x", 0)
    #             y = facial_area.get("y", 0)
    #             w = facial_area.get("w", 0)
    #             h = facial_area.get("h", 0)
    #             box = [int(x), int(y), int(x + w), int(y + h)]

    #             face_img = face_obj.get("face")
    #             if face_img is None:
    #                 continue

    #             embedding_objs = _deepface().represent(
    #                 img_path=face_img,
    #                 model_name=Config.FACE_MODEL,
    #                 enforce_detection=False,
    #             )
    #             if not embedding_objs:
    #                 continue

    #             results.append(
    #                 {
    #                     "box": box,
    #                     "embedding": embedding_objs[0]["embedding"],
    #                     "face_index": idx,
    #                 }
    #             )
    #         except Exception:
    #             continue

    #     return results

    def extract_faces(
            self,
            image: Union[str, np.ndarray, bytes],
            enforce_detection: bool = False,
        ) -> List[Dict[str, Any]]:
            """
            Detect all faces and return bounding boxes with 512-d embeddings.

            Returns:
                [{"box": [x1,y1,x2,y2], "embedding": [...], "face_index": 0}, ...]
            """
            try:
                img = self._prepare_image_input(image)
                # Call represent directly on the full image to get all embeddings correctly
                embedding_objs = _deepface().represent(
                    img_path=img,
                    model_name=Config.FACE_MODEL,
                    detector_backend=Config.FACE_DETECTOR,
                    enforce_detection=enforce_detection,
                )
            except Exception as e:
                print(f"Face extraction failed: {e}")
                return []

            results: List[Dict[str, Any]] = []
            for idx, emb_obj in enumerate(embedding_objs):
                try:
                    facial_area = emb_obj.get("facial_area", {})
                    x = facial_area.get("x", 0)
                    y = facial_area.get("y", 0)
                    w = facial_area.get("w", 0)
                    h = facial_area.get("h", 0)
                    box = [int(x), int(y), int(x + w), int(y + h)]

                    results.append(
                        {
                            "box": box,
                            "embedding": emb_obj["embedding"],
                            "face_index": idx,
                        }
                    )
                except Exception as e:
                    print(f"Error processing face item {idx}: {e}")
                    continue

            return results


    def get_primary_embedding(
        self,
        image: Union[str, np.ndarray, bytes],
        enforce_detection: bool = True,
    ) -> Optional[List[float]]:
        """Return the first detected face embedding (used for reference person)."""
        try:
            img = self._prepare_image_input(image)
            embedding_objs = _deepface().represent(
                img_path=img,
                model_name=Config.FACE_MODEL,
                detector_backend=Config.FACE_DETECTOR,
                enforce_detection=enforce_detection,
            )
            if not embedding_objs:
                return None
            return embedding_objs[0]["embedding"]
        except Exception as e:
            print(f"Primary embedding extraction failed: {e}")
            return None

    @staticmethod
    def cosine_similarity(vec1: Union[List[float], np.ndarray], vec2: Union[List[float], np.ndarray]) -> float:
        v1 = np.array(vec1, dtype=np.float64).flatten()
        v2 = np.array(vec2, dtype=np.float64).flatten()
        norm_a = np.linalg.norm(v1)
        norm_b = np.linalg.norm(v2)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(v1, v2) / (norm_a * norm_b))

    @staticmethod
    def cosine_distance(vec1: Union[List[float], np.ndarray], vec2: Union[List[float], np.ndarray]) -> float:
        return 1.0 - FaceRecognitionProcessor.cosine_similarity(vec1, vec2)

    def compare_embeddings(
        self,
        embedding_a: List[float],
        embedding_b: List[float],
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compare two embeddings and decide match/no-match."""
        threshold = threshold if threshold is not None else self.similarity_threshold
        similarity = self.cosine_similarity(embedding_a, embedding_b)
        distance = 1.0 - similarity
        return {
            "similarity": round(similarity, 4),
            "distance": round(distance, 4),
            "is_match": similarity >= threshold,
            "threshold": threshold,
        }

    def compare_two_images(
        self,
        reference_image: Union[str, np.ndarray, bytes],
        target_image: Union[str, np.ndarray, bytes],
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Compare a reference person image against all faces in a target image.
        Returns the best matching face in the target.
        """
        threshold = threshold if threshold is not None else self.similarity_threshold

        ref_embedding = self.get_primary_embedding(reference_image, enforce_detection=True)
        if ref_embedding is None:
            return {
                "success": False,
                "error": "No face detected in reference (person) image",
                "faces_in_target": 0,
            }

        target_faces = self.extract_faces(target_image, enforce_detection=False)
        if not target_faces:
            return {
                "success": False,
                "error": "No face detected in target image",
                "faces_in_target": 0,
            }

        best: Optional[Dict[str, Any]] = None
        comparisons: List[Dict[str, Any]] = []

        for face in target_faces:
            comparison = self.compare_embeddings(ref_embedding, face["embedding"], threshold)
            entry = {
                "face_index": face["face_index"],
                "box": face["box"],
                **comparison,
            }
            comparisons.append(entry)
            if best is None or entry["similarity"] > best["similarity"]:
                best = entry

        return {
            "success": True,
            "is_match": best["is_match"] if best else False,
            "best_match": best,
            "all_comparisons": comparisons,
            "faces_in_target": len(target_faces),
            "threshold": threshold,
        }

    def match_against_named_database(
        self,
        query_image: Union[str, np.ndarray, bytes],
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Optional: match query image against face_database/ named profiles."""
        threshold = threshold if threshold is not None else self.similarity_threshold

        if self.face_db is None:
            self.load_face_database()

        query_embedding = self.get_primary_embedding(query_image, enforce_detection=True)
        if query_embedding is None:
            return {"success": False, "error": "No face detected in query image"}

        if not self.face_db:
            return {"success": False, "error": "Face database is empty or not configured"}

        best_name = None
        best_similarity = -1.0
        scores: Dict[str, float] = {}

        for name, db_embedding in self.face_db.items():
            similarity = self.cosine_similarity(query_embedding, db_embedding)
            scores[name] = round(similarity, 4)
            if similarity > best_similarity:
                best_similarity = similarity
                best_name = name

        is_match = best_similarity >= threshold
        return {
            "success": True,
            "is_match": is_match,
            "identity": best_name if is_match else None,
            "similarity": round(best_similarity, 4),
            "distance": round(1.0 - best_similarity, 4),
            "all_scores": scores,
            "threshold": threshold,
        }

    def _find_best_match(self, face_embedding: np.ndarray) -> Optional[Tuple[str, float, float]]:
        """Legacy helper for named database lookup."""
        if not self.face_db:
            return None

        best_distance = float("inf")
        best_name = None
        face_embedding = np.array(face_embedding)

        for name, db_embedding in self.face_db.items():
            distance = self.cosine_distance(face_embedding, db_embedding)
            if distance < best_distance:
                best_distance = distance
                best_name = name

        confidence = max(0.0, 1.0 - best_distance)
        if best_distance <= (1.0 - self.similarity_threshold):
            return (best_name, confidence, best_distance)
        return None

    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Legacy API: detect faces and match against named face_database/."""
        try:
            if self.face_db is None:
                self.load_face_database()
            if not self.face_db:
                return []

            faces = self.extract_faces(image, enforce_detection=False)
            results = []
            for face in faces:
                best_match = self._find_best_match(np.array(face["embedding"]))
                if best_match:
                    name, confidence, distance = best_match
                    results.append(
                        {
                            "box": face["box"],
                            "identity": name,
                            "confidence": confidence,
                            "distance": distance,
                        }
                    )
            return results
        except Exception as e:
            print(f"Error detecting faces: {e}")
            return []
