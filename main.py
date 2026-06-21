"""FastAPI Application for Traffic Violation Detection System"""

import os
import uuid
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from datetime import datetime, timezone
from pipeline import TrafficViolationPipeline
from pipeline.config import Config
from pipeline.utils import load_all_evidence
from pipeline.face_match_service import FaceMatchService
from pipeline.stopline import process_video_headless

# ============ INITIALIZATION ============

app = FastAPI(
    title="Traffic Violation Detection System",
    description="Detects traffic violations: helmets, seatbelts, and extracts license plates",
    version="1.0.0"
)

# CORS — required for Vercel frontend calling this HF Spaces backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS,
    allow_origin_regex=Config.CORS_ORIGIN_REGEX or None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline (singleton instance)
pipeline = TrafficViolationPipeline()
face_match_service = FaceMatchService()

# Serve stored evidence images/JSON for the frontend violations log
Config.ensure_folders_exist()
app.mount("/evidence", StaticFiles(directory=Config.EVIDENCE_FOLDER), name="evidence")


# ============ HEALTH CHECK ============

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "Traffic Violation Detection System",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "models_loaded": False,  # Models lazy-load on first use
        "evidence_folder_exists": os.path.exists(Config.EVIDENCE_FOLDER),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# ============ ENDPOINT 1: ANALYZE (MAIN) ============

@app.post("/analyze")
async def analyze_violation(
    file: UploadFile = File(...),
    timestamp: str = Form(None),
    gps_lat: float = Form(0.0),
    gps_lon: float = Form(0.0)
):
    """
    Analyze image for traffic violations.
    
    Args:
        file: Image file (JPEG, PNG, etc.)
        timestamp: ISO format timestamp (default: current UTC time)
        gps_lat: Latitude coordinate (default: 0.0)
        gps_lon: Longitude coordinate (default: 0.0)
    
    Returns:
        JSON with evidence record and annotated image (base64)
    """
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty image file")
        
        # Use provided timestamp or current UTC time
        if not timestamp:
            timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        
        # Validate GPS
        gps = (gps_lat, gps_lon)
        
        # Run pipeline
        result = pipeline.run(image_bytes, timestamp, gps)
        
        if not result["success"]:
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "Pipeline processing failed")
            )
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "timestamp": timestamp,
                "gps": {"latitude": gps_lat, "longitude": gps_lon},
                "evidence": result["evidence"],
                "annotated_image_base64": result.get("annotated_image_base64", ""),
                "violations_count": len(result["evidence"].get("violations", [])),
                "plates_count": len(result["evidence"].get("plates", []))
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /analyze: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# ============ ENDPOINT: ANALYZE VIDEO (STOP-LINE) ============
@app.post("/analyze-video")
async def analyze_video_upload(
    file: UploadFile = File(..., description="Video file (mp4, mov)"),
    stop_line: str = Form(None, description="Optional normalized stop line as JSON '[x1,y1,x2,y2]' (0..1)"),
    initial_light_state: str = Form("red", description="Initial light state: 'red' or 'green'"),
    conf_thres: float = Form(0.3, description="Detection confidence threshold"),
):
    """
    Headless video analysis that runs the stop-line pipeline on an uploaded video.
    Returns annotated video URL and JSON results saved under /evidence.
    """
    try:
        # Save uploaded video to the evidence folder
        Config.ensure_folders_exist()
        tmp_name = f"upload_{uuid.uuid4().hex}.mp4"
        tmp_path = os.path.join(Config.EVIDENCE_FOLDER, tmp_name)

        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty video file uploaded")

        with open(tmp_path, "wb") as f:
            f.write(contents)

        # parse stop_line JSON if provided
        stop_line_norm = None
        if stop_line:
            try:
                parsed = json.loads(stop_line)
                if (isinstance(parsed, list) or isinstance(parsed, tuple)) and len(parsed) == 4:
                    stop_line_norm = tuple(map(float, parsed))
                else:
                    raise ValueError("stop_line must be a JSON array of 4 numbers [x1,y1,x2,y2]")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid stop_line: {e}")

        # Run the headless video processing function (synchronous)
        result = process_video_headless(
            video_path=tmp_path,
            model_path=None,
            stop_line_norm=stop_line_norm,
            initial_light_state=initial_light_state,
            conf_thres=float(conf_thres),
            output_basename=f"stopline_{uuid.uuid4().hex}"
        )

        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Processing failed"))

        # Return the web-accessible URL for the annotated video and JSON
        return JSONResponse(status_code=200, content={
            "success": True,
            "annotated_video_url": result.get("annotated_video_url"),
            "annotated_video_path": result.get("annotated_video_path"),
            "results_json_url": result.get("results_json_url"),
            "results_json_path": result.get("results_json_path"),
            "violations_count": len(result.get("violations", [])),
            "violations": result.get("violations", []),
        })

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /analyze-video: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


# ============ ENDPOINT 2: GET VIOLATIONS ============

@app.get("/violations")
async def get_violations():
    """
    Retrieve all stored violation evidence records.
    """
    try:
        evidence_list = load_all_evidence()
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "total_records": len(evidence_list),
                "violations": evidence_list
            }
        )
    
    except Exception as e:
        print(f"Error in /violations: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# ============ ENDPOINT 3: GET STATS ============

@app.get("/stats")
async def get_stats():
    """
    Get aggregated violation statistics from all stored records.
    """
    try:
        evidence_list = load_all_evidence()
        
        # Aggregate statistics
        stats = {
            "total_records": len(evidence_list),
            "total_violations": 0,
            "total_vehicles": 0,
            "total_plates": 0,
            "violation_counts": {},
            "vehicle_class_counts": {}
        }
        
        for evidence in evidence_list:
            # Count violations
            violations = evidence.get("violations", [])
            stats["total_violations"] += len(violations)
            
            for violation in violations:
                vtype = violation.get("type", "unknown")
                stats["violation_counts"][vtype] = stats["violation_counts"].get(vtype, 0) + 1
                
                # Count by vehicle class
                vclass = violation.get("vehicle_class", "unknown")
                stats["vehicle_class_counts"][vclass] = stats["vehicle_class_counts"].get(vclass, 0) + 1
            
            # Count plates
            plates = evidence.get("plates", [])
            stats["total_plates"] += len(plates)
        
        # Calculate some simple metrics
        stats["average_violations_per_record"] = (
            stats["total_violations"] / stats["total_records"]
            if stats["total_records"] > 0 else 0
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "stats": stats
            }
        )
    
    except Exception as e:
        print(f"Error in /stats: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# ============ ENDPOINT 4: FACE MATCH (INDEPENDENT) ============

@app.post("/face-match/compare")
async def face_match_compare(
    person_image: UploadFile = File(..., description="Reference person/criminal image"),
    target_image: UploadFile = File(..., description="Image to compare against"),
):
    """Compare two uploaded face images and return whether they match."""
    try:
        person_bytes = await person_image.read()
        target_bytes = await target_image.read()
        if not person_bytes or not target_bytes:
            raise HTTPException(status_code=400, detail="Both image files are required")

        result = face_match_service.compare_images(person_bytes, target_bytes)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Comparison failed"))

        return JSONResponse(status_code=200, content={"success": True, **result})
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /face-match/compare: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/face-match/violation")
async def face_match_violation(
    person_image: UploadFile = File(..., description="Reference person/criminal image"),
    violation_id: str = Form(None, description="Stored violation id/timestamp to compare against"),
    scan_all_violations: bool = Form(False, description="Scan entire violation database"),
):
    """Compare person against one stored violation or scan the violation database."""
    try:
        person_bytes = await person_image.read()
        if not person_bytes:
            raise HTTPException(status_code=400, detail="Empty person_image file")

        has_violation_id = violation_id is not None and violation_id.strip() != ""

        if has_violation_id and scan_all_violations:
            raise HTTPException(
                status_code=400,
                detail="Provide either violation_id or scan_all_violations, not both",
            )
        if not has_violation_id and not scan_all_violations:
            raise HTTPException(
                status_code=400,
                detail="Provide violation_id or set scan_all_violations=true",
            )

        if has_violation_id:
            result = face_match_service.compare_with_violation(
                person_bytes,
                violation_id.strip(),
            )
        else:
            result = face_match_service.scan_violation_database(person_bytes)

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Face match failed"))

        return JSONResponse(status_code=200, content={"success": True, **result})
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /face-match/violation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# ============ ENDPOINT 5: API INFO ============

@app.get("/info")
async def info():
    """Get API information and available endpoints"""
    return JSONResponse(
        status_code=200,
        content={
            "api_name": "Traffic Violation Detection System",
            "version": "1.0.0",
            "endpoints": {
                "health": {
                    "method": "GET",
                    "path": "/health",
                    "description": "Health check"
                },
                "analyze": {
                    "method": "POST",
                    "path": "/analyze",
                    "description": "Main endpoint: analyze image for violations",
                    "parameters": ["file (image)", "timestamp (ISO)", "gps_lat", "gps_lon"]
                },
                "violations": {
                    "method": "GET",
                    "path": "/violations",
                    "description": "Get all stored violation records"
                },
                "stats": {
                    "method": "GET",
                    "path": "/stats",
                    "description": "Get aggregated statistics"
                },
                "face_match_compare": {
                    "method": "POST",
                    "path": "/face-match/compare",
                    "description": "Compare two uploaded face images",
                    "parameters": ["person_image", "target_image"],
                },
                "face_match_violation": {
                    "method": "POST",
                    "path": "/face-match/violation",
                    "description": "Compare person against stored violation(s)",
                    "parameters": ["person_image", "violation_id OR scan_all_violations"],
                },
            },
            "models": {
                "vehicle_detection": ["bdd100k_opensource.pt", "UVH-26-MV-YOLOv11-S.pt"],
                "license_plate": "License_plate_bb.pt",
                "helmet_detection": "helmet_detection.pt",
                "seatbelt_detection": "seatbelt_detection.pt",
                "facial_recognition": "DeepFace (Facenet512)"
            }
        }
    )


# ============ ERROR HANDLERS ============

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error"}
    )


# ============ STARTUP/SHUTDOWN ============

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("Traffic Violation Detection System started")
    print("Models will be lazy-loaded on first use")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("Traffic Violation Detection System shutting down")


# ============ RUN ============

if __name__ == "__main__":
    import uvicorn
    from pipeline.config import Config
    
    uvicorn.run(
        "main:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        workers=Config.API_WORKERS,
        reload=False
    )
