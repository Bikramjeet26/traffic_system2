"""FastAPI Application for Traffic Violation Detection System"""

import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from datetime import datetime, timezone
from pipeline import TrafficViolationPipeline
from pipeline.config import Config
from pipeline.utils import load_all_evidence

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
        "evidence_folder_exists": os.path.exists("./evidence"),
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
    
    Example:
        POST /analyze
        Form data:
            - file: [image file]
            - timestamp: "2025-06-18T14:30:22.123Z"
            - gps_lat: 28.6139
            - gps_lon: 77.2090
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


# ============ ENDPOINT 2: GET VIOLATIONS ============

@app.get("/violations")
async def get_violations():
    """
    Retrieve all stored violation evidence records.
    
    Returns:
        List of evidence dictionaries (most recent first)
    
    Example:
        GET /violations
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
    
    Returns:
        Violation counts by type and total vehicles detected
    
    Example:
        GET /stats
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
                "recognize_faces": {
                    "method": "POST",
                    "path": "/recognize-faces",
                    "description": "Optional: facial recognition",
                    "parameters": ["file (image)"]
                }
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
