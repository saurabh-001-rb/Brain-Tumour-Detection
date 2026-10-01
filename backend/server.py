import io
import os
from pathlib import Path
from typing import Optional
from PIL import Image

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

try:
    from .model_engine import engine, CLINICAL_PROFILES, CLASS_NAMES
except ImportError:
    from model_engine import engine, CLINICAL_PROFILES, CLASS_NAMES

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="NeuroScan AI - Brain Tumor Diagnostic Workstation",
    description="Advanced Medical Imaging & Deep Learning System for Intracranial Neoplasm Classification",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "active_model": engine.active_model_path.name if engine.active_model_path else "Not Loaded",
        "input_resolution": engine.input_size,
    }


@app.get("/api/samples")
async def get_samples():
    samples = engine.get_sample_scans()
    return {"samples": samples}


@app.get("/api/sample-image/{cls_name}/{filename}")
async def get_sample_image(cls_name: str, filename: str):
    file_path = BASE_DIR / "dataset" / "Testing" / cls_name / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Sample image not found.")
    return FileResponse(file_path)


@app.post("/api/predict")
async def predict_scan(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image (JPEG/PNG).")

    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(exc)}")

    try:
        results = engine.analyze_scan(pil_img)
        return JSONResponse(content=results)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(exc)}")


@app.get("/api/metrics")
async def get_metrics():
    metrics = engine.get_metrics_data()
    return metrics


@app.get("/api/pathology")
async def get_pathology_info():
    return {"pathology": CLINICAL_PROFILES}


# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>NeuroScan AI is starting. Index file being prepared...</h1>")
    return FileResponse(index_path)


def start_server(host: str = "127.0.0.1", port: int = 8000):
    uvicorn.run("server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    start_server()
