"""
server.py
---------
FastAPI server — receives JPEG frames from the ESP32, runs the full
yoga-assist pipeline, and serves pre-generated voice MP3 files back.

Endpoints
---------
POST /process
    Body  : raw JPEG bytes
    Return: {"voice_id": int | null}

GET /voice/{vid}
    Return: audio/mpeg  (the .mp3 for that voice_id)

Run with:
    uvicorn server:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import numpy as np
import cv2
import os
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from server_pipeline import run_pipeline, run_pipeline_landmarks, VOICE_DIR

app = FastAPI(title="AI Yoga Assist — Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/process_landmarks")
async def process_landmarks(request: Request):
    """Expects JSON body: {"landmarks": [{"x": 0.5, "y": 0.5, "z": 0.1, "visibility": 0.9}, ...]}"""
    data = await request.json()
    landmarks = data.get("landmarks")
    if not landmarks:
        raise HTTPException(status_code=400, detail="no landmarks provided")

    voice_id = run_pipeline_landmarks(landmarks)
    return {"voice_id": voice_id}


@app.post("/process")
async def process_frame(request: Request):
    jpg_bytes = await request.body()
    if not jpg_bytes:
        raise HTTPException(status_code=400, detail="empty body")

    jpg = np.frombuffer(jpg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="could not decode JPEG")

    voice_id = run_pipeline(frame)
    return JSONResponse({"voice_id": voice_id})   # voice_id may be null


@app.get("/voice/{vid}")
def get_voice(vid: int):
    path = VOICE_DIR / f"{vid:03d}.mp3"
    if not path.exists():
        raise HTTPException(status_code=404, detail="voice file not found")
    return FileResponse(str(path), media_type="audio/mpeg")


@app.get("/health")
def health():
    """Quick liveness probe — useful during ESP32 bring-up."""
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app)
