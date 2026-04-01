"""
server.py
---------
FastAPI server — receives landmarks from the client, runs the
yoga-assist pipeline, and returns pose classification and corrections.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from server_pipeline import run_pipeline

app = FastAPI(title="AI Yoga Assist — Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/process")
async def process(request: Request):
    """Expects JSON body: {"landmarks": [[x,y,z], ...33 items...]}"""
    data = await request.json()
    landmarks = data.get("landmarks")
    if not landmarks or not isinstance(landmarks, list) or len(landmarks) != 33:
        raise HTTPException(status_code=400, detail="33 landmarks required")

    result = run_pipeline(landmarks)
    return result


@app.get("/health")
def health():
    """Quick liveness probe."""
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
