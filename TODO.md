**CHUNK 1 — Data Pipeline (do this first, everything depends on it)**

- Find and download kaggle datasets that have the 10 poses (or at least the 4 missing ones)
- Write a script to convert kaggle data (likely images or csvs) into our landmark format using mediapipe, so it matches what collect_data.py produces
- Merge kaggle landmarks + our custom dataset into one poses.csv
- Since our custom data is small, add augmentation (small rotations, flips, slight noise) so it doesn't get drowned out
- Make sure all 10 pose labels are present and balanced enough
- Verify the sequence_id grouping still works after merging so train/test split has no leakage

---

**CHUNK 2 — Missing Pose Corrections**

- Write check_corpse_pose() in corrections.py
- Write check_bridge_pose() in corrections.py
- Write check_supine_twist_pose() in corrections.py
- Write check_happy_baby_pose() in corrections.py
- Register all 4 in the POSE_CHECKERS dispatcher at the bottom of corrections.py
- Add all 10 labels to collect_data.py LABELS list (for any future custom data collection)

---

**CHUNK 3 — Fix Existing Code Bugs**

- Remove the duplicate process() method in pipeline.py (keep the EMA smooth one)
- Remove the dead unused line in session_logger.py get_live_stats
- Remove the debug frame counter and probability printer from realtime.py
- Remove the stray `import sys` inside the frame loop in realtime.py

---

**CHUNK 4 — Retrain the Model**

- Retrain pose_classifier notebook with the new full 10-pose dataset
- Make sure normalization in notebook exactly matches pipeline.py (it already does, just verify after any changes)
- Save new .h5 and label_encoder.pkl
- Evaluate: confusion matrix, per-class F1, check all 10 poses have decent scores

---

**CHUNK 5 — New Architecture: Client/Server Split**

Current state: everything runs in one process on one machine.
New architecture: client runs mediapipe and display, server runs tensorflow and corrections, server lives in docker.

Client side (runs on any machine, no tensorflow needed):
- Client captures webcam frames
- Client runs mediapipe locally to get landmarks
- Client sends landmark vectors (the 99-float array) to server over HTTP, NOT jpeg frames
- Client receives back a list of correction messages + pose label + confidence
- Client displays pose label, confidence, correction text on screen (the cv2 overlay stuff)
- Client handles the "step into frame" visibility check locally since it has the landmarks

Server side (runs in docker):
- Remove mediapipe from server entirely, server no longer processes images
- Server receives landmark vectors, runs classifier, runs check_pose, returns corrections as JSON
- Add a cooldown/throttle on server so it doesn't return a correction on every single frame (port the FeedbackManager tick logic, or just let client handle throttle)
- Actually simpler: server is stateless, just classify+correct per request, client decides when to show/speak
- Remove voice/audio generation from server entirely (no gTTS, no pyttsx3 on server)

Client voice:
- Client does the voice feedback locally using pyttsx3 (already works offline)
- Client has its own FeedbackManager with tick and cooldown
- So voice stays offline on the client machine

---

**CHUNK 6 — Docker Setup for Server**

- Write a Dockerfile for the server: python base image, install tensorflow, fastapi, uvicorn, scikit-learn, numpy
- Write docker-compose.yml so it's just `docker compose up` on friend's laptop
- Models (.h5 and .pkl) mounted as a volume so they don't have to be baked into the image, or bake them in, either works
- Server exposes port 8000
- Add a /health endpoint (already exists, keep it)
- Make sure the server's requirements are minimal: no mediapipe, no opencv, no pyttsx3, no gTTS

---

**CHUNK 7 — Client Packaging / Ease of Use**

- Write a client requirements.txt: opencv, mediapipe, pyttsx3, requests, numpy
- The client script should accept server URL as a config or argument so friend can point it at the docker server
- Ideally one command to run client: `python client.py --server http://localhost:8000`
- Write a top level README: how to start docker server, how to run client, what poses are supported

---

**CHUNK 8 — Remove / Clean Up What's No Longer Needed**

- session_logger.py — not needed per your requirement, remove or just don't wire it up
- server_pipeline.py — will be heavily rewritten for new arch, old version goes away
- server.py /voice and /process endpoints change completely
- The old realtime.py classify() function gets split: mediapipe+display part becomes the new client.py, tensorflow part moves to server
- main.py can be simplified or replaced with just "run client" and "collect data" options
- Remove TFLite export from notebook since raspberry pi is dropped

---

**Order to do these:**

1 → 4 → 2 → 3 → 5 → 6 → 7 → 8

Data first, then retrain, then fix corrections, then fix bugs, then build new arch, then docker, then packaging, then cleanup.
