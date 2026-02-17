# Step 1 — Data Collection

## Goal: 
Build a CSV dataset of pose sequences recorded from your webcam.

## How it works:
You write a simple Python script that opens your webcam, lets you press R to start recording and S to stop. While recording, every frame MediaPipe gives you 33 landmarks (x, y, z each = 99 numbers). Each frame is one row. When you stop, all rows from that one sequence get saved to a CSV with a label column (e.g. "warrior").

## What you record:
For each of the 6 poses — Vrikshasana, Utkatasana, Virabhadrasana, Bhujangasana, Adho Mukha Svanasana, Utkata Konasana — you record the full motion. Start standing neutral, move into the pose naturally, hold 2-3 seconds, come back to neutral. That whole journey = one sample. Record 50-80 samples per pose. You and Shivang both record, different days, different lighting. More variety = better model.

## What you don't do: 
Don't record static holds. Don't record wrong versions. Unknown poses are handled later by confidence thresholding, not by training.

## Output:
One CSV file. Columns are frame_id, sequence_id, label, x0, y0, z0, x1, y1, z1 ... x32, y32, z32. Roughly 300-480 sequences total.

## Tools:
Python, MediaPipe, OpenCV on your laptop. No Colab needed here, you need the webcam.

Step 2 — Pose Classification Model (CNN/Dense Network)
Goal: A model that looks at a single frame and says "which of the 6 poses is this person attempting?"
Why you need this: Before correction can happen, the system needs to know what pose you're trying to do. You can't give Warrior Pose feedback to someone doing Tree Pose.
How it works: From your CSV, each individual frame (one row = 99 numbers) is a training sample. Label is the pose name. You train a simple neural network — Input layer of 99 neurons → Dense 128 → Dropout → Dense 64 → Dropout → Output 6 classes with softmax. If highest confidence is below say 0.6, output "unknown pose."
Where you train: Google Colab. Upload your CSV, train in 10-15 minutes, download the saved model file (.h5 or .tflite).
Output: A trained classification model file. Target accuracy around 90% which is very achievable for 6 distinct poses.
Tools: Colab, TensorFlow/Keras, your CSV from Step 1.

Step 3 — Pose Correction Model (BiLSTM)
Goal: A model that watches a sliding window of the last 30 frames and continuously judges — is this motion trajectory correct for the detected pose?
Why BiLSTM: This is the real intelligence. It has memory of how you got to where you are. It knows the correct path into each pose. If your arm is rotating wrong mid-motion, it catches it before you even finish the pose.
How it works: From your CSV, you group rows back into sequences by sequence_id. Each sequence becomes a 3D input — shape is (number of frames, 99 landmarks). You use a sliding window of 30 frames. The model architecture is Input sequence of 30 frames → BiLSTM 64 units → Dropout 0.3 → BiLSTM 32 units → Dense 16 → Output which is a correction category per body region (arms wrong, legs wrong, torso wrong, or all correct).
The labels for correction: Since you only recorded correct sequences, each window from a correct sequence is labeled "correct." To generate "incorrect" labels, you artificially perturb the landmark coordinates — shift arm landmarks by a random amount, flip left-right, etc. This is called data augmentation and it's how you create wrong examples without recording them.
Where you train: Google Colab again. This takes a bit longer, maybe 20-30 minutes.
Output: A trained BiLSTM model file. Plus a correction dictionary — a simple Python dict that maps correction category to a text message like "Raise your right arm higher" or "Bend your left knee more."
Tools: Colab, TensorFlow/Keras, same CSV from Step 1.

Step 4 — Real-Time Integration
Goal: One Python script that ties everything together and runs live on your laptop webcam.
How the pipeline flows: Webcam frame comes in → MediaPipe extracts 33 landmarks → Classification model looks at this single frame and identifies which pose you're attempting → That frame gets added to a rolling buffer of last 30 frames → BiLSTM looks at the buffer every few frames and outputs correction flags → If correction needed, text is displayed on the OpenCV window immediately → Buffer slides forward, process repeats.
What the user sees: A live webcam feed with the MediaPipe skeleton drawn on them. Top of screen shows the detected pose name. If something is wrong, a correction message appears in red — like "Lower your hips more" or "Extend your left arm fully." If everything looks good, it shows "Good form" in green. If pose is unrecognised, it shows "Unknown Pose."
Important detail: The correction text should stay on screen for about 2-3 seconds after being triggered, not flicker every frame. Use a simple timer for this.
Tools: Python, OpenCV, MediaPipe, TensorFlow, both your saved model files. Runs entirely on your laptop, no internet needed.

Step 5 — Testing, Polish and Report
Goal: Prove the system works, document it properly, and package the unfinished parts as future scope.
Testing: Record yourself doing poses and measure how often the classifier gets it right — this gives you your accuracy number. Make a confusion matrix (6×6 grid showing which poses get confused with which). This is one graph that looks very professional in a report and takes 5 minutes to generate with sklearn.
Polish: Handle edge cases gracefully — what if person walks out of frame (landmarks disappear), what if lighting is bad (confidence drops). Just show "Please adjust position" in these cases. Nothing fancy.
What goes in Future Scope: Raspberry Pi deployment, voice audio feedback (pyttsx3 is literally 3 lines of code so you could actually add it as a bonus in a day), mobile app, more poses beyond 6.
Report content: System architecture diagram (already in your synopsis), confusion matrix graph, accuracy graph over training epochs, a table comparing your accuracy with the papers in your literature review, and a short demo video of the live system working. That demo video is the most important thing — evaluators remember what they see.
Tools: sklearn for confusion matrix, matplotlib for graphs, your phone to record the demo video.
