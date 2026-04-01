# Project Requirements: AI-Based Real-Time Yoga Pose Correction System

## 1. System Overview

The system is an **AI-driven yoga assistance platform** designed to:

* Detect human body posture in real time
* Classify yoga poses accurately
* Identify deviations from correct posture
* Provide **instant voice-based corrective feedback**

The system must operate in real time, ensuring low latency and high usability during live yoga sessions.

---

## 2. Functional Requirements

### 2.1 Pose Detection

* The system shall capture live video input from a camera.
* It shall detect and track **33 human body landmarks** per frame.
* Landmark detection must include spatial coordinates (x, y, z).
* The system shall handle:

  * Different lighting conditions
  * Variations in user position and orientation

---

### 2.2 Feature Processing

* Landmark data must be:

  * Normalized relative to body proportions
  * Transformed into structured feature vectors (99 features per frame)
* Temporal smoothing must be applied to reduce noise and jitter.
* The system must maintain consistency between training and inference preprocessing.

---

### 2.3 Pose Classification

* The system shall classify yoga poses using a **lightweight neural network model (MLP)**.

* Input: 99-dimensional feature vector

* Output: Pose class probabilities

* The classifier must:

  * Support **multi-class classification**
  * Achieve **≥90% accuracy** (target ~94%)
  * Operate in real time

---

### 2.4 Supported Yoga Poses

The system must support classification of the following **10 poses**:

* Tree Pose (Vrikshasana)
    https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset
    https://www.kaggle.com/datasets/sumanthvrao/yoga-poses
    https://www.kaggle.com/datasets/bharatwajc/yoga-dataset-11-poses
    https://www.kaggle.com/datasets/tr1gg3rtrash/yoga-posture-dataset/data
* Chair Pose (Utkatasana)
    https://www.kaggle.com/datasets/bharatwajc/yoga-dataset-11-poses
    https://www.kaggle.com/datasets/tr1gg3rtrash/yoga-posture-dataset/data
* Warrior Pose (Virabhadrasana)
    https://www.kaggle.com/datasets/tr1gg3rtrash/yoga-posture-dataset/data
    https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset
    https://www.kaggle.com/datasets/sumanthvrao/yoga-poses
    https://www.kaggle.com/datasets/bharatwajc/yoga-dataset-11-poses
* Cobra Pose (Bhujangasana)
    https://www.kaggle.com/datasets/shrutisaxena/yoga-pose-image-classification-dataset
* Downward Dog (Adho Mukha Svanasana)
    https://www.kaggle.com/datasets/shrutisaxena/yoga-pose-image-classification-dataset
    https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset
    https://www.kaggle.com/datasets/sumanthvrao/yoga-poses
    https://www.kaggle.com/datasets/bharatwajc/yoga-dataset-11-poses
    https://www.kaggle.com/datasets/tr1gg3rtrash/yoga-posture-dataset/data
* Goddess Pose (Utkata Konasana)
    https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset
    https://www.kaggle.com/datasets/bharatwajc/yoga-dataset-11-poses
    https://www.kaggle.com/datasets/tr1gg3rtrash/yoga-posture-dataset/data
* Corpse Pose (Savasana)
    https://www.kaggle.com/datasets/bharatwajc/yoga-dataset-11-poses
* Bridge Pose (Setu Bandhasana)
    https://www.kaggle.com/datasets/sumanthvrao/yoga-poses
    https://www.kaggle.com/datasets/tr1gg3rtrash/yoga-posture-dataset/data
* Supine Twist (Supta Matsyendrasana)
    https://www.kaggle.com/datasets/shrutisaxena/yoga-pose-image-classification-dataset
* Happy Baby Pose (Ananda Balasana)
    https://www.kaggle.com/datasets/shrutisaxena/yoga-pose-image-classification-dataset

---

### 2.5 Pose Correction Logic

* The system shall evaluate posture correctness using:

  * Joint angles
  * Limb alignment
  * Body symmetry and positioning

* It must:

  * Detect incorrect posture in real time
  * Identify specific errors (e.g., knee misalignment, hip height)

* Each correction must include:

  * Body region identification
  * Severity classification (levels 1–3)

---

### 2.6 Feedback Generation

* The system shall generate **natural language correction instructions**.

* Feedback must be:

  * Clear and concise
  * Actionable (e.g., “Raise your hips higher”)

* A **priority mechanism** must:

  * Rank corrections based on severity
  * Deliver the most critical correction first
  * Avoid excessive feedback (controlled frequency)

---

### 2.7 Voice Output

* The system shall convert feedback into speech using an **offline text-to-speech engine**.
* Voice output must:

  * Be understandable and natural
  * Operate without internet dependency

---

### 2.8 Session Tracking

* The system shall log session data including:

  * Duration of session
  * Pose durations
  * Number of corrections
  * Correction resolution rate

* Logs must be stored in a structured format (e.g., JSON).

---

## 3. Non-Functional Requirements

### 3.1 Performance

* Frame processing rate: **~25 FPS**
* Per-frame inference latency: **<100 ms**
* End-to-end feedback latency: **≤300 ms**

---

### 3.2 Accuracy

* Pose classification accuracy: **≥90%**
* Target achieved: ~94%
* High precision and recall across all pose classes

---

### 3.3 Reliability

* The system must:

  * Maintain stable detection across frames
  * Handle missing or low-confidence landmark data
  * Avoid jitter through smoothing techniques

---

### 3.4 Usability

* The system must:

  * Provide real-time guidance without delays
  * Be usable by beginners without prior training
  * Deliver non-intrusive feedback

---

### 3.5 Privacy

* All processing must occur locally.
* No user video or data should be transmitted externally.

---

## 4. Data Requirements

### 4.1 Dataset Structure

* Input data: Landmark coordinates (33 points × 3 dimensions)
* Each sample:

  * Represents a pose or sequence of frames
  * Includes normalized coordinates

---

### 4.2 Dataset Composition

* Balanced dataset:

  * Multiple samples per pose
  * Includes both correct and incorrect posture examples

---

### 4.3 Data Processing

* Must include:

  * Normalization (hip-centered, torso-scaled)
  * Sequence grouping
  * Train-test split without data leakage

---

## 5. Software Requirements

### 5.1 Core Libraries

* Computer vision and processing:

  * OpenCV
  * MediaPipe

* Machine learning:

  * TensorFlow / TFLite
  * Scikit-learn

* Data handling:

  * NumPy
  * Pandas

---

### 5.2 Audio Processing

* Text-to-speech engine:

  * Offline capability required
* Audio playback support

---

### 5.3 Development Tools

* Python 3.8+
* Jupyter Notebook (for training and evaluation)
* Visualization libraries (Matplotlib)

---

## 6. System Constraints

* Must operate in **real time**
* Must function **offline**
* Must use **lightweight models suitable for edge deployment**
* Must maintain **low computational overhead**

---

## 7. Evaluation Requirements

* The system shall be evaluated on:

  * Classification accuracy
  * Confusion matrix analysis
  * Precision, recall, F1-score
  * Real-time performance metrics

* Validation must include:

  * Controlled dataset testing
  * Live testing scenarios

---

## 8. Expected Outcomes

* Accurate real-time yoga pose recognition
* Immediate corrective feedback
* Improved user posture and safety
* Consistent performance across multiple poses
