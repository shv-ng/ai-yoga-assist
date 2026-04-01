# AI Yoga Assist

AI Yoga Assist is a real-time yoga pose classification and correction system. It uses computer vision to detect body landmarks, processes them through a normalized pipeline, and employs a machine learning model to identify poses and provide spoken feedback for form corrections.

## Requirements
- **Server:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for running the FastAPI server)
- **Client:** Python 3.10+ (for running the local MediaPipe detection)

## Server Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd yoga-docker
   ```
2. Start the server using Docker Compose:
   ```bash
   docker compose up
   ```
   The server will be available at `http://localhost:8000`.

## Client Setup
1. Install client dependencies:
   ```bash
   pip install -r client-requirements.txt
   ```
2. Run the client:
   ```bash
   python client.py --server http://localhost:8000
   ```

## Supported Poses
The system currently supports the following 10 poses:
- BridgePose
- ChairPose
- CobraPose
- CorpsePose
- DownwardDog
- GoddessPose
- HappyBabyPose
- SupineTwist
- TreePose
- WarriorPose

## Controls
- Press **ESC** to quit the client application.

## Architecture
The system follows a client-server architecture designed for low latency and flexibility. The **client** runs the MediaPipe pose detection locally on the user's machine to capture 33 body landmarks in real-time from the webcam. These landmarks are then sent as a lightweight JSON payload to the **server**, which runs a TensorFlow MLP model inside a Docker container. The server classifies the pose, evaluates it against biomechanical rules for corrections, and returns a voice feedback ID if corrections are needed.
