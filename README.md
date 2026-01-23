# Disaster Monitoring System with YOLO & Gemini

This project implements a Client-Server architecture for advanced disaster monitoring and object detection. It leverages **YOLO** for real-time object detection on the client side (e.g., a drone-mounted Raspberry Pi) and **Google's Gemini/Gemma** models on the server for dynamic prompt discovery and detailed scene analysis.

## Features

- **Real-time Object Detection**: Uses YOLOv11 (YOLOE) for efficient client-side detection.
- **Dynamic Prompt Discovery**: The server continuously analyzes incoming images using Gemini/Gemma to discover new relevant objects (prompts) and pushes them to the client.
- **GPS Integration**: Captures real-time GPS coordinates from a Pixhawk controller via **MAVLink** and tags detections.
- **Web Dashboard**: A real-time web interface to view live detections, current active prompts, and suggested prompts.
- **Headless Operation**: The client is designed to run headlessly on edge devices like Raspberry Pi.
- **Video Recording**: Automatically records annotated video feeds locally on the client.

## Architecture

### Server (`server/`)
- Built with **FastAPI**.
- Manages WebSocket connections for real-time communication.
- Integrates with **Google Gemini** (cloud) or **Gemma** (local/hosted) for image intelligence.
- Serves a static web frontend ("Dashboard").

### Client (`client/`)
- Python-based application.
- Uses **Ultralytics YOLO** for object detection.
- Connects to Pixhawk flight controllers via `pymavlink` for GPS data.
- Sends detected objects (images + metadata) to the server via WebSockets.
- Includes a `fake_client.py` for testing without hardware.

## Prerequisites

- **Python 3.10+**
- **USB Camera** (for Client)
- **Pixhawk / MAVLink generic device** (Optional, for GPS)
- **Gemini API Key** (for Server features)

## Installation

### 1. Clone the Repository
```bash
git clone <repository_url>
cd client_server
```

### 2. Server Setup
```bash
cd server
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Client Setup
```bash
cd client
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
*Note: You may need to download the YOLO model weights (e.g., `yoloe-11s-seg.pt`) and place them in the `client/` directory if they are not included.*

## Usage

### starting the Server
1. Set your Gemini API Key:
   ```bash
   export GEMINI_API_KEY="your_api_key_here"
   ```
   *(Optional) To use Gemma instead, set `MODEL_PROVIDER=gemma`.*

2. Run the server:
   ```bash
   cd server
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```
   The dashboard will be available at `http://localhost:8000`.

### Starting the Client (Hardware)
1. Ensure your camera and MAVLink device are connected.
2. Update `SERVER_URL` in `main.py` if the server is not on `localhost`.
3. Run the client:
   ```bash
   cd client
   python main.py
   ```

### Starting the Fake Client (Testing)
To simulate a client sending images without a camera/drone:
```bash
cd client
python fake_client.py
```

## Configuration

- **Server**:
  - `GEMINI_API_KEY`: API key for Google Gemini.
  - `MODEL_PROVIDER`: Set to `gemini` (default) or `gemma`.
- **Client**:
  - `SERVER_URL`: WebSocket URL of the server (default: `ws://192.168.0.54:8000/ws`).
  - `MAV_PORT`: Serial port for MAVLink (default: `/dev/ttyAMA0`).

## Project Structure
```
├── client/
│   ├── main.py          # Main client application
│   ├── fake_client.py   # Simulator for testing
│   ├── requirements.txt # Client dependencies
│   └── yoloe-11s-seg.pt # YOLO model weights
├── server/
│   ├── main.py          # FastAPI server & WebSocket handler
│   ├── model_provider.py# Interface for Gemini/Gemma
│   ├── requirements.txt # Server dependencies
│   └── static/          # Web dashboard assets
└── tests/               # Tests
```
