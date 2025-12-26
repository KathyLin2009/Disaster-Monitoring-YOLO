import cv2
import websocket
import json
import base64
import time
import threading
from ultralytics import YOLOE
import numpy as np

# Configuration
SERVER_URL = "ws://192.168.0.114:8000/ws" 
# Note: For real Pi deployment, 'localhost' should be changed to Server IP.

class ObjectDetectionClient:
    def __init__(self):
        self.ws = None
        self.prompts = []
        self.model = YOLOE("yoloe-11s-seg.pt")  # Using YOLOE small model
        self.cap = None
        self.running = True
        self.last_detection_time = 0
        self.detection_cooldown = 2.0

    def on_message(self, ws, message):
        data = json.loads(message)
        if data.get("type") == "update_prompt":
            new_prompts = data.get("prompts", [])
            print(f"Updating prompts to: {new_prompts}")
            self.prompts = new_prompts
            if self.prompts:
                # YOLOE requires text embeddings along with names
                self.model.set_classes(self.prompts, self.model.get_text_pe(self.prompts))
            else:
                # If no prompts, maybe reset classes or pause detection logic
                pass

    def on_error(self, ws, error):
        print(f"WebSocket Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print("WebSocket Closed")

    def on_open(self, ws):
        print("WebSocket Connected")

    def connect(self):
        # Run WebSocket in a separate thread so it doesn't block the camera loop
        self.ws = websocket.WebSocketApp(SERVER_URL,
                                         on_open=self.on_open,
                                         on_message=self.on_message,
                                         on_error=self.on_error,
                                         on_close=self.on_close)
        wst = threading.Thread(target=self.ws.run_forever)
        wst.daemon = True
        wst.start()

    def detect_and_send(self):
        # Open Camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return

        print("Camera started. Waiting for prompts...")

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                time.sleep(0.1)
                continue

            if not self.prompts:
                # No prompts set, just show fee locally to indicate it's running
                cv2.imshow('Client Feed', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break
                time.sleep(0.01)
                continue

            # Run Usage
            # YOLOE inference
            results = self.model.predict(frame, conf=0.25, verbose=False)
            
            detected = False
            best_conf = 0
            best_label = ""

            # Check results
            if results and len(results) > 0:
                result = results[0]
                if result.boxes:
                    for box in result.boxes:
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        label = result.names[cls_id]
                        
                        # Check strictly if this label is in our current prompts 
                        # (YOLOE set_classes should handle this, but good to double check)
                        if label in self.prompts:
                            detected = True
                            if conf > best_conf:
                                best_conf = conf
                                best_label = label

            current_time = time.time()
            if detected and (current_time - self.last_detection_time > self.detection_cooldown):
                print(f"Detected {best_label} ({best_conf:.2f}). Sending to server...")
                
                # Encode frame to jpg -> base64
                _, buffer = cv2.imencode('.jpg', frame)
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                
                payload = {
                    "type": "detection",
                    "image": jpg_as_text,
                    "label": best_label,
                    "confidence": best_conf
                }
                
                try:
                    self.ws.send(json.dumps(payload))
                    self.last_detection_time = current_time
                except Exception as e:
                    print(f"Failed to send detection: {e}")

            # Show local video feed with bounding boxes
            annotated_frame = results[0].plot() if results else frame
            cv2.imshow('Client Feed', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
               self.running = False
               break
            
            # Simple sleep to prevent 100% CPU usage if no specific FPS target
            time.sleep(0.01)

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    client = ObjectDetectionClient()
    client.connect()
    
    try:
        client.detect_and_send()
    except KeyboardInterrupt:
        print("Stopping client...")
        client.running = False
