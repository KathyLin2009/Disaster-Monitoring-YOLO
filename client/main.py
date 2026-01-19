import cv2
import websocket
import json
import base64
import time
import threading
from ultralytics import YOLOE
import numpy as np
from pymavlink import mavutil

# Configuration
SERVER_URL = "ws://192.168.0.116:8000/ws" 
# Note: For real Pi deployment, 'localhost' should be changed to Server IP.

class ObjectDetectionClient:
    def __init__(self):
        self.ws = None
        self.prompts = []
        self.model = YOLOE("yoloe-11s.yaml")  # Using YOLOE small model
        self.model.load("yoloe-11s-seg.pt")
        self.cap = None
        self.running = True
        self.last_detection_time = 0
        self.detection_cooldown = 4.0
        
        # MAVLink / GPS State
        self.mav_connection = None
        self.current_gps = {"lat": 0.0, "lon": 0.0}
        self.mav_thread = None
        self.mav_port = "/dev/ttyAMA0" # Default Pixhawk port on Linux/Pi, might need adjustment
        self.mav_baud = 57600

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
        # Register as Pi
        ws.send(json.dumps({"type": "register_pi"}))

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

        # Start MAVLink thread
        self.mav_thread = threading.Thread(target=self.mavlink_loop)
        self.mav_thread.daemon = True
        self.mav_thread.start()

    def mavlink_loop(self):
        print(f"Connecting to MAVLink on {self.mav_port}...")
        try:
            # Try to connect. If it fails, we'll just log and continue with 0,0
            self.mav_connection = mavutil.mavlink_connection(self.mav_port, baud=self.mav_baud)
            self.mav_connection.wait_heartbeat(timeout=5)
            print("MAVLink Heartbeat received!")
            
            # Request position data stream (10Hz)
            # This is often needed if the Pixhawk is not configured to stream by default
            print("Requesting position data stream...")
            self.mav_connection.mav.request_data_stream_send(
                self.mav_connection.target_system,
                self.mav_connection.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_POSITION,
                10, # 10 Hz
                1   # Start
            )
        except Exception as e:
            print(f"MAVLink Connection Error: {e}")
            return

        while self.running:
            try:
                # Wait for any message to see what's actually arriving
                msg = self.mav_connection.recv_match(blocking=True, timeout=1.0)
                if not msg:
                    # Optional: print a dot or something to show the loop is alive but silent
                    # print("DEBUG: No message received in 1s")
                    continue
                
                # Print all message types to see if we are getting anything at all
                # This helps verify the connection is alive even if GPS isn't coming through
                if msg.get_type() not in ['BAD_DATA', 'HEARTBEAT']: # Filter noise if needed
                     print(f"DEBUG: Received message type: {msg.get_type()}")

                if msg.get_type() == 'GLOBAL_POSITION_INT':
                    # Scale is 1e7
                    self.current_gps["lat"] = msg.lat / 1e7
                    self.current_gps["lon"] = msg.lon / 1e7
                    print(f"GPS Updated: {self.current_gps}")
                elif msg.get_type() == 'GPS_RAW_INT':
                    self.current_gps["lat"] = msg.lat / 1e7
                    self.current_gps["lon"] = msg.lon / 1e7
                    print(f"GPS Updated (RAW): {self.current_gps}")
            except Exception as e:
                print(f"MAVLink Loop Error: {e}")

            time.sleep(2)

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
                # No prompts set, just sleep and continue
                time.sleep(0.1)
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
                
                # Encode annotated frame (with boxes) to jpg -> base64
                annotated_img = results[0].plot(conf=False)
                _, buffer = cv2.imencode('.jpg', annotated_img)
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                
                payload = {
                    "type": "detection",
                    "image": jpg_as_text,
                    "label": best_label,
                    "confidence": best_conf,
                    "gps": self.current_gps
                }
                
                try:
                    self.ws.send(json.dumps(payload))
                    self.last_detection_time = current_time
                except Exception as e:
                    print(f"Failed to send detection: {e}")

            # Loop sleep
            time.sleep(0.01)

        self.cap.release()

if __name__ == "__main__":
    client = ObjectDetectionClient()
    client.connect()
    
    try:
        client.detect_and_send()
    except KeyboardInterrupt:
        print("Stopping client...")
        client.running = False
