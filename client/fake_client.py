import cv2
import websocket
import json
import base64
import time
import threading
from ultralytics import YOLOE
import numpy as np
import os
import glob

# Configuration
SERVER_URL = "ws://192.168.0.116:8000/ws" 
# Note: For real Pi deployment, 'localhost' should be changed to Server IP.

class ObjectDetectionClient:
    def __init__(self):
        self.ws = None
        self.prompts = []
        self.model = YOLOE("yoloe-11s.yaml")  # Using YOLOE small model
        self.model.load("yoloe-11s-seg.pt")
        self.running = True
        self.last_detection_time = 0
        self.detection_cooldown = 4.0 # Retaining config, though loop sleep overrides frequency

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
        # Run WebSocket in a separate thread so it doesn't block the loop
        self.ws = websocket.WebSocketApp(SERVER_URL,
                                         on_open=self.on_open,
                                         on_message=self.on_message,
                                         on_error=self.on_error,
                                         on_close=self.on_close)
        wst = threading.Thread(target=self.ws.run_forever)
        wst.daemon = True
        wst.start()

    def detect_and_send(self):
        images_dir = os.path.join(os.path.dirname(__file__), "test_images")
        print(f"Fake Client started. Reading images from {images_dir}")
        print("Waiting for prompts...")

        while self.running:
            # Re-scan directory every loop to pick up new files
            image_files = sorted(glob.glob(os.path.join(images_dir, "*")))
            # Filter for images
            image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            if not image_files:
                print("No images found in test_images. Waiting...")
                time.sleep(2)
                continue

            for img_path in image_files:
                if not self.running:
                    break

                frame = cv2.imread(img_path)
                if frame is None:
                    print(f"Failed to read {img_path}")
                    continue

                if not self.prompts:
                    # No prompts set, just show locally
                    cv2.imshow('Client Feed (Fake)', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                        break
                    time.sleep(5)
                    continue

                # Run YOLOE inference
                # Lowering conf to 0.1 for zero-shot detection. 
                # retina_masks=True provides higher resolution segmentation masks.
                results = self.model.predict(frame, conf=0.05, retina_masks=True, verbose=False)
                
                detected = False
                best_conf = 0
                best_label = ""

                if results and len(results) > 0:
                    result = results[0]
                    # Log what we found
                    num_boxes = len(result.boxes) if result.boxes else 0
                    num_masks = len(result.masks) if result.masks else 0
                    print(f"Inference results: {num_boxes} boxes, {num_masks} masks")

                    if result.boxes:
                        for box in result.boxes:
                            conf = float(box.conf[0])
                            cls_id = int(box.cls[0])
                            label = result.names[cls_id]
                            
                            if label in self.prompts:
                                detected = True
                                if conf > best_conf:
                                    best_conf = conf
                                    best_label = label

                # Logic: In fake client, if detected, we send. 
                # We enforce 5s wait regardless of cooldown logic to match "every 5 seconds" request exactly as loop interval.
                current_time = time.time()
                
                if detected:
                    print(f"Detected {best_label} ({best_conf:.2f}) in {os.path.basename(img_path)}. Sending...")
                    
                    annotated_img = results[0].plot(conf=False)
                    _, buffer = cv2.imencode('.jpg', annotated_img)
                    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                    
                    payload = {
                        "type": "detection",
                        "image": jpg_as_text,
                        "label": best_label,
                        "confidence": best_conf,
                        "gps": {
                            "lat": 42.21796186856417,
                            "lon": -71.16652560166608
                        }
                    }
                    
                    try:
                        self.ws.send(json.dumps(payload))
                        self.last_detection_time = current_time
                    except Exception as e:
                        print(f"Failed to send detection: {e}")
                else:
                    print(f"No target object detected in {os.path.basename(img_path)}")

                # Show local video feed
                annotated_frame = results[0].plot(conf=False) if results else frame
                
                # Resize for display
                h, w = annotated_frame.shape[:2]
                max_w = 400
                if w > max_w:
                    scale = max_w / w
                    new_w = max_w
                    new_h = int(h * scale)
                    annotated_frame = cv2.resize(annotated_frame, (new_w, new_h))
                
                cv2.imshow('Client Feed (Fake)', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                   self.running = False
                   break
                
                # Wait 5 seconds before next picture
                time.sleep(5.0)

        cv2.destroyAllWindows()

if __name__ == "__main__":
    client = ObjectDetectionClient()
    client.connect()
    
    try:
        client.detect_and_send()
    except KeyboardInterrupt:
        print("Stopping client...")
        client.running = False
