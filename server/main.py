from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import logging
from typing import List
import json
import os
import base64
import time
import google.generativeai as genai
from google.api_core import exceptions
from pydantic import BaseModel

app = FastAPI()

# Serve static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Store connected clients (browser and raspberry pi)
# For simplicity, we'll try to distinguish or just broadcast to relevant parties.
# But logically:
# - Browser sends prompts -> Server -> Pi
# - Pi sends images -> Server -> Browser

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.pi_connected = False
        self.client_info = {} # ws -> client_type
        # Seed prompts to bootstrap discovery if none are set
        self.seed_prompts = ["person", "car", "dog", "tree", "fire"]
        self.current_prompts = []
        self._last_discovery_time = 0

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        # Send current prompts (or seed prompts if empty) and pi status to newly connected client
        prompts_to_send = self.current_prompts if self.current_prompts else self.seed_prompts
        await websocket.send_json({
            "type": "init", 
            "prompts": prompts_to_send,
            "pi_connected": self.pi_connected
        })
        
        # If Pi just connected and we have no prompts, push seed prompts immediately
        if self.client_info.get(websocket) == "pi" and not self.current_prompts:
            self.current_prompts = self.seed_prompts.copy()
            await self.broadcast({"type": "update_prompt", "prompts": self.current_prompts})

    async def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        if self.client_info.get(websocket) == "pi":
            self.pi_connected = False
            del self.client_info[websocket]
            await self.broadcast({"type": "pi_status", "connected": False})
        elif websocket in self.client_info:
             del self.client_info[websocket]

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Error sending to client: {e}")

    async def handle_prompt_update(self, prompts: List[str]):
        self.current_prompts = prompts
        await self.broadcast({"type": "update_prompt", "prompts": prompts})

    async def handle_detection_event(self, data: dict):
        # Broadcast the detection to browsers
        await self.broadcast({
            "type": "detection_event", 
            "image": data.get("image"),
            "label": data.get("label"),
            "confidence": data.get("confidence"),
            "gps": data.get("gps")
        })
        
        # Also use this image for discovery updates
        await self.run_discovery(data.get("image"))

    async def run_discovery(self, image_b64: str):
        if not image_b64 or not GEMINI_API_KEY:
            return

        # Simple rate limit to avoid GEMINI spam (5 second cooldown)
        if time.time() - self._last_discovery_time < 5.0:
            return
        self._last_discovery_time = time.time()

        print("Running Gemini prompt discovery from detection...")
        new_discovered = await analyze_for_prompts(image_b64, self.current_prompts)
        
        if new_discovered:
            updated = False
            for p in new_discovered:
                if p not in self.current_prompts:
                    print(f"Discovered new prompt: {p}")
                    self.current_prompts.append(p)
                    updated = True
            
            if updated:
                await self.broadcast({"type": "update_prompt", "prompts": self.current_prompts})

            if updated:
                # Also ensure the seed prompts aren't just duplicating
                await self.broadcast({"type": "update_prompt", "prompts": self.current_prompts})

    async def register_pi(self, websocket: WebSocket):
        self.client_info[websocket] = "pi"
        self.pi_connected = True
        await self.broadcast({"type": "pi_status", "connected": True})
        
        # Push initial prompts to Pi immediately
        prompts_to_send = self.current_prompts if self.current_prompts else self.seed_prompts
        if not self.current_prompts:
            self.current_prompts = self.seed_prompts.copy()
        
        await websocket.send_json({
            "type": "update_prompt",
            "prompts": self.current_prompts
        })



manager = ConnectionManager()

# Gemini Configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("Warning: GEMINI_API_KEY not set. Image analysis will fail.")

class ImageAnalysisRequest(BaseModel):
    image: str

@app.post("/analyze_image")
async def analyze_image(request: ImageAnalysisRequest):
    if not GEMINI_API_KEY:
        return {"description": "Error: Server missing GEMINI_API_KEY."}

    try:
        # Remove header if present
        img_str = request.image
        if "," in img_str:
            img_str = img_str.split(",")[1]
        
        img_bytes = base64.b64decode(img_str)
        
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        prompt = """Analyze the provided image for any potential risks or hazards. Also give a brief description. 
        Do not mention the YOLO bounding boxes or percentages.
        Do not format the output, just use plain text in one paragraph."""

        # Retry logic for rate limits
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.generate_content([
                    prompt,
                    {"mime_type": "image/jpeg", "data": img_bytes}
                ])
                # Also trigger discovery based on this manual analysis
                await manager.run_discovery(request.image)
                
                return {"description": response.text}
            except exceptions.ResourceExhausted as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + 1 # 2s, 3s, 5s...
                    print(f"Rate limit hit. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise e
    except Exception as e:
        print(f"Gemini Error: {e}")
        return {"description": f"Failed to analyze image: {str(e)}"}

async def analyze_for_prompts(image_b64: str, current_prompts: List[str]):
    if not GEMINI_API_KEY:
        return []

    try:
        # Remove header if present
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]
            
        img_bytes = base64.b64decode(image_b64)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        if not current_prompts:
            prompt = """Analyze this image and identify the 5 most important or distinct object categories present. 
            Focus on things that would be relevant for a drone monitoring for disasters or interesting sights (e.g. people, fire, flood, cars, trees, animals).
            Return only a comma-separated list of the 5 labels as lowercase text. No explanation."""
        else:
            prompt = f"""Analyze this image. The current detected objects are: {', '.join(current_prompts)}.
            Identify up to 3 additional unique objects of interest not already listed. These must be important for aiding a search and rescue team and must be related to the landscape after a natural disaster.
            Priority: Only suggest objects that are more visually prominent or contextually significant than those currently listed. 
            Constraints: Do not exceed a total of 20 prompts (including current ones).
            Output: Return only a comma-separated list of new lowercase labels. 
            If no significant new objects are found, return an empty string. Do not include any introductory text or explanation."""

        # Run in a thread or use async if available (Gemini SDK is synchronous mostly but can use generate_content_async)
        response = await model.generate_content_async([
            prompt,
            {"mime_type": "image/jpeg", "data": img_bytes}
        ])
        
        text = response.text.strip()
        if not text:
            return []
            
        # Basic cleanup: remove markdown if Gemini added it
        text = text.replace("```", "").replace("csv", "").strip()
        
        new_prompts = [p.strip().lower() for p in text.split(",") if p.strip()]
        return new_prompts
    except Exception as e:
        print(f"Gemini Discovery Error: {e}")
        return []

@app.get("/")
async def get():
    with open(os.path.join(static_dir, "index.html")) as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "register_pi":
                print("Pi connected and registered.")
                await manager.register_pi(websocket)

            elif msg_type == "set_prompt":
                prompts = data.get("prompts", [])
                print(f"Received new prompts: {prompts}")
                await manager.handle_prompt_update(prompts)
            
            elif msg_type == "detection":
                await manager.handle_detection_event(data)

    except WebSocketDisconnect:
        await manager.disconnect(websocket)
        print("Client disconnected")
