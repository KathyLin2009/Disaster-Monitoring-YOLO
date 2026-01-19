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
from model_provider import GeminiProvider, GemmaProvider


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
        self.current_prompts = []
        self.suggested_prompts = []
        self._last_discovery_time = 0

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        # Send current prompts (or seed prompts if empty) and pi status to newly connected client
        prompts_to_send = self.current_prompts
        await websocket.send_json({
            "type": "init", 
            "prompts": prompts_to_send,
            "suggested_prompts": self.suggested_prompts,
            "pi_connected": self.pi_connected
        })
        
        # If Pi just connected and we have no prompts, push seed prompts immediately
        if self.client_info.get(websocket) == "pi" and not self.current_prompts:
            # Wait for user input
            await self.broadcast({"type": "update_prompt", "prompts": []})

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

    async def handle_accept_suggestion(self, prompt: str):
        if prompt in self.suggested_prompts:
            self.suggested_prompts.remove(prompt)
            if prompt not in self.current_prompts:
                self.current_prompts.append(prompt)
                
            await self.broadcast({"type": "update_prompt", "prompts": self.current_prompts})
            await self.broadcast({"type": "suggested_prompts_update", "prompts": self.suggested_prompts})

    async def handle_reject_suggestion(self, prompt: str):
        if prompt in self.suggested_prompts:
            self.suggested_prompts.remove(prompt)
            await self.broadcast({"type": "suggested_prompts_update", "prompts": self.suggested_prompts})

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
        if not image_b64:
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
                # Add to suggested if not already active and not already suggested
                if p not in self.current_prompts and p not in self.suggested_prompts:
                    print(f"Discovered new prompt suggestion: {p}")
                    self.suggested_prompts.append(p)
                    updated = True
            
            if updated:
                await self.broadcast({"type": "suggested_prompts_update", "prompts": self.suggested_prompts})

    async def register_pi(self, websocket: WebSocket):
        self.client_info[websocket] = "pi"
        self.pi_connected = True
        await self.broadcast({"type": "pi_status", "connected": True})
        
        # Push initial prompts to Pi immediately
        prompts_to_send = self.current_prompts
        
        await websocket.send_json({
            "type": "update_prompt",
            "prompts": self.current_prompts
        })



manager = ConnectionManager()

# Model Provider Configuration
PROVIDER_TYPE = os.environ.get("MODEL_PROVIDER", "gemini").lower()
print(f"Initializing Model Provider: {PROVIDER_TYPE}")

if PROVIDER_TYPE == "gemma":
    model_provider = GemmaProvider()
else:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        print("Warning: GEMINI_API_KEY not set. Gemini will not work.")
    model_provider = GeminiProvider(api_key=GEMINI_API_KEY)

class ImageAnalysisRequest(BaseModel):
    image: str

@app.post("/analyze_image")
async def analyze_image(request: ImageAnalysisRequest):
    try:
        # Also trigger discovery based on this manual analysis
        # We do this asynchronously/background usually, but here await is fine
        await manager.run_discovery(request.image)
        
        description = await model_provider.analyze_image(request.image)
        return {"description": description}
    except Exception as e:
        print(f"Analysis Error: {e}")
        return {"description": f"Failed to analyze image: {str(e)}"}

async def analyze_for_prompts(image_b64: str, current_prompts: List[str]):
    return await model_provider.discover_prompts(image_b64, current_prompts)

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
                
            elif msg_type == "accept_suggestion":
                await manager.handle_accept_suggestion(data.get("prompt"))
                
            elif msg_type == "reject_suggestion":
                await manager.handle_reject_suggestion(data.get("prompt"))

    except WebSocketDisconnect:
        await manager.disconnect(websocket)
        print("Client disconnected")
