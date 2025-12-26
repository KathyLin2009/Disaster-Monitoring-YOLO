from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import logging
from typing import List
import json
import os

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
        self.current_prompts: List[str] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        # Send current prompts to newly connected client immediately
        if self.current_prompts:
            await websocket.send_json({"type": "update_prompt", "prompts": self.current_prompts})

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Error sending to client: {e}")

    async def handle_prompt_update(self, prompts: List[str]):
        self.current_prompts = prompts
        # Broadcast new prompts to all (specifically needed for the Pi)
        await self.broadcast({"type": "update_prompt", "prompts": prompts})

    async def handle_detection_event(self, data: dict):
        # Broadcast detection image/info to all (specifically needed for the Browser)
        await self.broadcast({
            "type": "detection_event", 
            "image": data.get("image"),
            "label": data.get("label"),
            "confidence": data.get("confidence")
        })

manager = ConnectionManager()

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

            if msg_type == "set_prompt":
                prompts = data.get("prompts", [])
                print(f"Received new prompts: {prompts}")
                await manager.handle_prompt_update(prompts)
            
            elif msg_type == "detection":
                # print(f"Received detection for: {data.get('label')}")
                await manager.handle_detection_event(data)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("Client disconnected")
