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
        self.pi_connected = False
        self.client_info = {} # ws -> client_type

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        # Send current prompts and pi status to newly connected client
        await websocket.send_json({
            "type": "init", 
            "prompts": self.current_prompts,
            "pi_connected": self.pi_connected
        })

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
        await self.broadcast({
            "type": "detection_event", 
            "image": data.get("image"),
            "label": data.get("label"),
            "confidence": data.get("confidence"),
            "gps": data.get("gps")
        })

    async def register_pi(self, websocket: WebSocket):
        self.client_info[websocket] = "pi"
        self.pi_connected = True
        await self.broadcast({"type": "pi_status", "connected": True})

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
