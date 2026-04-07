from __future__ import annotations

import json
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from .env import SupportInboxEnv
from .models import Action
from .tasks import list_tasks


app = FastAPI(title="Support Inbox OpenEnv", version="0.1.0")
_ENV = SupportInboxEnv()


class ResetRequest(BaseModel):
    task_id: Optional[str] = None


class StepRequest(BaseModel):
    action: Dict[str, Any]


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/tasks")
def tasks() -> Dict[str, Any]:
    return {
        "tasks": [
            {
                "task_id": task.task_id,
                "title": task.title,
                "difficulty": task.difficulty,
                "max_steps": task.max_steps,
            }
            for task in list_tasks()
        ]
    }


@app.post("/reset")
def reset(request: Optional[ResetRequest] = None) -> Dict[str, Any]:
    request = request or ResetRequest()
    observation = _ENV.reset(task_id=request.task_id)
    return {"observation": observation.model_dump(), "state": _ENV.state().model_dump()}


@app.post("/step")
def step(request: StepRequest) -> Dict[str, Any]:
    result = _ENV.step(Action.model_validate(request.action))
    return result.model_dump()


@app.get("/state")
def state() -> Dict[str, Any]:
    return _ENV.state().model_dump()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    env = SupportInboxEnv()
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_json()
            message_type = message.get("type")
            if message_type == "reset":
                observation = env.reset(task_id=message.get("task_id"))
                await websocket.send_json({"type": "reset", "observation": observation.model_dump(), "state": env.state().model_dump()})
            elif message_type == "step":
                result = env.step(Action.model_validate(message.get("action", {})))
                await websocket.send_json({"type": "step", **result.model_dump()})
            elif message_type == "state":
                await websocket.send_json({"type": "state", "state": env.state().model_dump()})
            else:
                await websocket.send_json({"type": "error", "error": f"unknown message type: {message_type}"})
    except WebSocketDisconnect:
        return
