from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

from support_inbox.models import Action, Observation, Reward, State, StepResult


@dataclass
class SupportInboxClient:
    base_url: str = "http://127.0.0.1:7860"

    def reset(self, task_id: Optional[str] = None) -> StepResult:
        response = requests.post(f"{self.base_url}/reset", json={"task_id": task_id})
        response.raise_for_status()
        payload = response.json()
        return StepResult(
            observation=Observation.model_validate(payload["observation"]),
            reward=Reward(total=0.0, shaped={"reset": 1.0}, final=False),
            done=False,
            info={"state": payload["state"]},
        )

    def step(self, action: Action | Dict[str, Any]) -> StepResult:
        payload = action.model_dump() if isinstance(action, Action) else action
        response = requests.post(f"{self.base_url}/step", json={"action": payload})
        response.raise_for_status()
        data = response.json()
        return StepResult(
            observation=Observation.model_validate(data["observation"]),
            reward=Reward.model_validate(data["reward"]),
            done=bool(data["done"]),
            info=data.get("info", {}),
        )

    def state(self) -> State:
        response = requests.get(f"{self.base_url}/state")
        response.raise_for_status()
        return State.model_validate(response.json())


__all__ = ["SupportInboxClient"]
