from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, conlist, confloat


Priority = Literal["low", "normal", "high", "urgent"]
Category = Literal["billing", "technical", "security", "account", "cancellation"]
RouteTeam = Literal["billing", "support", "identity", "trust-safety", "technical", "retention"]


class Action(BaseModel):
    category: Optional[Category] = None
    priority: Optional[Priority] = None
    route_to: Optional[RouteTeam] = None
    needs_human_review: Optional[bool] = None
    submit: bool = False
    response_draft: Optional[str] = None
    summary: Optional[str] = None
    tags: conlist(str, min_length=0) = Field(default_factory=list)
    internal_note: Optional[str] = None
    next_step: Optional[str] = None


class Observation(BaseModel):
    task_id: str
    title: str
    difficulty: str
    step_index: int
    max_steps: int
    subject: str
    customer_message: str
    customer_profile: str
    policy_excerpt: str
    current_draft: Dict[str, Any]
    last_feedback: str
    remaining_steps: int
    allowed_actions: List[str]


class Reward(BaseModel):
    total: confloat(ge=0.0, le=1.0)
    shaped: Dict[str, float]
    final: bool = False


class State(BaseModel):
    task_id: str
    step_index: int
    cumulative_reward: float
    current_score: float
    submitted: bool
    current_draft: Dict[str, Any]
    last_action: Optional[Action] = None
    last_feedback: str
    history: List[Dict[str, Any]]


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
