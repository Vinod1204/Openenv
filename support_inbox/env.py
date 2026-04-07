from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

from pydantic import ValidationError

from .models import Action, Observation, Reward, State, StepResult
from .tasks import TaskSpec, expected_fields, get_task, list_tasks, score_action


class SupportInboxEnv:
    def __init__(self, task_id: Optional[str] = None):
        self._task_id = task_id or list_tasks()[0].task_id
        self._task: TaskSpec = get_task(self._task_id)
        self._step_index = 0
        self._cumulative_reward = 0.0
        self._current_score = 0.0
        self._submitted = False
        self._current_draft: Dict[str, Any] = {
            "category": None,
            "priority": None,
            "route_to": None,
            "needs_human_review": None,
            "submit": False,
            "response_draft": None,
            "summary": None,
            "tags": [],
            "internal_note": None,
            "next_step": None,
        }
        self._last_feedback = "Environment initialized."
        self._last_action: Optional[Action] = None
        self._history: list[dict[str, Any]] = []

    @property
    def task(self) -> TaskSpec:
        return self._task

    @classmethod
    def from_task(cls, task_id: str) -> "SupportInboxEnv":
        return cls(task_id=task_id)

    def reset(self, task_id: Optional[str] = None) -> Observation:
        if task_id is not None:
            self._task_id = task_id
        self._task = get_task(self._task_id)
        self._step_index = 0
        self._cumulative_reward = 0.0
        self._current_score = 0.0
        self._submitted = False
        self._current_draft = {
            "category": None,
            "priority": None,
            "route_to": None,
            "needs_human_review": None,
            "submit": False,
            "response_draft": None,
            "summary": None,
            "tags": [],
            "internal_note": None,
            "next_step": None,
        }
        self._last_feedback = "Fresh ticket loaded."
        self._last_action = None
        self._history = []
        return self._build_observation()

    def step(self, action: Action | Dict[str, Any]) -> StepResult:
        if self._submitted:
            observation = self._build_observation()
            reward = Reward(total=0.0, shaped={"terminal": self._current_score}, final=True)
            return StepResult(observation=observation, reward=reward, done=True, info={"message": "Episode already submitted."})

        try:
            parsed_action = action if isinstance(action, Action) else Action.model_validate(action)
        except ValidationError as exc:
            observation = self._build_observation()
            self._last_feedback = f"Invalid action: {exc.errors()[0]['msg']}"
            reward = Reward(total=0.0, shaped={"validation": 0.0}, final=False)
            self._history.append({"step": self._step_index + 1, "action": deepcopy(self._current_draft), "error": self._last_feedback})
            self._step_index += 1
            done = self._step_index >= self._task.max_steps
            if done:
                self._submitted = True
            return StepResult(observation=observation, reward=reward, done=done, info={"last_action_error": self._last_feedback})

        for field_name, value in parsed_action.model_dump(exclude_none=True).items():
            if field_name == "tags" and value is None:
                continue
            self._current_draft[field_name] = value

        total_score, shaped, feedback = score_action(self._task, self._draft_action())
        reward_value = round(max(total_score - self._current_score, -1.0), 4)
        self._current_score = total_score
        self._cumulative_reward = round(self._cumulative_reward + reward_value, 4)
        self._last_feedback = feedback
        self._last_action = parsed_action
        self._history.append(
            {
                "step": self._step_index + 1,
                "action": parsed_action.model_dump(),
                "score": round(total_score, 4),
                "reward": reward_value,
                "feedback": feedback,
            }
        )

        self._step_index += 1
        done = bool(parsed_action.submit) or self._step_index >= self._task.max_steps
        if done:
            self._submitted = True

        observation = self._build_observation()
        reward = Reward(total=max(min(reward_value, 1.0), 0.0), shaped=shaped, final=done)
        info = {
            "task_id": self._task.task_id,
            "last_action_error": None,
            "feedback": feedback,
            "current_score": round(self._current_score, 4),
            "expected_fields": expected_fields(self._task),
        }
        return StepResult(observation=observation, reward=reward, done=done, info=info)

    def state(self) -> State:
        return State(
            task_id=self._task.task_id,
            step_index=self._step_index,
            cumulative_reward=round(self._cumulative_reward, 4),
            current_score=round(self._current_score, 4),
            submitted=self._submitted,
            current_draft=deepcopy(self._current_draft),
            last_action=self._last_action,
            last_feedback=self._last_feedback,
            history=deepcopy(self._history),
        )

    def close(self) -> None:
        return None

    def _draft_action(self) -> Action:
        return Action.model_validate(self._current_draft)

    def _build_observation(self) -> Observation:
        remaining_steps = max(self._task.max_steps - self._step_index, 0)
        return Observation(
            task_id=self._task.task_id,
            title=self._task.title,
            difficulty=self._task.difficulty,
            step_index=self._step_index,
            max_steps=self._task.max_steps,
            subject=self._task.subject,
            customer_message=self._task.customer_message,
            customer_profile=self._task.customer_profile,
            policy_excerpt=self._task.policy_excerpt,
            current_draft=deepcopy(self._current_draft),
            last_feedback=self._last_feedback,
            remaining_steps=remaining_steps,
            allowed_actions=[
                "set category",
                "set priority",
                "set route_to",
                "set needs_human_review",
                "write response_draft",
                "write summary",
                "add tags",
                "set next_step",
                "submit=true when ready",
            ],
        )
