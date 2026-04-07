from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .models import Action, Category, Priority, RouteTeam


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    title: str
    difficulty: str
    subject: str
    customer_message: str
    customer_profile: str
    policy_excerpt: str
    expected_category: Category
    expected_priority: Priority
    expected_route: RouteTeam
    expected_needs_human_review: bool
    required_keywords: Tuple[str, ...]
    bonus_keywords: Tuple[str, ...]
    forbidden_keywords: Tuple[str, ...]
    max_steps: int
    response_weight: float
    summary_weight: float
    routing_weight: float
    safety_weight: float


TASKS: Dict[str, TaskSpec] = {
    "billing_refund_triage": TaskSpec(
        task_id="billing_refund_triage",
        title="Billing refund triage",
        difficulty="easy",
        subject="Charged twice after renewal",
        customer_message=(
            "Hi support, I was charged twice for my Pro renewal this morning. "
            "I need someone to fix the duplicate charge and let me know whether a refund is possible."
        ),
        customer_profile="Paid Pro customer, account in good standing, no prior disputes.",
        policy_excerpt=(
            "Duplicate renewal charges should be acknowledged, routed to billing, and reviewed against the last 30 days of transactions. "
            "Do not promise a refund before checking the invoice trail."
        ),
        expected_category="billing",
        expected_priority="normal",
        expected_route="billing",
        expected_needs_human_review=False,
        required_keywords=("apolog", "duplicate", "invoice", "billing", "refund"),
        bonus_keywords=("transaction id", "review", "account"),
        forbidden_keywords=("immediate refund", "guaranteed refund", "password"),
        max_steps=4,
        response_weight=0.45,
        summary_weight=0.20,
        routing_weight=0.25,
        safety_weight=0.10,
    ),
    "outage_status_response": TaskSpec(
        task_id="outage_status_response",
        title="Outage status response",
        difficulty="medium",
        subject="Login failures after SSO change",
        customer_message=(
            "Several teammates cannot sign in after this morning's SSO configuration change. "
            "The error says 'invalid assertion' and the dashboard is timing out."
        ),
        customer_profile="Admin from a mid-market workspace with eight affected users.",
        policy_excerpt=(
            "Identity and authentication incidents should be routed to technical support, marked high priority, "
            "and should ask for the error code, timestamp, and affected browser while providing a status-page link."
        ),
        expected_category="technical",
        expected_priority="high",
        expected_route="technical",
        expected_needs_human_review=True,
        required_keywords=("sso", "error", "status", "browser", "timestamp"),
        bonus_keywords=("affected users", "diagnose", "workaround"),
        forbidden_keywords=("wait 24 hours", "clear cache only", "password reset"),
        max_steps=4,
        response_weight=0.40,
        summary_weight=0.20,
        routing_weight=0.25,
        safety_weight=0.15,
    ),
    "account_takeover_escalation": TaskSpec(
        task_id="account_takeover_escalation",
        title="Account takeover escalation",
        difficulty="hard",
        subject="Suspicious login from new country and mailbox changes",
        customer_message=(
            "I got a login alert from a country I have never visited, then my recovery email was changed. "
            "Please stop whatever is happening and secure my account now."
        ),
        customer_profile="Enterprise admin with SSO disabled for this tenant and payment history under review.",
        policy_excerpt=(
            "Possible account takeover incidents require immediate trust-safety escalation, urgent priority, "
            "no account disclosure, and a concise internal summary with risk factors and next steps."
        ),
        expected_category="security",
        expected_priority="urgent",
        expected_route="trust-safety",
        expected_needs_human_review=True,
        required_keywords=("takeover", "urgent", "secure", "recovery email", "escalate"),
        bonus_keywords=("risk", "verify identity", "lock"),
        forbidden_keywords=("share your password", "we cannot help", "ignore this"),
        max_steps=5,
        response_weight=0.40,
        summary_weight=0.25,
        routing_weight=0.20,
        safety_weight=0.15,
    ),
}


def list_tasks() -> List[TaskSpec]:
    return [TASKS[key] for key in TASKS]


def get_task(task_id: str) -> TaskSpec:
    try:
        return TASKS[task_id]
    except KeyError as exc:
        raise KeyError(f"Unknown task_id: {task_id}") from exc


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _phrase_hits(text: str, phrases: Sequence[str]) -> int:
    normalized = _normalize(text)
    return sum(1 for phrase in phrases if _normalize(phrase) in normalized)


def _category_score(expected: Category, actual: str | None) -> float:
    if actual is None:
        return 0.0
    mapping = {
        "refund": "billing",
        "billing": "billing",
        "charge": "billing",
        "login": "technical",
        "outage": "technical",
        "incident": "technical",
        "security": "security",
        "breach": "security",
        "account": "account",
        "cancel": "cancellation",
    }
    normalized = _normalize(actual)
    mapped = mapping.get(normalized, normalized)
    return 1.0 if mapped == expected else 0.0


def _priority_score(expected: Priority, actual: str | None) -> float:
    return 1.0 if actual == expected else 0.0


def _route_score(expected: RouteTeam, actual: str | None) -> float:
    if actual is None:
        return 0.0
    aliases = {
        "billing": {"billing", "finance"},
        "technical": {"technical", "support", "engineering"},
        "trust-safety": {"trust-safety", "security", "fraud"},
        "identity": {"identity", "sso"},
        "retention": {"retention", "cancellation"},
    }
    normalized = _normalize(actual)
    return 1.0 if normalized in aliases.get(expected, {expected}) else 0.0


def _boolean_score(expected: bool, actual: bool | None) -> float:
    if actual is None:
        return 0.0
    return 1.0 if bool(actual) is expected else 0.0


def _text_score(text: str | None, required: Tuple[str, ...], bonus: Tuple[str, ...], forbidden: Tuple[str, ...]) -> Tuple[float, Dict[str, float]]:
    text = text or ""
    required_hits = _phrase_hits(text, required)
    bonus_hits = _phrase_hits(text, bonus)
    forbidden_hits = _phrase_hits(text, forbidden)
    required_score = required_hits / max(len(required), 1)
    bonus_score = min(bonus_hits / max(len(bonus), 1), 1.0)
    penalty = min(forbidden_hits * 0.3, 1.0)
    score = max(min(required_score * 0.75 + bonus_score * 0.25 - penalty, 1.0), 0.0)
    return score, {
        "required_hits": float(required_hits),
        "bonus_hits": float(bonus_hits),
        "forbidden_hits": float(forbidden_hits),
    }


def score_action(task: TaskSpec, action: Action) -> Tuple[float, Dict[str, float], str]:
    category = _category_score(task.expected_category, action.category)
    priority = _priority_score(task.expected_priority, action.priority)
    route = _route_score(task.expected_route, action.route_to)
    human = _boolean_score(task.expected_needs_human_review, action.needs_human_review)
    response_score, response_detail = _text_score(action.response_draft, task.required_keywords, task.bonus_keywords, task.forbidden_keywords)
    summary_score, summary_detail = _text_score(action.summary, task.required_keywords[: max(2, len(task.required_keywords) // 2)], task.bonus_keywords, task.forbidden_keywords)

    safety_score = 1.0 - min(response_detail["forbidden_hits"] * 0.35 + summary_detail["forbidden_hits"] * 0.35, 1.0)
    safety_score = max(safety_score, 0.0)

    total = (
        category * 0.15
        + priority * 0.10
        + route * task.routing_weight
        + human * 0.10
        + response_score * task.response_weight
        + summary_score * task.summary_weight
        + safety_score * task.safety_weight
    )

    total = max(min(total, 1.0), 0.0)
    feedback_parts = []
    if category < 1.0:
        feedback_parts.append(f"category:{action.category or 'missing'}")
    if priority < 1.0:
        feedback_parts.append(f"priority:{action.priority or 'missing'}")
    if route < 1.0:
        feedback_parts.append(f"route:{action.route_to or 'missing'}")
    if human < 1.0:
        feedback_parts.append("human_review:missing_or_wrong")
    if response_score < 1.0:
        feedback_parts.append("response:improve_keywords")
    if summary_score < 1.0 and action.summary is not None:
        feedback_parts.append("summary:improve_keywords")
    if response_detail["forbidden_hits"] or summary_detail["forbidden_hits"]:
        feedback_parts.append("safety:remove_forbidden_phrases")

    if not feedback_parts:
        feedback = "All target fields aligned with the task rubric."
    else:
        feedback = "; ".join(feedback_parts)

    shaped = {
        "category": round(category, 4),
        "priority": round(priority, 4),
        "route": round(route, 4),
        "human_review": round(human, 4),
        "response": round(response_score, 4),
        "summary": round(summary_score, 4),
        "safety": round(safety_score, 4),
        "total": round(total, 4),
    }
    return total, shaped, feedback


def expected_fields(task: TaskSpec) -> Dict[str, str | bool]:
    return {
        "category": task.expected_category,
        "priority": task.expected_priority,
        "route_to": task.expected_route,
        "needs_human_review": task.expected_needs_human_review,
    }


def _grade(task_id: str, action: Action) -> Dict[str, object]:
    task = get_task(task_id)
    score, shaped, feedback = score_action(task, action)
    return {
        "task_id": task_id,
        "score": round(score, 4),
        "components": shaped,
        "feedback": feedback,
    }


def grade_billing_refund_triage(action: Action | Dict[str, object]) -> Dict[str, object]:
    return _grade("billing_refund_triage", action if isinstance(action, Action) else Action.model_validate(action))


def grade_outage_status_response(action: Action | Dict[str, object]) -> Dict[str, object]:
    return _grade("outage_status_response", action if isinstance(action, Action) else Action.model_validate(action))


def grade_account_takeover_escalation(action: Action | Dict[str, object]) -> Dict[str, object]:
    return _grade("account_takeover_escalation", action if isinstance(action, Action) else Action.model_validate(action))
