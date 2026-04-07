from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from client import SupportInboxClient
from server.app import app
from support_inbox.env import SupportInboxEnv
from support_inbox.models import Action
from support_inbox.tasks import list_tasks


class SupportInboxTaskTests(unittest.TestCase):
    def test_task_catalog(self) -> None:
        tasks = list_tasks()
        self.assertEqual(len(tasks), 3)
        self.assertEqual([task.difficulty for task in tasks], ["easy", "medium", "hard"])

    def test_environment_progression_across_tasks(self) -> None:
        cases = [
            (
                "billing_refund_triage",
                Action(
                    category="billing",
                    priority="normal",
                    route_to="billing",
                    needs_human_review=False,
                    response_draft="Sorry about the duplicate charge. I am routing this to billing for invoice review and possible refund assessment.",
                    summary="Duplicate renewal charge needs billing review before any refund promise.",
                    tags=["billing", "refund"],
                    next_step="Review the invoice trail and confirm the duplicate transaction.",
                    submit=True,
                ),
            ),
            (
                "outage_status_response",
                Action(
                    category="technical",
                    priority="high",
                    route_to="technical",
                    needs_human_review=True,
                    response_draft="I am escalating this SSO incident to technical support. Please share the exact error code, timestamp, affected browser, and status page context.",
                    summary="SSO login failures affecting multiple users should be escalated with error, timestamp, and browser details.",
                    tags=["sso", "outage"],
                    next_step="Collect browser, timestamp, and error details for technical triage.",
                    submit=True,
                ),
            ),
            (
                "account_takeover_escalation",
                Action(
                    category="security",
                    priority="urgent",
                    route_to="trust-safety",
                    needs_human_review=True,
                    response_draft="This looks like a possible account takeover. I am escalating it immediately and urging the customer to secure the account.",
                    summary="Possible account takeover with foreign login alert and recovery email change; urgent trust-safety escalation required.",
                    tags=["security", "account-takeover"],
                    next_step="Lock down the account and escalate with risk indicators.",
                    submit=True,
                ),
            ),
        ]

        for task_id, action in cases:
            with self.subTest(task_id=task_id):
                env = SupportInboxEnv.from_task(task_id)
                observation = env.reset(task_id=task_id)
                self.assertEqual(observation.task_id, task_id)
                self.assertEqual(env.state().step_index, 0)

                result = env.step(action)
                self.assertTrue(0.0 <= result.reward.total <= 1.0)
                self.assertTrue(result.done)
                self.assertEqual(result.observation.task_id, task_id)
                self.assertGreaterEqual(env.state().current_score, 0.0)
                self.assertEqual(env.state().submitted, True)

    def test_partial_progress_and_history(self) -> None:
        env = SupportInboxEnv.from_task("billing_refund_triage")
        env.reset(task_id="billing_refund_triage")

        first = env.step(
            Action(
                category="billing",
                priority="normal",
                route_to="billing",
                needs_human_review=False,
                tags=["billing"],
                submit=False,
            )
        )
        self.assertFalse(first.done)
        self.assertTrue(0.0 <= first.reward.total <= 1.0)
        self.assertEqual(len(env.state().history), 1)
        self.assertGreater(env.state().current_score, 0.0)

        second = env.step(
            Action(
                category="billing",
                priority="normal",
                route_to="billing",
                needs_human_review=False,
                response_draft="Sorry about the duplicate charge. I am routing this to billing for invoice review and possible refund assessment.",
                summary="Duplicate renewal charge needs billing review before any refund promise.",
                tags=["billing", "refund"],
                next_step="Review the invoice trail and confirm the duplicate transaction.",
                submit=True,
            )
        )
        self.assertTrue(second.done)
        self.assertGreaterEqual(env.state().current_score, first.reward.total)
        self.assertEqual(len(env.state().history), 2)


class SupportInboxServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_http_contract(self) -> None:
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")

        tasks = self.client.get("/tasks")
        self.assertEqual(tasks.status_code, 200)
        self.assertEqual(len(tasks.json()["tasks"]), 3)

        reset = self.client.post("/reset", json={"task_id": "outage_status_response"})
        self.assertEqual(reset.status_code, 200)
        self.assertEqual(reset.json()["observation"]["task_id"], "outage_status_response")

        step = self.client.post(
            "/step",
            json={
                "action": {
                    "category": "technical",
                    "priority": "high",
                    "route_to": "technical",
                    "needs_human_review": True,
                    "response_draft": "I am escalating this SSO incident to technical support.",
                    "summary": "SSO login failures should be escalated.",
                    "tags": ["sso"],
                    "submit": True,
                }
            },
        )
        self.assertEqual(step.status_code, 200)
        payload = step.json()
        self.assertIn("reward", payload)
        self.assertIn("done", payload)
        self.assertTrue(0.0 <= payload["reward"]["total"] <= 1.0)

        state = self.client.get("/state")
        self.assertEqual(state.status_code, 200)
        self.assertEqual(state.json()["task_id"], "outage_status_response")

    def test_websocket_contract(self) -> None:
        with self.client.websocket_connect("/ws") as websocket:
            websocket.send_json({"type": "reset", "task_id": "account_takeover_escalation"})
            reset_message = websocket.receive_json()
            self.assertEqual(reset_message["type"], "reset")
            self.assertEqual(reset_message["observation"]["task_id"], "account_takeover_escalation")

            websocket.send_json({"type": "state"})
            state_message = websocket.receive_json()
            self.assertEqual(state_message["type"], "state")
            self.assertEqual(state_message["state"]["task_id"], "account_takeover_escalation")


class TutorialClientTests(unittest.TestCase):
    def test_client_wrapper_uses_http_contract(self) -> None:
        test_client = TestClient(app)

        class FakeResponse:
            def __init__(self, response) -> None:
                self._response = response

            def json(self):
                return self._response.json()

            def raise_for_status(self):
                self._response.raise_for_status()

        def fake_get(url, timeout=None):
            path = url.replace("http://127.0.0.1:7860", "")
            return FakeResponse(test_client.get(path))

        def fake_post(url, json=None, timeout=None):
            path = url.replace("http://127.0.0.1:7860", "")
            return FakeResponse(test_client.post(path, json=json))

        with patch("client.requests.get", side_effect=fake_get), patch("client.requests.post", side_effect=fake_post):
            client = SupportInboxClient(base_url="http://127.0.0.1:7860")
            state = client.state()
            self.assertIn(state.task_id, {task.task_id for task in list_tasks()})
            reset_result = client.reset(task_id="billing_refund_triage")
            self.assertEqual(reset_result.observation.task_id, "billing_refund_triage")


if __name__ == "__main__":
    unittest.main()
