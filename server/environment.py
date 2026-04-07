from __future__ import annotations

from support_inbox.env import SupportInboxEnv


class SupportInboxEnvironment(SupportInboxEnv):
    """Tutorial-aligned alias for the support inbox environment."""


def build_environment() -> SupportInboxEnvironment:
    return SupportInboxEnvironment()
