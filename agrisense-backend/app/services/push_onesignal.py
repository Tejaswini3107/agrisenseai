"""OneSignal push provider shim.

Environment variables required:
- `ONESIGNAL_APP_ID`
- `ONESIGNAL_API_KEY` (REST API key)

This module provides a minimal `send_push_notification` and `send_push_to_topic`
implementation so the rest of the app can call the same function names as before.
"""
from typing import Any, Dict, List
import os
import requests

ONESIGNAL_APP_ID = os.getenv("ONESIGNAL_APP_ID")
ONESIGNAL_API_KEY = os.getenv("ONESIGNAL_API_KEY")
ONESIGNAL_API_URL = "https://onesignal.com/api/v1/notifications"


def is_configured() -> bool:
    return bool(ONESIGNAL_APP_ID and ONESIGNAL_API_KEY)


def _post(payload: Dict[str, Any]) -> Any:
    headers = {
        "Authorization": f"Basic {ONESIGNAL_API_KEY}",
        "Content-Type": "application/json",
    }
    resp = requests.post(ONESIGNAL_API_URL, json=payload, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json()


def send_push_notification(tokens: List[str], title: str, body: str, data: Dict[str, str] | None = None) -> Any:
    if not tokens:
        return {"error": "no_tokens"}
    payload = {
        "app_id": ONESIGNAL_APP_ID,
        "include_player_ids": tokens,
        "headings": {"en": title},
        "contents": {"en": body},
        "data": data or {},
    }
    return _post(payload)


def send_push_to_topic(topic: str, title: str, body: str, data: Dict[str, str] | None = None) -> Any:
    # Use OneSignal segments for topics/roles; ensure segments exist in OneSignal console
    payload = {
        "app_id": ONESIGNAL_APP_ID,
        "included_segments": [topic],
        "headings": {"en": title},
        "contents": {"en": body},
        "data": data or {},
    }
    return _post(payload)
