"""Lightweight Google ID token verifier using Google's tokeninfo endpoint.

Requires at least `GOOGLE_OAUTH_WEB_CLIENT_ID` to be set in env for production.
"""
from typing import Any, Dict
import os
import requests


def _configured_client_ids() -> tuple[str | None, str | None]:
    return os.getenv("GOOGLE_OAUTH_WEB_CLIENT_ID"), os.getenv("GOOGLE_OAUTH_IOS_CLIENT_ID")


def is_configured() -> bool:
    web_client_id, ios_client_id = _configured_client_ids()
    return bool(web_client_id or ios_client_id)


def verify_id_token(id_token: str) -> Dict[str, Any]:
    """Verify an `id_token` using Google's tokeninfo endpoint.

    Returns a dict with at least `uid` (sub) and `email` on success.
    Raises Exception on verification failure.
    """
    if not id_token:
        raise ValueError("id_token is required")

    resp = requests.get("https://oauth2.googleapis.com/tokeninfo", params={"id_token": id_token}, timeout=5)
    if resp.status_code != 200:
        raise Exception(f"Invalid ID token: {resp.status_code} {resp.text}")

    data = resp.json()
    # Validate audience against provided client IDs if configured
    web_client_id, ios_client_id = _configured_client_ids()
    aud = data.get("aud")
    if web_client_id and aud != web_client_id:
        # allow ios client id match as well
        if ios_client_id and aud != ios_client_id:
            raise Exception("ID token audience does not match configured client IDs")

    # Normalize fields for app-wide user profile shape
    return {
        "uid": data.get("sub"),
        "email": data.get("email"),
        "name": data.get("name") or data.get("email"),
        "picture": data.get("picture"),
        "claims": data,
    }
