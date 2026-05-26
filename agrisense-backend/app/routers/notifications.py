from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from typing import Any, cast

from app.database import NotificationDevice, RegisteredFarmer, get_db
from app.services.push_onesignal import is_configured as is_push_configured, send_push_notification, send_push_to_topic
from app.services.notification_delivery import format_emergency_alert, is_email_configured, send_email_message, normalize_language

router = APIRouter(prefix="/api/notifications", tags=["Notifications"])


class RegisterTokenRequest(BaseModel):
    user_uid: str | None = None
    role: str = Field(pattern="^(farmer|admin)$")
    token: str
    device_name: str | None = None
    platform: str | None = None
    preferred_language: str | None = None


class SendNotificationRequest(BaseModel):
    roles: list[str] = Field(default_factory=lambda: ["farmer", "admin"])
    title: str
    body: str
    data: dict[str, str] = Field(default_factory=dict)


class EmergencyAlertRequest(BaseModel):
    roles: list[str] = Field(default_factory=lambda: ["farmer"])
    crop: str = "crop"
    location: str = "your area"
    detail: str
    severity: str = "critical"
    language: str | None = None
    send_email: bool = True
    data: dict[str, str] = Field(default_factory=dict)


@router.get("/status")
def notification_status() -> dict[str, Any]:
    return {
        "push_provider_configured": is_push_configured(),
        "email_configured": is_email_configured(),
        "roles": ["farmer", "admin"],
    }


@router.get("/status-readable")
def notification_status_readable(db: Session = Depends(get_db)) -> dict[str, Any]:
    devices = db.query(NotificationDevice).all()
    role_counts: dict[str, int] = {"farmer": 0, "admin": 0}
    language_counts: dict[str, int] = {}

    for device in devices:
        item = cast(Any, device)
        role = str(getattr(item, "role", "farmer"))
        role_counts[role] = role_counts.get(role, 0) + 1

        language = normalize_language(getattr(item, "preferred_language", None))
        language_counts[language] = language_counts.get(language, 0) + 1

    return {
        "push_provider_configured": is_push_configured(),
        "email_configured": is_email_configured(),
        "registered_devices_total": len(devices),
        "devices_by_role": role_counts,
        "devices_by_language": language_counts,
        "summary": (
            f"{len(devices)} device(s) registered. "
            f"Farmer: {role_counts.get('farmer', 0)}, Admin: {role_counts.get('admin', 0)}."
        ),
    }


@router.post("/register-token")
def register_token(payload: RegisterTokenRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    existing = db.query(NotificationDevice).filter(NotificationDevice.token == payload.token).first()
    if existing:
        existing_device = cast(Any, existing)
        existing_device.user_uid = payload.user_uid
        existing_device.role = payload.role
        existing_device.device_name = payload.device_name
        existing_device.platform = payload.platform
        existing_device.preferred_language = payload.preferred_language
        db.commit()
        db.refresh(existing)
        return {"status": "updated", "id": existing.id, "role": existing.role}

    record = NotificationDevice(
        user_uid=payload.user_uid,
        role=payload.role,
        token=payload.token,
        device_name=payload.device_name,
        platform=payload.platform,
        preferred_language=payload.preferred_language,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return {"status": "registered", "id": record.id, "role": record.role}


@router.post("/send")
def send_notification(payload: SendNotificationRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    if not is_push_configured():
        raise HTTPException(status_code=503, detail="Push provider is not configured")

    results: list[dict[str, Any]] = []
    for role in payload.roles:
        tokens = [
            row.token
            for row in db.query(NotificationDevice.token)
            .filter(NotificationDevice.role == role)
            .all()
        ]
        if not tokens:
            results.append({"role": role, "status": "no_tokens"})
            continue

        try:
            response = send_push_notification(tokens, payload.title, payload.body, payload.data)
            results.append({"role": role, "status": "sent", "responses": getattr(response, "responses", None)})
        except Exception as exc:
            results.append({"role": role, "status": "failed", "error": str(exc)})

    return {"status": "processed", "results": results}


@router.post("/emergency-alert")
def send_emergency_alert(payload: EmergencyAlertRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    if not is_push_configured():
        raise HTTPException(status_code=503, detail="Push provider is not configured")

    devices = db.query(NotificationDevice).filter(NotificationDevice.role.in_(payload.roles)).all()

    push_tokens_by_language: dict[str, list[str]] = {}
    email_groups: dict[str, set[str]] = {}

    for device in devices:
        device_obj = cast(Any, device)
        language = normalize_language(getattr(device_obj, "preferred_language", None) or payload.language)
        push_tokens_by_language.setdefault(language, []).append(str(device_obj.token))

        user_uid = getattr(device_obj, "user_uid", None)
        if user_uid:
            farmer = db.query(RegisteredFarmer).filter(RegisteredFarmer.google_uid == user_uid).first()
            if farmer is not None:
                farmer_obj = cast(Any, farmer)
                farmer_email = getattr(farmer_obj, "email", None)
                if farmer_email:
                    email_groups.setdefault(language, set()).add(str(farmer_email))

    results: list[dict[str, Any]] = []
    for language, tokens in push_tokens_by_language.items():
        localized = format_emergency_alert(
            language=language,
            crop=payload.crop,
            location=payload.location,
            detail=payload.detail,
            severity=payload.severity,
        )

        push_result: dict[str, Any] = {"status": "no_tokens", "token_count": 0}
        if tokens:
            response = send_push_notification(tokens, localized["title"], localized["body"], {
                **payload.data,
                "alert_type": "emergency",
                "language": language,
                "crop": payload.crop,
                "location": payload.location,
                "severity": payload.severity,
            })
            push_result = {
                "status": "sent",
                "token_count": len(tokens),
                "responses": getattr(response, "responses", None),
            }

        email_result: dict[str, Any] = {"status": "skipped", "recipient_count": 0}
        if payload.send_email:
            recipients = sorted(email_groups.get(language, set()))
            if recipients:
                send_email_message(recipients, localized["subject"], localized["email_body"])
                email_result = {"status": "sent", "recipient_count": len(recipients)}
            else:
                email_result = {"status": "no_recipients", "recipient_count": 0}

        results.append({
            "language": language,
            "push": push_result,
            "email": email_result,
        })

    return {
        "status": "processed",
        "results": results,
    }


@router.post("/send-topic")
def send_topic_notification(payload: SendNotificationRequest) -> dict[str, Any]:
    if not is_push_configured():
        raise HTTPException(status_code=503, detail="Push provider is not configured")

    results: list[dict[str, Any]] = []
    for role in payload.roles:
        try:
            message_id = send_push_to_topic(role, payload.title, payload.body, payload.data)
            results.append({"role": role, "status": "sent", "message_id": message_id})
        except Exception as exc:
            results.append({"role": role, "status": "failed", "error": str(exc)})

    return {"status": "processed", "results": results}