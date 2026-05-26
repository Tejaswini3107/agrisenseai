from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from app.database import get_db, FarmerProfile, RegisteredFarmer, FarmerCropSearch, utc_now
from app.services.crops import CropType, normalize_crop
from app.services.auth_google import is_configured, verify_id_token

router = APIRouter(prefix="/api/farmers", tags=["Farmers"])


class FarmerCreate(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None
    location: Optional[str] = None


class FarmerUpdate(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None


class GoogleLoginRequest(BaseModel):
    id_token: str
    crop: Optional[CropType] = None


class CropSearchRequest(BaseModel):
    crop: CropType
    source: Optional[str] = "search"


def _serialize_registered_farmer(farmer: RegisteredFarmer) -> dict:
    return {
        "id": farmer.id,
        "google_uid": farmer.google_uid,
        "name": farmer.name,
        "email": farmer.email,
        "photo_url": farmer.photo_url,
        "current_crop": farmer.current_crop,
        "created_at": farmer.created_at,
        "updated_at": farmer.updated_at,
        "last_login_at": farmer.last_login_at,
    }


@router.post("/google-signin")
def google_signin(payload: GoogleLoginRequest, db: Session = Depends(get_db)):
    if not is_configured():
        raise HTTPException(status_code=503, detail="Google authentication is not configured")

    decoded_token = verify_id_token(payload.id_token)
    google_uid = decoded_token.get("uid")
    email = decoded_token.get("email")
    name = decoded_token.get("name") or decoded_token.get("displayName") or (email.split("@")[0] if email else "Farmer")
    photo_url = decoded_token.get("picture") or decoded_token.get("photoUrl")
    if not google_uid:
        raise HTTPException(status_code=400, detail="Google token did not include a UID")

    selected_crop = normalize_crop(payload.crop.value if payload.crop else None)
    farmer = db.query(RegisteredFarmer).filter(RegisteredFarmer.google_uid == google_uid).first()
    if not farmer and email:
        farmer = db.query(RegisteredFarmer).filter(RegisteredFarmer.email == email).first()

    created = False
    if not farmer:
        farmer = RegisteredFarmer(
            google_uid=google_uid,
            name=name,
            email=email or f"{google_uid}@google.local",
            photo_url=photo_url,
            current_crop=selected_crop,
            last_login_at=utc_now(),
        )
        db.add(farmer)
        created = True
    else:
        farmer.name = name
        if email:
            farmer.email = email
        farmer.photo_url = photo_url
        farmer.current_crop = selected_crop or farmer.current_crop
        farmer.last_login_at = utc_now()

    db.commit()
    db.refresh(farmer)

    return {
        "status": "created" if created else "updated",
        "farmer": _serialize_registered_farmer(farmer),
    }


@router.post("/{farmer_id}/crop-search")
def record_crop_search(farmer_id: int, payload: CropSearchRequest, db: Session = Depends(get_db)):
    farmer = db.query(RegisteredFarmer).filter(RegisteredFarmer.id == farmer_id).first()
    if not farmer:
        raise HTTPException(status_code=404, detail="Farmer not found")

    normalized_crop = normalize_crop(payload.crop.value)
    farmer.current_crop = normalized_crop
    farmer.last_login_at = utc_now()

    crop_event = FarmerCropSearch(
        farmer_id=farmer.id,
        crop=normalized_crop,
        source=payload.source or "search",
    )
    db.add(crop_event)
    db.commit()
    db.refresh(farmer)

    return {
        "status": "saved",
        "farmer": _serialize_registered_farmer(farmer),
        "crop_event": {
            "id": crop_event.id,
            "crop": crop_event.crop,
            "source": crop_event.source,
            "created_at": crop_event.created_at,
        },
    }


@router.post("/")
def create_farmer(farmer: FarmerCreate, db: Session = Depends(get_db)):
    existing = db.query(FarmerProfile).filter(
        FarmerProfile.email == farmer.email
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    db_farmer = FarmerProfile(**farmer.dict())
    db.add(db_farmer)
    db.commit()
    db.refresh(db_farmer)
    return db_farmer


@router.get("/")
def get_all_farmers(db: Session = Depends(get_db)):
    return db.query(FarmerProfile).all()


@router.get("/{farmer_id}")
def get_farmer(farmer_id: int, db: Session = Depends(get_db)):
    farmer = db.query(FarmerProfile).filter(
        FarmerProfile.id == farmer_id
    ).first()
    if not farmer:
        raise HTTPException(status_code=404, detail="Farmer not found")
    return farmer


@router.put("/{farmer_id}")
def update_farmer(
    farmer_id: int,
    updates: FarmerUpdate,
    db: Session = Depends(get_db)
):
    farmer = db.query(FarmerProfile).filter(
        FarmerProfile.id == farmer_id
    ).first()
    if not farmer:
        raise HTTPException(status_code=404, detail="Farmer not found")
    for key, value in updates.dict(exclude_none=True).items():
        setattr(farmer, key, value)
    db.commit()
    db.refresh(farmer)
    return farmer


@router.delete("/{farmer_id}")
def delete_farmer(farmer_id: int, db: Session = Depends(get_db)):
    farmer = db.query(FarmerProfile).filter(
        FarmerProfile.id == farmer_id
    ).first()
    if not farmer:
        raise HTTPException(status_code=404, detail="Farmer not found")
    db.delete(farmer)
    db.commit()
    return {"message": "Farmer deleted"}
