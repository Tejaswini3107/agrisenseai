from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from app.database import get_db, FarmerProfile

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
