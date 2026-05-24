from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.database import get_db, AdminUser
import bcrypt

router = APIRouter(prefix="/api/admin", tags=["Admin Auth"])


class AdminLogin(BaseModel):
    email: str
    password: str


@router.post("/verify")
def verify_admin(login: AdminLogin, db: Session = Depends(get_db)):
    admin = db.query(AdminUser).filter(
        AdminUser.email == login.email
    ).first()

    if not admin:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not admin.password:
        raise HTTPException(status_code=401, detail="Use Google login")

    if not bcrypt.checkpw(login.password.encode(), admin.password.encode()):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return {
        "id": admin.id,
        "name": admin.name,
        "email": admin.email,
        "role": admin.role
    }


@router.get("/check-email/{email}")
def check_email(email: str, db: Session = Depends(get_db)):
    admin = db.query(AdminUser).filter(
        AdminUser.email == email
    ).first()

    if not admin:
        raise HTTPException(status_code=401, detail="Not authorized")

    return {
        "id": admin.id,
        "name": admin.name,
        "email": admin.email,
        "role": admin.role
    }


@router.post("/create")
def create_admin(
    name: str,
    email: str,
    password: str,
    db: Session = Depends(get_db)
):
    existing = db.query(AdminUser).filter(
        AdminUser.email == email
    ).first()

    if existing:
        raise HTTPException(status_code=400, detail="Email already exists")

    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    admin = AdminUser(
        name=name,
        email=email,
        password=hashed,
        role="admin"
    )
    db.add(admin)
    db.commit()
    db.refresh(admin)

    return {
        "id": admin.id,
        "name": admin.name,
        "email": admin.email,
        "role": admin.role
    }


@router.get("/list")
def list_admins(db: Session = Depends(get_db)):
    admins = db.query(AdminUser).all()
    return [
        {
            "id": a.id,
            "name": a.name,
            "email": a.email,
            "role": a.role,
            "created_at": a.created_at
        }
        for a in admins
    ]


@router.delete("/delete/{admin_id}")
def delete_admin(admin_id: int, db: Session = Depends(get_db)):
    admin = db.query(AdminUser).filter(
        AdminUser.id == admin_id
    ).first()

    if not admin:
        raise HTTPException(status_code=404, detail="Admin not found")

    db.delete(admin)
    db.commit()
    return {"message": f"Admin {admin.name} deleted"}
