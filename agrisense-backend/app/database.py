from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.types import Enum as SAEnum
import os
from datetime import datetime, timezone

from app.services.crops import CropType

_db_user = os.getenv('DB_USER')
_db_password = os.getenv('DB_PASSWORD')
_db_host = os.getenv('DB_HOST')

# Use PostgreSQL when credentials are provided, otherwise fall back to SQLite for local dev.
if _db_user and _db_password and _db_host:
    DATABASE_URL = (
        f"postgresql://{_db_user}:{_db_password}@{_db_host}:5432/agrisense"
    )
    _connect_args = {}
else:
    _db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'agrisense_local.db')
    DATABASE_URL = f"sqlite:///{os.path.abspath(_db_path)}"
    _connect_args = {"check_same_thread": False}
    print(f"[DB] No PostgreSQL credentials found — using SQLite at {os.path.abspath(_db_path)}")


def utc_now():
    return datetime.now(timezone.utc)

engine = create_engine(DATABASE_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

CropEnum = SAEnum(
    CropType,
    name="crop_type_enum",
    native_enum=False,
    values_callable=lambda enum_cls: [item.value for item in enum_cls],
)


class FarmerProfile(Base):
    __tablename__ = "farmer_profiles"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    phone = Column(String)
    location = Column(String)
    created_at = Column(DateTime, default=utc_now)


class NotificationDevice(Base):
    __tablename__ = "notification_devices"

    id = Column(Integer, primary_key=True, index=True)
    user_uid = Column(String, index=True)
    role = Column(String, nullable=False, index=True)
    token = Column(String, unique=True, nullable=False, index=True)
    device_name = Column(String)
    platform = Column(String)
    preferred_language = Column(String)
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)


class AdminUser(Base):
    __tablename__ = "admin_users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=True)
    role = Column(String, default="admin")
    created_at = Column(DateTime, default=datetime.utcnow)


class RegisteredFarmer(Base):
    __tablename__ = "registered_farmers"

    id = Column(Integer, primary_key=True, index=True)
    google_uid = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    photo_url = Column(String)
    current_crop = Column(CropEnum, nullable=False, default=CropType.RICE.value)
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)
    last_login_at = Column(DateTime, default=utc_now, onupdate=utc_now)


class FarmerCropSearch(Base):
    __tablename__ = "farmer_crop_searches"

    id = Column(Integer, primary_key=True, index=True)
    farmer_id = Column(Integer, ForeignKey("registered_farmers.id"), nullable=False, index=True)
    crop = Column(CropEnum, nullable=False)
    source = Column(String, nullable=False, default="search")
    created_at = Column(DateTime, default=utc_now)


class AgriParamTemplate(Base):
    __tablename__ = "agriparam_templates"

    id = Column(Integer, primary_key=True, index=True)
    language = Column(String, unique=True, nullable=False, index=True)
    template_text = Column(String, nullable=False)
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)


def init_db():
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        print(f"Database connection failed: {e}")
        print("Running without database")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
