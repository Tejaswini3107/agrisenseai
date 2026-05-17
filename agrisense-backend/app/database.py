from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
import os
from datetime import datetime

DATABASE_URL = (
    f"postgresql://{os.getenv('agrisense_admin')}:{os.getenv('sEBtT6U9fswuCeG')}"
    f"@{os.getenv('agrisense-db.cfoqq0o00rh0.eu-north-1.rds.amazonaws.com')}:5432/agrisense"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class FarmerProfile(Base):
    __tablename__ = "farmer_profiles"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    phone = Column(String)
    location = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


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
