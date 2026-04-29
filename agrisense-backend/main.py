from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="AgriSenseAI API",
    description="Agricultural Intelligence Platform API",
    version="1.0.0"
)

# Configure CORS
origins = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/")
async def root():
    return {
        "message": "AgriSenseAI API is running",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "agrisense-backend"
    }

# Pest detection endpoint
@app.get("/api/pest-detection/{crop}")
async def get_pest_detection(crop: str):
    return {
        "crop": crop,
        "pests": [],
        "risk_level": "low",
        "recommendation": "Monitor crop regularly"
    }

# Crop health endpoint
@app.get("/api/crop-health")
async def get_crop_health():
    return {
        "crops": [
            {"name": "rice", "health": 85, "status": "healthy"},
            {"name": "wheat", "health": 78, "status": "good"},
            {"name": "cotton", "health": 72, "status": "fair"},
            {"name": "tomato", "health": 90, "status": "excellent"}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
