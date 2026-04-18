from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../')

from app.routers import weather

# Initialize FastAPI application
app = FastAPI(
    title="AgriSense AI Backend",
    description="Climate and agricultural prediction API for crop recommendations, disease risk, and yield forecasting",
    version="1.0.0"
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(weather.router)


@app.get('/')
async def root():
    """
    Root endpoint - API information.
    """
    return {
        'name': 'AgriSense AI Backend',
        'version': '1.0.0',
        'description': 'Climate and agricultural prediction API',
        'endpoints': {
            'current_weather': '/api/weather/current?lat=<latitude>&lon=<longitude>',
            'forecast': '/api/weather/forecast?lat=<latitude>&lon=<longitude>',
            'health': '/api/weather/health',
            'docs': '/docs',
            'openapi': '/openapi.json'
        }
    }


@app.get('/health')
async def app_health():
    """
    Application health check.
    """
    return {
        'status': 'ok',
        'service': 'agrisense-backend'
    }


if __name__ == '__main__':
    import uvicorn
    
    # Run with: uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
    uvicorn.run(
        'app.main:app',
        host='0.0.0.0',
        port=8001,
        reload=True,
        log_level='info'
    )
