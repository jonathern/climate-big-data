from fastapi import APIRouter
from datetime import datetime
import psutil
import os

router = APIRouter()

@router.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "weather-ml-api"
    }

@router.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes"""
    # Check if model is loaded
    from src.api.routes.predict import predictor
    
    if predictor is None:
        return {
            "status": "not ready",
            "reason": "Model not loaded"
        }, 503
    
    return {
        "status": "ready",
        "model_loaded": True,
        "model_version": predictor.model_version
    }

@router.get("/metrics/system")
async def system_metrics():
    """System metrics endpoint"""
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent,
        "timestamp": datetime.now().isoformat()
    }