# src/api/main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Africa Extreme Weather ML API",
    description="API for predicting extreme weather events in Africa",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor = None

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global predictor
    logger.info("Starting up API server...")
    
    # Try to load the model
    model_path = Path("extreme_weather_ml/models")
    
    if model_path.exists():
        try:
            from src.inference.predictor import WeatherPredictor
            predictor = WeatherPredictor(model_path=str(model_path))
            logger.info("✓ Model loaded successfully")
        except Exception as e:
            logger.warning(f"⚠ Could not load model: {e}")
            logger.info("API will run without model (health checks only)")
    else:
        logger.warning(f"⚠ Model directory not found: {model_path}")
        logger.info("Run training first: python africa_extreme_weather_ml.py --mode local")

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Africa Extreme Weather ML API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "timestamp": time.time()
    }

@app.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes"""
    if predictor is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not ready",
                "reason": "Model not loaded"
            }
        )
    
    return {
        "status": "ready",
        "model_loaded": True,
        "model_version": getattr(predictor, 'model_version', 'unknown')
    }

@app.post("/predict")
async def predict(request: dict):
    """
    Predict extreme weather event
    
    Example request:
    {
        "station_id": "STATION_001",
        "features": {
            "tmax": 32.5,
            "tmin": 18.2,
            "prcp": 0.5,
            "latitude": -1.2864,
            "longitude": 36.8172,
            "tmax_rolling_7d_mean": 31.2,
            "tmax_rolling_30d_mean": 30.5,
            "prcp_rolling_30d_sum": 45.3,
            "month": 3,
            "day_of_year": 75,
            "season": 2
        }
    }
    """
    if predictor is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Model not loaded",
                "message": "Please train model first or check model path"
            }
        )
    
    try:
        import numpy as np
        
        station_id = request.get("station_id", "unknown")
        features = request.get("features", {})
        
        # Prepare features
        feature_vector = predictor.prepare_features(features)
        
        # Make prediction
        prediction = predictor.predict(feature_vector)
        probabilities = predictor.predict_proba(feature_vector)
        
        # Get class names
        class_names = ['Normal', 'Drought', 'Flood', 'Heatwave', 'Cold']
        predicted_class = class_names[int(prediction[0])]
        confidence = float(np.max(probabilities[0]))
        
        return {
            "station_id": station_id,
            "predicted_event": predicted_class,
            "confidence": confidence,
            "probabilities": {
                name: float(prob) 
                for name, prob in zip(class_names, probabilities[0])
            },
            "model_version": predictor.model_version
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
