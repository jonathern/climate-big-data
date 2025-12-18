from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
import numpy as np
from datetime import datetime

from src.api.schemas.request import PredictionRequest, BatchPredictionRequest
from src.api.schemas.response import PredictionResponse, BatchPredictionResponse
from src.inference.predictor import WeatherPredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Global predictor instance (loaded on startup)
predictor = None

def get_predictor() -> WeatherPredictor:
    """Dependency to get predictor instance"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return predictor

@router.post("/predict", response_model=PredictionResponse)
async def predict_weather_event(
    request: PredictionRequest,
    predictor: WeatherPredictor = Depends(get_predictor)
) -> PredictionResponse:
    """
    Predict extreme weather event for a single location
    
    - **station_id**: Weather station identifier
    - **features**: Dictionary of weather features (tmax, tmin, prcp, etc.)
    """
    try:
        start_time = datetime.now()
        
        # Prepare features
        feature_vector = predictor.prepare_features(request.features)
        
        # Make prediction
        prediction = predictor.predict(feature_vector)
        probabilities = predictor.predict_proba(feature_vector)
        
        # Get class name
        class_names = ['Normal', 'Drought', 'Flood', 'Heatwave', 'Cold']
        predicted_class = class_names[prediction[0]]
        
        # Calculate confidence
        confidence = float(np.max(probabilities[0]))
        
        # Get top 3 predictions
        top_3_indices = np.argsort(probabilities[0])[-3:][::-1]
        top_3_predictions = [
            {
                "event_type": class_names[idx],
                "probability": float(probabilities[0][idx])
            }
            for idx in top_3_indices
        ]
        
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info(f"Prediction made: {predicted_class} (confidence: {confidence:.3f})")
        
        return PredictionResponse(
            station_id=request.station_id,
            predicted_event=predicted_class,
            confidence=confidence,
            probabilities={
                name: float(prob) 
                for name, prob in zip(class_names, probabilities[0])
            },
            top_3_predictions=top_3_predictions,
            model_version=predictor.model_version,
            prediction_timestamp=datetime.now().isoformat(),
            latency_ms=latency_ms
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    predictor: WeatherPredictor = Depends(get_predictor)
) -> BatchPredictionResponse:
    """
    Predict extreme weather events for multiple locations
    """
    try:
        start_time = datetime.now()
        predictions = []
        
        for item in request.predictions:
            feature_vector = predictor.prepare_features(item.features)
            prediction = predictor.predict(feature_vector)
            probabilities = predictor.predict_proba(feature_vector)
            
            class_names = ['Normal', 'Drought', 'Flood', 'Heatwave', 'Cold']
            predicted_class = class_names[prediction[0]]
            confidence = float(np.max(probabilities[0]))
            
            predictions.append(
                PredictionResponse(
                    station_id=item.station_id,
                    predicted_event=predicted_class,
                    confidence=confidence,
                    probabilities={
                        name: float(prob) 
                        for name, prob in zip(class_names, probabilities[0])
                    },
                    model_version=predictor.model_version,
                    prediction_timestamp=datetime.now().isoformat()
                )
            )
        
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info(f"Batch prediction completed: {len(predictions)} predictions")
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_predictions=len(predictions),
            model_version=predictor.model_version,
            batch_timestamp=datetime.now().isoformat(),
            total_latency_ms=latency_ms
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")