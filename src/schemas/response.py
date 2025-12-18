from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class PredictionResponse(BaseModel):
    """Response schema for single prediction"""
    station_id: str
    predicted_event: str = Field(..., description="Predicted weather event type")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    probabilities: Dict[str, float] = Field(..., description="Probabilities for all classes")
    top_3_predictions: Optional[List[Dict[str, float]]] = None
    model_version: str
    prediction_timestamp: str
    latency_ms: Optional[float] = None

class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions"""
    predictions: List[PredictionResponse]
    total_predictions: int
    model_version: str
    batch_timestamp: str
    total_latency_ms: float