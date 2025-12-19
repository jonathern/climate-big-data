from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional

class PredictionRequest(BaseModel):
    """Request schema for single prediction"""
    station_id: str = Field(..., description="Weather station identifier")
    features: Dict[str, float] = Field(..., description="Weather features")
    
    @validator('features')
    def validate_features(cls, v):
        """Validate that required features are present"""
        required_features = ['tmax', 'tmin', 'prcp', 'latitude', 'longitude']
        for feature in required_features:
            if feature not in v:
                raise ValueError(f"Missing required feature: {feature}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "station_id": "STATION_001",
                "features": {
                    "tmax": 32.5,
                    "tmin": 18.2,
                    "prcp": 0.5,
                    "latitude": -1.2864,
                    "longitude": 36.8172,
                    "tmax_rolling_7d_mean": 31.2,
                    "prcp_rolling_30d_sum": 45.3,
                    "month": 3,
                    "season": 2
                }
            }
        }

class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions"""
    predictions: List[PredictionRequest] = Field(
        ..., 
        description="List of prediction requests",
        min_items=1,
        max_items=1000
    )