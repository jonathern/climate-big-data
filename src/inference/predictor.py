import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)

class WeatherPredictor:
    """Production predictor for extreme weather events"""
    
    def __init__(self, model_path: str):
        """
        Initialize predictor
        
        Args:
            model_path: Path to model directory
        """
        self.model_path = Path(model_path)
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.model_version = "unknown"
        
        self._load_model()
        self._load_metadata()
    
    def _load_model(self):
        """Load the trained model"""
        # Try loading XGBoost JSON format first
        model_file = self.model_path / "xgboost_model.json"
        
        if model_file.exists():
            try:
                # Load using native XGBoost booster
                import xgboost as xgb
                booster = xgb.Booster()
                booster.load_model(str(model_file))
                
                # Wrap in sklearn interface
                self.model = xgb.XGBClassifier()
                self.model._Booster = booster
                logger.info(f"Loaded XGBoost model from {model_file}")
                return
            except Exception as e:
                logger.warning(f"Failed to load JSON format: {e}")
        
        # Try loading pickle format
        model_file = self.model_path / "model.pkl"
        if model_file.exists():
            try:
                import pickle
                with open(model_file, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"Loaded model from {model_file} (pickle format)")
                return
            except Exception as e:
                logger.warning(f"Failed to load pickle format: {e}")
        
        raise FileNotFoundError(f"No model found at {self.model_path}")
    
    def _load_metadata(self):
        """Load model metadata"""
        metadata_file = self.model_path / "feature_names.json"
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                self.feature_names = metadata.get('feature_names', [])
                self.model_version = metadata.get('timestamp', 'unknown')
            logger.info(f"Loaded {len(self.feature_names)} feature names")
        
        # Load scaler if available
        scaler_file = self.model_path / "scaler.pkl"
        if scaler_file.exists():
            with open(scaler_file, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info("Loaded feature scaler")
    
    def prepare_features(self, features: Dict[str, float]) -> np.ndarray:
        """
        Prepare features for prediction
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Feature array ready for prediction
        """
        # If feature_names is defined, use it to order features
        if self.feature_names:
            feature_vector = np.array([
                features.get(name, 0.0) for name in self.feature_names
            ]).reshape(1, -1)
        else:
            # Use features as-is
            feature_vector = np.array(list(features.values())).reshape(1, -1)
        
        # Apply scaling if available
        if self.scaler is not None:
            feature_vector = self.scaler.transform(feature_vector)
        
        return feature_vector
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Make prediction
        
        Args:
            features: Feature array
            
        Returns:
            Predicted class
        """
        return self.model.predict(features)
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities
        
        Args:
            features: Feature array
            
        Returns:
            Class probabilities
        """
        return self.model.predict_proba(features)