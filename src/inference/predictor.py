# src/inference/predictor.py
import json
import pickle
from pathlib import Path
from typing import Dict
import numpy as np

class WeatherPredictor:
    """Production predictor for extreme weather events"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None
        self.feature_names = []
        self.model_version = "unknown"
        self._use_raw_booster = False
        
        self._load_model()
        self._load_metadata()
    
    def _load_model(self):
        """Load the trained model"""
        # Try pickle format first (most reliable for sklearn wrapper)
        model_file = self.model_path / "model.pkl"
        if model_file.exists():
            try:
                with open(model_file, 'rb') as f:
                    self.model = pickle.load(f)
                print(f"✓ Loaded model from {model_file} (pickle format)")
                return
            except Exception as e:
                print(f"Warning: Could not load pickle format: {e}")
        
        # Try XGBoost JSON format
        model_file = self.model_path / "xgboost_model.json"
        
        if model_file.exists():
            try:
                import xgboost as xgb
                # Load the booster
                booster = xgb.Booster()
                booster.load_model(str(model_file))
                
                # Use raw booster for predictions
                self.model = booster
                self._use_raw_booster = True
                print(f"✓ Loaded XGBoost booster from {model_file}")
                return
            except Exception as e:
                print(f"Warning: Could not load JSON format: {e}")
        
        raise FileNotFoundError(f"No model found at {self.model_path}")
    
    def _load_metadata(self):
        """Load model metadata"""
        metadata_file = self.model_path / "feature_names.json"
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                self.feature_names = metadata.get('feature_names', [])
                self.model_version = metadata.get('timestamp', 'unknown')
            print(f"✓ Loaded {len(self.feature_names)} feature names")
    
    def prepare_features(self, features: Dict[str, float]) -> np.ndarray:
        """Prepare features for prediction"""
        if self.feature_names:
            feature_vector = np.array([
                features.get(name, 0.0) for name in self.feature_names
            ]).reshape(1, -1)
        else:
            feature_vector = np.array(list(features.values())).reshape(1, -1)
        
        return feature_vector
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make prediction"""
        if self._use_raw_booster:
            # Use raw booster for prediction
            import xgboost as xgb
            dmatrix = xgb.DMatrix(features)
            probs = self.model.predict(dmatrix)
            # Get class with highest probability
            return np.argmax(probs, axis=1)
        else:
            # Use sklearn wrapper
            return self.model.predict(features)
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if self._use_raw_booster:
            # Use raw booster for prediction
            import xgboost as xgb
            dmatrix = xgb.DMatrix(features)
            return self.model.predict(dmatrix)
        else:
            # Use sklearn wrapper
            return self.model.predict_proba(features)
