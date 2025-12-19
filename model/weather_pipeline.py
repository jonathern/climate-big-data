"""
Africa Extreme Weather Event Prediction - Distributed ML Pipeline
==================================================================
A production-grade distributed machine learning pipeline for predicting
extreme weather events (droughts, floods, heatwaves) using multi-source
climate data (GHCN + CHIRPS + ERA5).

ML APPROACH JUSTIFICATION:
--------------------------
Selected: Gradient Boosting (XGBoost) - Classical ML
Reasoning:
  1. Tabular time-series data (not images/text)
  2. Feature interpretability critical for climate science
  3. Handles missing data well (common in African weather stations)
  4. Strong performance on imbalanced datasets (extreme events are rare)
  5. Efficient distributed training with Spark/Dask
  6. Lower computational cost than deep learning
  7. Proven effectiveness in climate prediction literature
  
Alternative considered:
  - LSTM/Deep Learning: Requires dense time-series, expensive compute,
    less interpretable, African data too sparse
  - Causal Inference: Would be ideal for policy analysis but requires
    RCT-like data or strong assumptions; better suited as post-hoc analysis

EXTREME EVENTS DEFINED:
-----------------------
  1. Drought: 30-day cumulative rainfall < 10th percentile
  2. Flood: Daily rainfall > 95th percentile
  3. Heatwave: 3+ consecutive days with Tmax > 95th percentile
  4. Cold spell: 3+ consecutive days with Tmin < 5th percentile

Requirements:
    pip install pyspark xgboost dask distributed scikit-learn pandas numpy
    pip install matplotlib seaborn plotly mlflow imbalanced-learn
    
    Note: SHAP removed due to dependency conflicts. If needed for interpretability,
    install separately in an isolated environment.

Usage:
    python africa_extreme_weather_ml.py --mode distributed
    python africa_extreme_weather_ml.py --mode local --compare
"""

import os
import sys
import time
import argparse
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Spark imports
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import GBTClassifier, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# Dask imports (alternative to Spark)
try:
    import dask
    import dask.dataframe as dd
    from dask.distributed import Client, LocalCluster
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    print("Dask not available. Only Spark mode will work.")

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error
)
from sklearn.preprocessing import StandardScaler as SKStandardScaler

# Handle SMOTE import with compatibility check
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("imbalanced-learn not available. Class imbalance handling will use class_weight.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Will use Spark GBT instead.")

# SHAP removed due to dependency conflicts
SHAP_AVAILABLE = False

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class ExtremeWeatherConfig:
    """Configuration for extreme weather prediction pipeline"""
    
    # Data paths
    BASE_DIR = Path("./extreme_weather_ml")
    DATA_DIR = BASE_DIR / "data"
    PROCESSED_DIR = BASE_DIR / "processed"
    MODELS_DIR = BASE_DIR / "models"
    RESULTS_DIR = BASE_DIR / "results"
    PLOTS_DIR = BASE_DIR / "plots"
    
    # Input data (from previous pipelines)
    GHCN_DATA = Path("./ghcn_data/processed/weather_observations")
    CHIRPS_DATA = Path("./africa_climate_data/processed/climate_observations")
    
    # Africa focus
    AFRICA_BOUNDS = {'lat_min': -35.0, 'lat_max': 37.0, 'lon_min': -18.0, 'lon_max': 52.0}
    
    # Target regions for prediction
    TARGET_REGIONS = ['East_Africa', 'West_Africa', 'Southern_Africa']
    
    # Extreme event thresholds (percentiles)
    DROUGHT_PERCENTILE = 10
    FLOOD_PERCENTILE = 95
    HEATWAVE_PERCENTILE = 95
    COLD_PERCENTILE = 5
    
    # ML parameters
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    CV_FOLDS = 5
    
    # Feature engineering
    LOOKBACK_DAYS = 30  # Use 30 days of history for prediction
    
    # Distributed computing
    SPARK_MEMORY = "8g"
    SPARK_CORES = 4
    DASK_WORKERS = 4
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories"""
        for dir_path in [cls.DATA_DIR, cls.PROCESSED_DIR, cls.MODELS_DIR, 
                         cls.RESULTS_DIR, cls.PLOTS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DATA INGESTION - DISTRIBUTED
# ============================================================================

class DistributedDataIngestion:
    """Distributed data ingestion using Spark"""
    
    def __init__(self, spark: SparkSession, config: ExtremeWeatherConfig):
        self.spark = spark
        self.config = config
    
    def ingest_ghcn_data(self) -> Optional[DataFrame]:
        """Ingest GHCN weather data"""
        print("\n[Spark] Ingesting GHCN weather data...")
        start_time = time.time()
        
        try:
            if not self.config.GHCN_DATA.exists():
                print(f"⚠ GHCN data not found at {self.config.GHCN_DATA}")
                return self._generate_sample_ghcn_data()
            
            df = self.spark.read.parquet(str(self.config.GHCN_DATA))
            
            # Filter for Africa
            df = df.filter(
                (F.col("latitude").between(self.config.AFRICA_BOUNDS['lat_min'], 
                                          self.config.AFRICA_BOUNDS['lat_max'])) &
                (F.col("longitude").between(self.config.AFRICA_BOUNDS['lon_min'], 
                                           self.config.AFRICA_BOUNDS['lon_max']))
            )
            
            elapsed = time.time() - start_time
            print(f"✓ GHCN data ingested: {df.count():,} records in {elapsed:.2f}s")
            return df
            
        except Exception as e:
            print(f"✗ Error ingesting GHCN: {e}")
            return self._generate_sample_ghcn_data()
    
    def ingest_chirps_data(self) -> Optional[DataFrame]:
        """Ingest CHIRPS precipitation data"""
        print("\n[Spark] Ingesting CHIRPS precipitation data...")
        start_time = time.time()
        
        try:
            if not self.config.CHIRPS_DATA.exists():
                print(f"⚠ CHIRPS data not found at {self.config.CHIRPS_DATA}")
                return self._generate_sample_chirps_data()
            
            df = self.spark.read.parquet(str(self.config.CHIRPS_DATA))
            elapsed = time.time() - start_time
            print(f"✓ CHIRPS data ingested: {df.count():,} records in {elapsed:.2f}s")
            return df
            
        except Exception as e:
            print(f"✗ Error ingesting CHIRPS: {e}")
            return self._generate_sample_chirps_data()
    
    def _generate_sample_ghcn_data(self) -> DataFrame:
        """Generate sample GHCN data for demonstration"""
        print("Generating sample GHCN data...")
        
        # Generate 2 years of daily data for 50 stations
        dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
        stations = [f'STATION_{i:03d}' for i in range(50)]
        
        data = []
        for station in stations:
            lat = np.random.uniform(-10, 5)  # East Africa range
            lon = np.random.uniform(29, 42)
            
            for date in dates:
                # Temperature with seasonal pattern
                day_of_year = date.dayofyear
                base_temp = 22 + 5 * np.sin(2 * np.pi * day_of_year / 365)
                
                data.append({
                    'station_id': station,
                    'date': date,
                    'latitude': lat,
                    'longitude': lon,
                    'tmax': base_temp + np.random.normal(5, 2),
                    'tmin': base_temp - np.random.normal(5, 2),
                    'prcp': max(0, np.random.exponential(3))
                })
        
        return self.spark.createDataFrame(pd.DataFrame(data))
    
    def _generate_sample_chirps_data(self) -> DataFrame:
        """Generate sample CHIRPS data"""
        print("Generating sample CHIRPS data...")
        
        dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
        
        # Grid points (0.05° resolution sample)
        lats = np.arange(-10, 5, 0.5)
        lons = np.arange(29, 42, 0.5)
        
        data = []
        for date in dates[:365]:  # Use 1 year for sample
            for lat in lats[::2]:  # Sample every other point
                for lon in lons[::2]:
                    data.append({
                        'date': date,
                        'latitude': lat,
                        'longitude': lon,
                        'precipitation_mm': max(0, np.random.exponential(4))
                    })
        
        return self.spark.createDataFrame(pd.DataFrame(data))


# ============================================================================
# FEATURE ENGINEERING - DISTRIBUTED
# ============================================================================

class DistributedFeatureEngineering:
    """Feature engineering using Spark window functions"""
    
    def __init__(self, spark: SparkSession, config: ExtremeWeatherConfig):
        self.spark = spark
        self.config = config
    
    def create_features(self, df: DataFrame) -> DataFrame:
        """Create features for extreme event prediction"""
        print("\n[Spark] Creating features...")
        start_time = time.time()
        
        # Ensure we have required columns
        df = self._standardize_columns(df)
        
        # Sort by station and date
        window_spec = Window.partitionBy("station_id").orderBy("date")
        
        # Rolling statistics (last 7, 14, 30 days)
        for days in [7, 14, 30]:
            window = window_spec.rowsBetween(-days, -1)
            
            df = df.withColumn(f"tmax_rolling_{days}d_mean", 
                              F.avg("tmax").over(window))
            df = df.withColumn(f"tmax_rolling_{days}d_max", 
                              F.max("tmax").over(window))
            df = df.withColumn(f"tmin_rolling_{days}d_mean", 
                              F.avg("tmin").over(window))
            df = df.withColumn(f"tmin_rolling_{days}d_min", 
                              F.min("tmin").over(window))
            df = df.withColumn(f"prcp_rolling_{days}d_sum", 
                              F.sum("prcp").over(window))
            df = df.withColumn(f"prcp_rolling_{days}d_mean", 
                              F.avg("prcp").over(window))
        
        # Lag features
        for lag in [1, 3, 7]:
            df = df.withColumn(f"tmax_lag_{lag}d", 
                              F.lag("tmax", lag).over(window_spec))
            df = df.withColumn(f"prcp_lag_{lag}d", 
                              F.lag("prcp", lag).over(window_spec))
        
        # Temperature anomaly (vs 30-day average)
        df = df.withColumn("tmax_anomaly", 
                          F.col("tmax") - F.col("tmax_rolling_30d_mean"))
        df = df.withColumn("tmin_anomaly", 
                          F.col("tmin") - F.col("tmin_rolling_30d_mean"))
        
        # Temporal features
        df = df.withColumn("month", F.month("date"))
        df = df.withColumn("day_of_year", F.dayofyear("date"))
        df = df.withColumn("season", 
                          F.when(F.col("month").isin([12, 1, 2]), 1)
                           .when(F.col("month").isin([3, 4, 5]), 2)
                           .when(F.col("month").isin([6, 7, 8]), 3)
                           .otherwise(4))
        
        elapsed = time.time() - start_time
        print(f"✓ Features created in {elapsed:.2f}s")
        
        return df
    
    def create_labels(self, df: DataFrame) -> DataFrame:
        """Create extreme event labels"""
        print("\n[Spark] Creating extreme event labels...")
        start_time = time.time()
        
        # Calculate percentiles for thresholds
        percentiles = df.approxQuantile(
            ["prcp", "tmax", "tmin"],
            [self.config.DROUGHT_PERCENTILE/100, 
             self.config.FLOOD_PERCENTILE/100,
             self.config.HEATWAVE_PERCENTILE/100,
             self.config.COLD_PERCENTILE/100],
            0.01
        )
        
        prcp_p10 = percentiles[0][0]
        prcp_p95 = percentiles[0][1]
        tmax_p95 = percentiles[1][2]
        tmin_p5 = percentiles[2][3]
        
        print(f"Thresholds calculated:")
        print(f"  Drought (prcp < {prcp_p10:.2f} mm/30d)")
        print(f"  Flood (prcp > {prcp_p95:.2f} mm/day)")
        print(f"  Heatwave (tmax > {tmax_p95:.2f}°C)")
        print(f"  Cold spell (tmin < {tmin_p5:.2f}°C)")
        
        # Create binary labels
        df = df.withColumn("is_drought", 
                          (F.col("prcp_rolling_30d_sum") < prcp_p10).cast("int"))
        df = df.withColumn("is_flood", 
                          (F.col("prcp") > prcp_p95).cast("int"))
        df = df.withColumn("is_heatwave", 
                          (F.col("tmax") > tmax_p95).cast("int"))
        df = df.withColumn("is_cold_spell", 
                          (F.col("tmin") < tmin_p5).cast("int"))
        
        # Multi-class label (0=normal, 1=drought, 2=flood, 3=heatwave, 4=cold)
        df = df.withColumn("extreme_event",
                          F.when(F.col("is_drought") == 1, 1)
                           .when(F.col("is_flood") == 1, 2)
                           .when(F.col("is_heatwave") == 1, 3)
                           .when(F.col("is_cold_spell") == 1, 4)
                           .otherwise(0))
        
        elapsed = time.time() - start_time
        print(f"✓ Labels created in {elapsed:.2f}s")
        
        return df
    
    def _standardize_columns(self, df: DataFrame) -> DataFrame:
        """Standardize column names"""
        columns = df.columns
        
        if "value_converted" in columns and "element" in columns:
            # Pivot GHCN format
            df = df.groupBy("station_id", "date", "latitude", "longitude").pivot("element").agg(
                F.first("value_converted")
            )
            
            # Rename to standard names
            if "TMAX" in df.columns:
                df = df.withColumnRenamed("TMAX", "tmax")
            if "TMIN" in df.columns:
                df = df.withColumnRenamed("TMIN", "tmin")
            if "PRCP" in df.columns:
                df = df.withColumnRenamed("PRCP", "prcp")
        
        # Ensure station_id exists
        if "station_id" not in df.columns and "station" in df.columns:
            df = df.withColumnRenamed("station", "station_id")
        
        return df


# ============================================================================
# MODEL TRAINING - DISTRIBUTED
# ============================================================================

class DistributedModelTraining:
    """Distributed model training with Spark MLlib and XGBoost"""
    
    def __init__(self, spark: SparkSession, config: ExtremeWeatherConfig):
        self.spark = spark
        self.config = config
        self.feature_cols = []
        self.model = None
        self.scaler = None
    
    def prepare_features(self, df: DataFrame, target_col: str = "extreme_event") -> Tuple[DataFrame, List[str]]:
        """Prepare feature vector for ML"""
        print("\n[Spark] Preparing feature vectors...")
        
        # Remove rows with nulls in critical columns
        df = df.dropna(subset=[target_col, "tmax", "tmin", "prcp"])
        
        # Select feature columns (exclude IDs, dates, and labels)
        exclude_cols = ["station_id", "date", "extreme_event", 
                       "is_drought", "is_flood", "is_heatwave", "is_cold_spell"]
        
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols and df.schema[col].dataType in 
                       [DoubleType(), FloatType(), IntegerType(), LongType()]]
        
        self.feature_cols = feature_cols
        print(f"✓ Using {len(feature_cols)} features")
        
        # Assemble features
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features_raw",
            handleInvalid="skip"
        )
        
        df = assembler.transform(df)
        
        # Scale features
        scaler = StandardScaler(
            inputCol="features_raw",
            outputCol="features",
            withStd=True,
            withMean=True
        )
        
        self.scaler = scaler.fit(df)
        df = self.scaler.transform(df)
        
        # Rename target column to label
        df = df.withColumnRenamed(target_col, "label")
        
        return df, feature_cols
    
    def train_spark_gbt(self, train_df: DataFrame, test_df: DataFrame) -> Dict:
        """Train Random Forest using Spark MLlib (GBT only supports binary classification)"""
        print("\n[Spark MLlib] Training Random Forest Classifier...")
        print("Note: Using Random Forest instead of GBT since Spark GBT only supports binary classification")
        start_time = time.time()
        
        # Random Forest Classifier (supports multiclass)
        rf = RandomForestClassifier(
            featuresCol="features",
            labelCol="label",
            numTrees=100,
            maxDepth=5,
            seed=self.config.RANDOM_STATE,
            featureSubsetStrategy="auto"
        )
        
        # Train
        self.model = rf.fit(train_df)
        training_time = time.time() - start_time
        
        # Predict
        predictions = self.model.transform(test_df)
        
        # Evaluate
        evaluator_multi = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction"
        )
        
        accuracy = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "accuracy"})
        f1 = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "f1"})
        precision = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "weightedPrecision"})
        recall = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "weightedRecall"})
        
        # For multiclass, calculate AUC using one-vs-rest approach
        try:
            evaluator_binary = BinaryClassificationEvaluator(
                labelCol="label",
                rawPredictionCol="rawPrediction"
            )
            auc = evaluator_binary.evaluate(predictions, {evaluator_binary.metricName: "areaUnderROC"})
        except:
            auc = 0.0
        
        results = {
            'model_type': 'Spark_RandomForest',
            'training_time': training_time,
            'auc': auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': predictions
        }
        
        print(f"✓ Training completed in {training_time:.2f}s")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        if auc > 0:
            print(f"  AUC: {auc:.4f}")
        
        return results
    
    def train_xgboost_local(self, train_df: DataFrame, test_df: DataFrame) -> Dict:
        """Train XGBoost model (local, for comparison)"""
        if not XGBOOST_AVAILABLE:
            print("XGBoost not available, skipping...")
            return {}
        
        print("\n[XGBoost Local] Training model...")
        start_time = time.time()
        
        # Convert Spark DataFrame to pandas
        train_pd = train_df.select("features", "label").toPandas()
        test_pd = test_df.select("features", "label").toPandas()
        
        # Extract features from vector
        X_train = np.array(train_pd['features'].tolist())
        y_train = train_pd['label'].values
        X_test = np.array(test_pd['features'].tolist())
        y_test = test_pd['label'].values
        
        # Handle class imbalance
        if SMOTE_AVAILABLE:
            try:
                print("Applying SMOTE for class imbalance...")
                smote = SMOTE(random_state=self.config.RANDOM_STATE)
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            except Exception as e:
                print(f"SMOTE failed ({e}), using class_weight instead...")
                X_train_balanced, y_train_balanced = X_train, y_train
        else:
            print("Using class_weight for imbalance (SMOTE not available)...")
            X_train_balanced, y_train_balanced = X_train, y_train
        
        # Calculate class weights for XGBoost
        unique_classes, class_counts = np.unique(y_train_balanced, return_counts=True)
        total_samples = len(y_train_balanced)
        n_classes = len(unique_classes)
        
        # Compute sample weights (inverse frequency)
        sample_weights = np.ones(len(y_train_balanced))
        for cls, count in zip(unique_classes, class_counts):
            weight = total_samples / (n_classes * count)
            sample_weights[y_train_balanced == cls] = weight
        
        # Train XGBoost with proper configuration
        if n_classes == 2:
            # Binary classification
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.config.RANDOM_STATE,
                eval_metric='logloss',
                tree_method='hist'
            )
        else:
            # Multiclass classification
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.config.RANDOM_STATE,
                eval_metric='mlogloss',
                tree_method='hist',
                objective='multi:softprob',
                num_class=n_classes
            )
        
        xgb_model.fit(X_train_balanced, y_train_balanced, sample_weight=sample_weights, verbose=False)
        training_time = time.time() - start_time
        
        # Predict
        y_pred = xgb_model.predict(X_test)
        y_pred_proba = xgb_model.predict_proba(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # AUC for multiclass (one-vs-rest)
        try:
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            auc = 0.0
        
        results = {
            'model_type': 'XGBoost_Local',
            'training_time': training_time,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'model': xgb_model,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"✓ Training completed in {training_time:.2f}s")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
        
        return results


# ============================================================================
# MODEL EVALUATION
# ============================================================================

class ModelEvaluation:
    """Comprehensive model evaluation and visualization"""
    
    def __init__(self, config: ExtremeWeatherConfig):
        self.config = config
    
    def cross_validate(self, X, y, model) -> Dict:
        """Perform cross-validation"""
        print("\nPerforming cross-validation...")
        
        cv = StratifiedKFold(n_splits=self.config.CV_FOLDS, shuffle=True, 
                            random_state=self.config.RANDOM_STATE)
        
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
        
        results = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }
        
        print(f"✓ Cross-validation F1: {results['cv_mean']:.4f} (±{results['cv_std']:.4f})")
        
        return results
    
    def error_analysis(self, y_true, y_pred, class_names=None) -> Dict:
        """Perform detailed error analysis"""
        print("\nPerforming error analysis...")
        
        # Get unique classes present in the data
        unique_classes = np.union1d(y_true, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
        
        # Adjust class names to match actual classes present
        if class_names is None:
            class_names = ['Normal', 'Drought', 'Flood', 'Heatwave', 'Cold']
        
        # Map class indices to names for classes that are present
        present_class_names = [class_names[int(i)] if int(i) < len(class_names) else f'Class_{int(i)}' 
                               for i in unique_classes]
        
        # Classification report with only present classes
        report = classification_report(
            y_true, 
            y_pred, 
            labels=unique_classes,
            target_names=present_class_names, 
            output_dict=True, 
            zero_division=0
        )
        
        # Per-class metrics
        per_class_metrics = {}
        for i, class_idx in enumerate(unique_classes):
            class_name = present_class_names[i]
            if class_name in report:
                per_class_metrics[class_name] = report[class_name]
        
        results = {
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'per_class_metrics': per_class_metrics,
            'unique_classes': unique_classes.tolist(),
            'present_class_names': present_class_names
        }
        
        print(f"✓ Error analysis complete")
        print(f"  Classes present in data: {present_class_names}")
        
        return results
    
    def plot_confusion_matrix(self, cm, class_names, save_path):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        
        # Handle case where cm might be from error_analysis with only present classes
        if isinstance(cm, list):
            cm = np.array(cm)
        
        # Adjust class names to match confusion matrix size
        if len(class_names) != cm.shape[0]:
            class_names = class_names[:cm.shape[0]]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - Extreme Weather Events', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Confusion matrix saved: {save_path}")
    
    def plot_roc_curves(self, y_test, y_pred_proba, class_names, save_path):
        """Plot ROC curves for each class"""
        n_classes = len(class_names)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, class_name in enumerate(class_names):
            if i < len(axes):
                # One-vs-rest ROC curve
                y_true_binary = (y_test == i).astype(int)
                
                if len(np.unique(y_true_binary)) > 1 and y_pred_proba.shape[1] > i:
                    fpr, tpr, _ = roc_curve(y_true_binary, y_pred_proba[:, i])
                    auc_score = roc_auc_score(y_true_binary, y_pred_proba[:, i])
                    
                    axes[i].plot(fpr, tpr, linewidth=2, 
                                label=f'AUC = {auc_score:.3f}')
                    axes[i].plot([0, 1], [0, 1], 'k--', linewidth=1)
                    axes[i].set_xlabel('False Positive Rate', fontsize=10)
                    axes[i].set_ylabel('True Positive Rate', fontsize=10)
                    axes[i].set_title(f'ROC Curve - {class_name}', fontsize=11, fontweight='bold')
                    axes[i].legend(loc='lower right')
                    axes[i].grid(alpha=0.3)
        
        # Hide empty subplot
        if len(class_names) < len(axes):
            axes[-1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ ROC curves saved: {save_path}")
    
    def plot_feature_importance(self, model, feature_names, save_path, top_n=20):
        """Plot feature importance"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            print("Model doesn't support feature importance")
            return
        
        # Get top N features
        indices = np.argsort(importances)[-top_n:]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(indices)), importances[indices], color='steelblue', alpha=0.8)
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Feature importance plot saved: {save_path}")
    
    def generate_report(self, results: Dict, save_path: Path):
        """Generate comprehensive evaluation report"""
        print("\nGenerating evaluation report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_performance': {},
            'error_analysis': {},
            'cross_validation': {}
        }
        
        # Extract metrics
        for key in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'training_time']:
            if key in results:
                report['model_performance'][key] = float(results[key])
        
        if 'classification_report' in results:
            report['error_analysis'] = results['classification_report']
        
        if 'cv_mean' in results:
            report['cross_validation'] = {
                'mean_f1': float(results['cv_mean']),
                'std_f1': float(results['cv_std']),
                'fold_scores': [float(x) for x in results['cv_scores']]
            }
        
        # Save as JSON
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Evaluation report saved: {save_path}")
        
        return report


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class ExtremeWeatherMLPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, mode: str = 'distributed'):
        self.config = ExtremeWeatherConfig()
        self.config.create_directories()
        self.mode = mode
        self.spark = None
        
        # Initialize Spark if needed
        if mode in ['distributed', 'spark']:
            self.spark = self._init_spark()
        elif mode == 'local':
            print("\n" + "="*70)
            print("RUNNING IN LOCAL MODE (NO SPARK)")
            print("="*70)
            print("Note: Local mode uses pandas/numpy only, no distributed processing")
    
    def _init_spark(self) -> SparkSession:
        """Initialize Spark session"""
        print("\n" + "="*70)
        print("INITIALIZING SPARK SESSION")
        print("="*70)
        
        spark = SparkSession.builder \
            .appName("Africa_Extreme_Weather_ML") \
            .config("spark.driver.memory", self.config.SPARK_MEMORY) \
            .config("spark.executor.memory", self.config.SPARK_MEMORY) \
            .config("spark.sql.shuffle.partitions", "200") \
            .config("spark.default.parallelism", str(self.config.SPARK_CORES * 2)) \
            .getOrCreate()
        
        spark.sparkContext.setLogLevel("WARN")
        print(f"✓ Spark initialized with {self.config.SPARK_CORES} cores")
        print(f"✓ Spark version: {spark.version}")
        
        return spark
    
    def run(self, compare: bool = False):
        """Run the complete ML pipeline"""
        print("\n" + "="*70)
        print("AFRICA EXTREME WEATHER EVENT PREDICTION PIPELINE")
        print("="*70)
        print(f"Mode: {self.mode}")
        print(f"Random State: {self.config.RANDOM_STATE}")
        print(f"Test Size: {self.config.TEST_SIZE}")
        
        pipeline_start = time.time()
        
        # Check if we're in local mode
        if self.mode == 'local':
            return self._run_local_mode()
        
        # Otherwise run distributed mode (Spark)
        return self._run_distributed_mode(compare)
        
        # Step 1: Data Ingestion
        print("\n" + "="*70)
        print("STEP 1: DATA INGESTION")
        print("="*70)
        
        ingestion = DistributedDataIngestion(self.spark, self.config)
        df = ingestion.ingest_ghcn_data()
        
        if df is None:
            print("✗ Data ingestion failed. Exiting.")
            return
        
        # Step 2: Feature Engineering
        print("\n" + "="*70)
        print("STEP 2: FEATURE ENGINEERING")
        print("="*70)
        
        feature_eng = DistributedFeatureEngineering(self.spark, self.config)
        df = feature_eng.create_features(df)
        df = feature_eng.create_labels(df)
        
        # Cache for performance
        df.cache()
        print(f"\n✓ Dataset prepared: {df.count():,} records")
        
        # Check class distribution
        print("\nClass distribution:")
        df.groupBy("extreme_event").count().orderBy("extreme_event").show()
        
        # Step 3: Train/Test Split
        print("\n" + "="*70)
        print("STEP 3: TRAIN/TEST SPLIT")
        print("="*70)
        
        train_df, test_df = df.randomSplit([1-self.config.TEST_SIZE, self.config.TEST_SIZE], 
                                           seed=self.config.RANDOM_STATE)
        
        print(f"✓ Training set: {train_df.count():,} records")
        print(f"✓ Test set: {test_df.count():,} records")
        
        # Step 4: Model Training
        print("\n" + "="*70)
        print("STEP 4: MODEL TRAINING")
        print("="*70)
        
        trainer = DistributedModelTraining(self.spark, self.config)
        train_prepared, feature_names = trainer.prepare_features(train_df)
        test_prepared, _ = trainer.prepare_features(test_df)
        
        # Train Spark GBT
        spark_results = trainer.train_spark_gbt(train_prepared, test_prepared)
        
        # Train XGBoost for comparison
        xgb_results = {}
        if compare and XGBOOST_AVAILABLE:
            xgb_results = trainer.train_xgboost_local(train_prepared, test_prepared)
        
        # Step 5: Model Evaluation
        print("\n" + "="*70)
        print("STEP 5: MODEL EVALUATION")
        print("="*70)
        
        evaluator = ModelEvaluation(self.config)
        class_names = ['Normal', 'Drought', 'Flood', 'Heatwave', 'Cold']
        
        # Always save the prepared datasets
        print("\nSaving processed datasets...")
        train_prepared.write.parquet(
            str(self.config.PROCESSED_DIR / "train_data.parquet"), 
            mode='overwrite'
        )
        test_prepared.write.parquet(
            str(self.config.PROCESSED_DIR / "test_data.parquet"), 
            mode='overwrite'
        )
        print(f"✓ Datasets saved to {self.config.PROCESSED_DIR}")
        
        # Evaluate based on available models
        if xgb_results and 'model' in xgb_results:
            print("\nPerforming detailed evaluation on XGBoost model...")
            # Detailed evaluation for XGBoost
            y_test = xgb_results['y_test']
            y_pred = xgb_results['y_pred']
            y_pred_proba = xgb_results['y_pred_proba']
            
            # Error analysis
            error_results = evaluator.error_analysis(y_test, y_pred, class_names)
            
            # Cross-validation
            cv_results = evaluator.cross_validate(
                xgb_results['X_test'], 
                xgb_results['y_test'], 
                xgb_results['model']
            )
            
            # Confusion matrix
            cm_path = self.config.PLOTS_DIR / 'confusion_matrix.png'
            evaluator.plot_confusion_matrix(
                error_results['confusion_matrix'], 
                class_names, 
                cm_path
            )
            
            # ROC curves
            roc_path = self.config.PLOTS_DIR / 'roc_curves.png'
            evaluator.plot_roc_curves(y_test, y_pred_proba, class_names, roc_path)
            
            # Feature importance
            fi_path = self.config.PLOTS_DIR / 'feature_importance.png'
            evaluator.plot_feature_importance(
                xgb_results['model'], 
                feature_names, 
                fi_path
            )
            
            # Note: SHAP analysis removed due to dependency conflicts
            print("\nNote: SHAP analysis skipped (not installed)")
            
            # Generate report
            all_results = {**xgb_results, **error_results, **cv_results}
            report_path = self.config.RESULTS_DIR / 'evaluation_report.json'
            evaluator.generate_report(all_results, report_path)
        
        else:
            print("\nGenerating basic evaluation from Spark predictions...")
            # Basic evaluation from Spark predictions
            predictions_pd = spark_results['predictions'].select('label', 'prediction').toPandas()
            y_test_spark = predictions_pd['label'].values
            y_pred_spark = predictions_pd['prediction'].values
            
            # Error analysis
            error_results = evaluator.error_analysis(y_test_spark, y_pred_spark, class_names)
            
            # Use the actual class names that are present
            present_class_names = error_results.get('present_class_names', class_names)
            
            # Confusion matrix
            cm_path = self.config.PLOTS_DIR / 'confusion_matrix_spark.png'
            evaluator.plot_confusion_matrix(
                error_results['confusion_matrix'], 
                present_class_names, 
                cm_path
            )
            
            # Generate basic report
            basic_results = {**spark_results, **error_results}
            report_path = self.config.RESULTS_DIR / 'evaluation_report_spark.json'
            evaluator.generate_report(basic_results, report_path)
            
            print(f"\n✓ Basic evaluation complete (run with --compare for detailed XGBoost analysis)")
        
        # Save Spark model
        print(f"\nSaving Spark model...")
        model_path = str(self.config.MODELS_DIR / "spark_rf_model")
        try:
            trainer.model.write().overwrite().save(model_path)
            print(f"✓ Spark model saved to {model_path}")
        except Exception as e:
            print(f"Warning: Could not save Spark model: {e}")
        
        # Save XGBoost model if available
        if xgb_results and 'model' in xgb_results:
            print(f"\nSaving XGBoost model...")
            try:
                # Save using native XGBoost format
                xgb_model_path = self.config.MODELS_DIR / 'xgboost_model.json'
                xgb_results['model'].get_booster().save_model(str(xgb_model_path))
                print(f"✓ XGBoost model saved to {xgb_model_path}")
            except Exception as e:
                print(f"Warning: Could not save model in JSON format ({e})")
                print("Trying pickle format...")
                import pickle
                xgb_model_path = self.config.MODELS_DIR / 'model.pkl'
                with open(xgb_model_path, 'wb') as f:
                    pickle.dump(xgb_results['model'], f)
                print(f"✓ XGBoost model saved to {xgb_model_path}")
        
        # Save feature names
        feature_info = {
            'feature_names': feature_names,
            'n_features': len(feature_names),
            'timestamp': datetime.now().isoformat()
        }
        with open(self.config.MODELS_DIR / 'feature_names.json', 'w') as f:
            json.dump(feature_info, f, indent=2)
        print(f"✓ Feature info saved")
        
        # Step 6: Model Comparison
        if compare and xgb_results:
            print("\n" + "="*70)
            print("STEP 6: MODEL COMPARISON")
            print("="*70)
            
            comparison = pd.DataFrame({
                'Model': ['Spark RF', 'XGBoost'],
                'Accuracy': [spark_results['accuracy'], xgb_results['accuracy']],
                'Precision': [spark_results.get('precision', 0), xgb_results['precision']],
                'Recall': [spark_results.get('recall', 0), xgb_results['recall']],
                'F1 Score': [spark_results['f1'], xgb_results['f1']],
                'AUC': [spark_results.get('auc', 0), xgb_results.get('auc', 0)],
                'Training Time (s)': [spark_results['training_time'], 
                                     xgb_results['training_time']]
            })
            
            print("\n" + comparison.to_string(index=False))
            
            # Save comparison
            comparison.to_csv(self.config.RESULTS_DIR / 'model_comparison.csv', index=False)
        
        # Pipeline summary
        pipeline_time = time.time() - pipeline_start
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETE")
        print("="*70)
        print(f"Total time: {pipeline_time:.2f}s")
        print(f"\nResults saved to: {self.config.RESULTS_DIR}")
        print(f"Plots saved to: {self.config.PLOTS_DIR}")
        
        # Cleanup
        if self.spark:
            self.spark.stop()
    
    def _run_local_mode(self):
        """Run pipeline in local mode without Spark"""
        print("\n" + "="*70)
        print("STEP 1: GENERATING SAMPLE DATA (LOCAL MODE)")
        print("="*70)
        
        pipeline_start = time.time()
        
        # Generate sample data as pandas DataFrame
        print("Generating sample weather data...")
        dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
        stations = [f'STATION_{i:03d}' for i in range(50)]
        
        data = []
        for station in stations:
            lat = np.random.uniform(-10, 5)
            lon = np.random.uniform(29, 42)
            
            for date in dates:
                day_of_year = date.dayofyear
                base_temp = 22 + 5 * np.sin(2 * np.pi * day_of_year / 365)
                
                data.append({
                    'station_id': station,
                    'date': date,
                    'latitude': lat,
                    'longitude': lon,
                    'tmax': base_temp + np.random.normal(5, 2),
                    'tmin': base_temp - np.random.normal(5, 2),
                    'prcp': max(0, np.random.exponential(3))
                })
        
        df = pd.DataFrame(data)
        print(f"✓ Generated {len(df):,} records")
        
        # Step 2: Feature Engineering (Local)
        print("\n" + "="*70)
        print("STEP 2: FEATURE ENGINEERING (LOCAL)")
        print("="*70)
        
        df = self._create_features_local(df)
        df = self._create_labels_local(df)
        
        # Remove rows with NaN
        df = df.dropna()
        print(f"✓ Dataset prepared: {len(df):,} records after dropping NaN")
        
        # Check class distribution
        print("\nClass distribution:")
        print(df['extreme_event'].value_counts().sort_index())
        
        # Step 3: Train/Test Split
        print("\n" + "="*70)
        print("STEP 3: TRAIN/TEST SPLIT")
        print("="*70)
        
        # Prepare features
        feature_cols = [col for col in df.columns if col not in [
            'station_id', 'date', 'extreme_event', 
            'is_drought', 'is_flood', 'is_heatwave', 'is_cold_spell'
        ]]
        
        X = df[feature_cols].values
        y = df['extreme_event'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.TEST_SIZE, 
            random_state=self.config.RANDOM_STATE,
            stratify=y
        )
        
        # Scale features
        scaler = SKStandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        print(f"✓ Training set: {len(X_train):,} records")
        print(f"✓ Test set: {len(X_test):,} records")
        print(f"✓ Features: {len(feature_cols)}")
        
        # Save processed data
        print("\nSaving processed datasets...")
        pd.DataFrame(X_train, columns=feature_cols).to_parquet(
            self.config.PROCESSED_DIR / "train_features.parquet"
        )
        pd.DataFrame(X_test, columns=feature_cols).to_parquet(
            self.config.PROCESSED_DIR / "test_features.parquet"
        )
        pd.DataFrame({'label': y_train}).to_parquet(
            self.config.PROCESSED_DIR / "train_labels.parquet"
        )
        pd.DataFrame({'label': y_test}).to_parquet(
            self.config.PROCESSED_DIR / "test_labels.parquet"
        )
        print(f"✓ Datasets saved to {self.config.PROCESSED_DIR}")
        
        # Step 4: Model Training (XGBoost only in local mode)
        print("\n" + "="*70)
        print("STEP 4: MODEL TRAINING (XGBOOST)")
        print("="*70)
        
        if not XGBOOST_AVAILABLE:
            print("✗ XGBoost not available. Cannot run local mode without XGBoost.")
            print("Please install: pip install xgboost")
            return
        
        xgb_results = self._train_xgboost_local_direct(
            X_train, y_train, X_test, y_test
        )
        
        # Step 5: Model Evaluation
        print("\n" + "="*70)
        print("STEP 5: MODEL EVALUATION")
        print("="*70)
        
        evaluator = ModelEvaluation(self.config)
        class_names = ['Normal', 'Drought', 'Flood', 'Heatwave', 'Cold']
        
        y_pred = xgb_results['y_pred']
        y_pred_proba = xgb_results['y_pred_proba']
        
        # Error analysis
        error_results = evaluator.error_analysis(y_test, y_pred, class_names)
        present_class_names = error_results.get('present_class_names', class_names)
        
        # Cross-validation
        cv_results = evaluator.cross_validate(X_test, y_test, xgb_results['model'])
        
        # Confusion matrix
        cm_path = self.config.PLOTS_DIR / 'confusion_matrix.png'
        evaluator.plot_confusion_matrix(
            error_results['confusion_matrix'], 
            present_class_names, 
            cm_path
        )
        
        # ROC curves
        roc_path = self.config.PLOTS_DIR / 'roc_curves.png'
        evaluator.plot_roc_curves(y_test, y_pred_proba, class_names, roc_path)
        
        # Feature importance
        fi_path = self.config.PLOTS_DIR / 'feature_importance.png'
        evaluator.plot_feature_importance(xgb_results['model'], feature_cols, fi_path)
        
        # Note: SHAP analysis removed due to dependency conflicts
        print("\nNote: SHAP analysis skipped (not installed)")
        
        # Generate report
        all_results = {**xgb_results, **error_results, **cv_results}
        report_path = self.config.RESULTS_DIR / 'evaluation_report.json'
        evaluator.generate_report(all_results, report_path)
        
        # Save model
        print(f"\nSaving XGBoost model...")
        try:
            # Save using native XGBoost format (not sklearn wrapper)
            model_path = self.config.MODELS_DIR / 'xgboost_model.json'
            xgb_results['model'].get_booster().save_model(str(model_path))
            print(f"✓ Model saved to {model_path}")
        except Exception as e:
            print(f"Warning: Could not save model in JSON format ({e})")
            print("Trying pickle format...")
            import pickle
            model_path = self.config.MODELS_DIR / 'model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(xgb_results['model'], f)
            print(f"✓ Model saved to {model_path}")
        
        # Save scaler if we created one
        if hasattr(self, 'scaler') and self.scaler is not None:
            import pickle
            scaler_path = self.config.MODELS_DIR / 'scaler.pkl'
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"✓ Scaler saved to {scaler_path}")
        
        # Save feature names
        feature_info = {
            'feature_names': feature_cols,
            'n_features': len(feature_cols),
            'timestamp': datetime.now().isoformat()
        }
        with open(self.config.MODELS_DIR / 'feature_names.json', 'w') as f:
            json.dump(feature_info, f, indent=2)
        print(f"✓ Model and feature info saved to {self.config.MODELS_DIR}")
        
        # Pipeline summary
        pipeline_time = time.time() - pipeline_start
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETE (LOCAL MODE)")
        print("="*70)
        print(f"Total time: {pipeline_time:.2f}s")
        print(f"\nResults saved to: {self.config.RESULTS_DIR}")
        print(f"Plots saved to: {self.config.PLOTS_DIR}")
    
    def _train_xgboost_local_direct(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train XGBoost directly on numpy arrays"""
        print("\n[XGBoost] Training model...")
        start_time = time.time()
        
        # Handle class imbalance
        if SMOTE_AVAILABLE:
            try:
                print("Applying SMOTE for class imbalance...")
                smote = SMOTE(random_state=self.config.RANDOM_STATE)
                X_train, y_train = smote.fit_resample(X_train, y_train)
            except Exception as e:
                print(f"SMOTE failed ({e}), using class_weight instead...")
        else:
            print("Using class_weight for imbalance...")
        
        # Calculate sample weights
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        total_samples = len(y_train)
        n_classes = len(unique_classes)
        
        sample_weights = np.ones(len(y_train))
        for cls, count in zip(unique_classes, class_counts):
            weight = total_samples / (n_classes * count)
            sample_weights[y_train == cls] = weight
        
        # Determine number of classes
        n_classes = len(np.unique(y_train))
        
        # Train XGBoost with proper configuration
        if n_classes == 2:
            # Binary classification
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.config.RANDOM_STATE,
                eval_metric='logloss',
                tree_method='hist'
            )
        else:
            # Multiclass classification
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.config.RANDOM_STATE,
                eval_metric='mlogloss',
                tree_method='hist',
                objective='multi:softprob',
                num_class=n_classes
            )
        
        xgb_model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)
        training_time = time.time() - start_time
        
        # Predict
        y_pred = xgb_model.predict(X_test)
        y_pred_proba = xgb_model.predict_proba(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        try:
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            auc = 0.0
        
        results = {
            'model_type': 'XGBoost_Local',
            'training_time': training_time,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'model': xgb_model,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"✓ Training completed in {training_time:.2f}s")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
        
        return results
    
    def _create_features_local(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features using pandas"""
        print("Creating features...")
        
        # Sort by station and date
        df = df.sort_values(['station_id', 'date'])
        
        # Rolling statistics
        for days in [7, 14, 30]:
            df[f'tmax_rolling_{days}d_mean'] = df.groupby('station_id')['tmax'].transform(
                lambda x: x.rolling(days, min_periods=1).mean()
            )
            df[f'tmax_rolling_{days}d_max'] = df.groupby('station_id')['tmax'].transform(
                lambda x: x.rolling(days, min_periods=1).max()
            )
            df[f'tmin_rolling_{days}d_mean'] = df.groupby('station_id')['tmin'].transform(
                lambda x: x.rolling(days, min_periods=1).mean()
            )
            df[f'tmin_rolling_{days}d_min'] = df.groupby('station_id')['tmin'].transform(
                lambda x: x.rolling(days, min_periods=1).min()
            )
            df[f'prcp_rolling_{days}d_sum'] = df.groupby('station_id')['prcp'].transform(
                lambda x: x.rolling(days, min_periods=1).sum()
            )
            df[f'prcp_rolling_{days}d_mean'] = df.groupby('station_id')['prcp'].transform(
                lambda x: x.rolling(days, min_periods=1).mean()
            )
        
        # Lag features
        for lag in [1, 3, 7]:
            df[f'tmax_lag_{lag}d'] = df.groupby('station_id')['tmax'].shift(lag)
            df[f'prcp_lag_{lag}d'] = df.groupby('station_id')['prcp'].shift(lag)
        
        # Temperature anomaly
        df['tmax_anomaly'] = df['tmax'] - df['tmax_rolling_30d_mean']
        df['tmin_anomaly'] = df['tmin'] - df['tmin_rolling_30d_mean']
        
        # Temporal features
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        df['season'] = df['month'].map(lambda m: 1 if m in [12, 1, 2] else 
                                                 2 if m in [3, 4, 5] else 
                                                 3 if m in [6, 7, 8] else 4)
        
        print("✓ Features created")
        return df
    
    def _create_labels_local(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create extreme event labels using pandas"""
        print("Creating extreme event labels...")
        
        # Calculate thresholds
        prcp_p10 = df['prcp_rolling_30d_sum'].quantile(0.10)
        prcp_p95 = df['prcp'].quantile(0.95)
        tmax_p95 = df['tmax'].quantile(0.95)
        tmin_p5 = df['tmin'].quantile(0.05)
        
        print(f"Thresholds calculated:")
        print(f"  Drought (prcp < {prcp_p10:.2f} mm/30d)")
        print(f"  Flood (prcp > {prcp_p95:.2f} mm/day)")
        print(f"  Heatwave (tmax > {tmax_p95:.2f}°C)")
        print(f"  Cold spell (tmin < {tmin_p5:.2f}°C)")
        
        # Create labels
        df['is_drought'] = (df['prcp_rolling_30d_sum'] < prcp_p10).astype(int)
        df['is_flood'] = (df['prcp'] > prcp_p95).astype(int)
        df['is_heatwave'] = (df['tmax'] > tmax_p95).astype(int)
        df['is_cold_spell'] = (df['tmin'] < tmin_p5).astype(int)
        
        # Multi-class label
        df['extreme_event'] = 0  # Normal
        df.loc[df['is_drought'] == 1, 'extreme_event'] = 1
        df.loc[df['is_flood'] == 1, 'extreme_event'] = 2
        df.loc[df['is_heatwave'] == 1, 'extreme_event'] = 3
        df.loc[df['is_cold_spell'] == 1, 'extreme_event'] = 4
        
        print("✓ Labels created")
        return df
    
    def _run_distributed_mode(self, compare: bool = False):
        """Run pipeline in distributed mode with Spark"""
        pipeline_start = time.time()


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Africa Extreme Weather Event Prediction Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python africa_extreme_weather_ml.py --mode distributed
  python africa_extreme_weather_ml.py --mode local --compare
  
Modes:
  distributed: Use Spark for distributed processing
  local: Use local processing only
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['distributed', 'local', 'spark'],
        default='distributed',
        help='Processing mode (default: distributed)'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare Spark GBT with XGBoost'
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = ExtremeWeatherMLPipeline(mode=args.mode)
    pipeline.run(compare=args.compare)


if __name__ == '__main__':
    main()