"""
NOAA GHCN-Daily Weather Data Ingestion Pipeline

A PySpark pipeline for ingesting and processing NOAA's 
Global Historical Climatology Network Daily dataset.
"""

import os
import requests
import gzip
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, 
    FloatType, DateType, TimestampType
)


class GHCNConfig:
    """Configuration for GHCN-Daily pipeline"""
    
    # Data URLs
    BASE_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily"
    DATA_URL = f"{BASE_URL}/by_year"
    STATIONS_URL = f"{BASE_URL}/ghcnd-stations.txt"
    
    # Local paths
    BASE_DIR = Path("./ghcn_data")
    RAW_DIR = BASE_DIR / "raw"
    PROCESSED_DIR = BASE_DIR / "processed"
    STAGING_DIR = BASE_DIR / "staging"
    METADATA_DIR = BASE_DIR / "metadata"
    CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
    
    # Processing parameters
    SAMPLE_YEARS = [2025]
    PARTITION_COLS = ["year", "month"]
    
    # Data quality thresholds
    MAX_TEMP_CELSIUS = 60
    MIN_TEMP_CELSIUS = -90
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories"""
        for dir_path in [cls.RAW_DIR, cls.PROCESSED_DIR, cls.STAGING_DIR, 
                         cls.METADATA_DIR, cls.CHECKPOINTS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)


class DataExtractor:
    """Handles data extraction from NOAA sources"""
    
    def __init__(self, config: GHCNConfig):
        self.config = config
    
    def download_file(self, url: str, destination: Path, 
                     decompress: bool = True) -> bool:
        """Download file with progress tracking"""
        try:
            print(f"Downloading: {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Save file
            temp_file = destination.with_suffix('.tmp')
            with open(temp_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Decompress if needed
            if decompress and str(url).endswith('.gz'):
                with gzip.open(temp_file, 'rb') as f_in:
                    with open(destination, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                temp_file.unlink()
            else:
                temp_file.rename(destination)
            
            print(f" Downloaded: {destination.name}")
            return True
            
        except Exception as e:
            print(f" Error downloading {url}: {e}")
            return False
    
    def download_stations_metadata(self) -> Optional[Path]:
        """Download stations metadata file"""
        dest = self.config.METADATA_DIR / "ghcnd-stations.txt"
        
        if dest.exists():
            print(f"Stations metadata already exists: {dest}")
            return dest
        
        if self.download_file(self.config.STATIONS_URL, dest, decompress=False):
            return dest
        return None
    
    def download_yearly_data(self, year: int) -> Optional[Path]:
        """Download weather data for a specific year"""
        filename = f"{year}.csv.gz"
        url = f"{self.config.DATA_URL}/{filename}"
        dest = self.config.RAW_DIR / f"{year}.csv"
        
        if dest.exists():
            print(f"Data for {year} already exists: {dest}")
            return dest
        
        if self.download_file(url, dest, decompress=True):
            return dest
        return None
    
    def extract_all(self, years: List[int]) -> Dict[str, Path]:
        """Extract all required data"""
        results = {"stations": None, "data": []}
        
        # Download stations metadata
        results["stations"] = self.download_stations_metadata()
        
        # Download yearly data
        for year in years:
            data_path = self.download_yearly_data(year)
            if data_path:
                results["data"].append(data_path)
        
        return results


class DataTransformer:
    """Handles data transformation and quality checks"""
    
    def __init__(self, spark: SparkSession, config: GHCNConfig):
        self.spark = spark
        self.config = config
    
    def get_weather_schema(self) -> StructType:
        """Define schema for GHCN-Daily weather data"""
        return StructType([
            StructField("station_id", StringType(), False),
            StructField("date", StringType(), False),
            StructField("element", StringType(), False),
            StructField("value", IntegerType(), False),
            StructField("mflag", StringType(), True),
            StructField("qflag", StringType(), True),
            StructField("sflag", StringType(), True),
            StructField("obs_time", StringType(), True)
        ])
    
    def get_stations_schema(self) -> StructType:
        """Define schema for stations metadata"""
        return StructType([
            StructField("station_id", StringType(), False),
            StructField("latitude", FloatType(), False),
            StructField("longitude", FloatType(), False),
            StructField("elevation", FloatType(), True),
            StructField("state", StringType(), True),
            StructField("name", StringType(), True),
            StructField("gsn_flag", StringType(), True),
            StructField("hcn_flag", StringType(), True),
            StructField("wmo_id", StringType(), True)
        ])
    
    def load_stations_metadata(self, file_path: Path) -> DataFrame:
        """Load and parse stations metadata (fixed-width format)"""
        print(f"Loading stations metadata from {file_path}")
        
        # Read as text first
        text_df = self.spark.read.text(str(file_path))
        
        # Parse fixed-width format
        stations_df = text_df.select(
            F.substring(F.col("value"), 1, 11).alias("station_id"),
            F.substring(F.col("value"), 13, 8).cast("float").alias("latitude"),
            F.substring(F.col("value"), 22, 9).cast("float").alias("longitude"),
            F.substring(F.col("value"), 32, 6).cast("float").alias("elevation"),
            F.substring(F.col("value"), 39, 2).alias("state"),
            F.substring(F.col("value"), 42, 30).alias("name"),
            F.substring(F.col("value"), 73, 3).alias("gsn_flag"),
            F.substring(F.col("value"), 77, 3).alias("hcn_flag"),
            F.substring(F.col("value"), 81, 5).alias("wmo_id")
        )
        
        # Clean up whitespace
        for col in stations_df.columns:
            stations_df = stations_df.withColumn(col, F.trim(F.col(col)))
        
        print(f" Loaded {stations_df.count()} stations")
        return stations_df
    
    def load_weather_data(self, file_paths: List[Path]) -> DataFrame:
        """Load weather observation data"""
        print(f"Loading weather data from {len(file_paths)} files")
        
        # Read CSV files
        df = self.spark.read.csv(
            [str(p) for p in file_paths],
            schema=self.get_weather_schema(),
            header=False
        )
        
        print(f" Loaded {df.count()} weather observations")
        return df
    
    def transform_weather_data(self, df: DataFrame) -> DataFrame:
        """Apply transformations to weather data"""
        print("Transforming weather data...")
        
        # Parse date and add partitioning columns
        df = df.withColumn("date", F.to_date(F.col("date"), "yyyyMMdd"))
        df = df.withColumn("year", F.year(F.col("date")))
        df = df.withColumn("month", F.month(F.col("date")))
        
        # Convert values to readable units
        # TMAX, TMIN are in tenths of degrees Celsius
        # PRCP is in tenths of mm
        df = df.withColumn(
            "value_converted",
            F.when(F.col("element").isin(["TMAX", "TMIN", "TAVG"]), 
                   F.col("value") / 10.0)
            .when(F.col("element").isin(["PRCP", "SNOW", "SNWD"]), 
                  F.col("value") / 10.0)
            .otherwise(F.col("value"))
        )
        
        # Add data quality flags
        df = df.withColumn(
            "quality_passed",
            (F.col("qflag").isNull() | (F.col("qflag") == ""))
        )
        
        # Add ingestion metadata
        df = df.withColumn("ingestion_timestamp", F.current_timestamp())
        df = df.withColumn("ingestion_date", F.current_date())
        
        print(" Transformation complete")
        return df
    
    def validate_data_quality(self, df: DataFrame) -> Dict[str, int]:
        """Perform data quality checks and return metrics"""
        print("Running data quality checks...")
        
        metrics = {}
        
        # Count total records
        metrics["total_records"] = df.count()
        
        # Count null values
        metrics["null_station_ids"] = df.filter(F.col("station_id").isNull()).count()
        metrics["null_dates"] = df.filter(F.col("date").isNull()).count()
        
        # Temperature range validation
        temp_df = df.filter(F.col("element").isin(["TMAX", "TMIN", "TAVG"]))
        metrics["temp_out_of_range"] = temp_df.filter(
            (F.col("value_converted") > self.config.MAX_TEMP_CELSIUS) |
            (F.col("value_converted") < self.config.MIN_TEMP_CELSIUS)
        ).count()
        
        # Quality flag analysis
        metrics["failed_quality_checks"] = df.filter(~F.col("quality_passed")).count()
        
        # Duplicate check
        metrics["duplicate_records"] = df.count() - df.dropDuplicates(
            ["station_id", "date", "element"]
        ).count()
        
        print("Data Quality Metrics:")
        for key, value in metrics.items():
            print(f" - {key}: {value:,}")
        
        return metrics
    
    def enrich_with_stations(self, weather_df: DataFrame, 
                            stations_df: DataFrame) -> DataFrame:
        """Enrich weather data with station metadata"""
        print("Enriching weather data with station metadata...")
        
        enriched_df = weather_df.join(
            stations_df,
            on="station_id",
            how="left"
        )
        
        print("Enrichment complete")
        return enriched_df


class DataLoader:
    """Handles writing processed data to storage"""
    
    def __init__(self, spark: SparkSession, config: GHCNConfig):
        self.spark = spark
        self.config = config
    
    def write_parquet(self, df: DataFrame, output_path: Path, 
                     partition_cols: List[str] = None) -> bool:
        """Write DataFrame to Parquet with partitioning"""
        try:
            print(f"Writing data to {output_path}...")
            
            writer = df.write.mode("overwrite").option("compression", "snappy")
            
            if partition_cols:
                writer = writer.partitionBy(*partition_cols)
            
            writer.parquet(str(output_path))
            
            print(f"Data written successfully to {output_path}")
            return True
            
        except Exception as e:
            print(f" Error writing data: {e}")
            return False
    
    def write_summary_statistics(self, df: DataFrame, output_path: Path):
        """Write summary statistics"""
        print("Generating summary statistics...")
        
        # Calculate statistics by element type
        summary = df.groupBy("element", "year", "month").agg(
            F.count("*").alias("observation_count"),
            F.countDistinct("station_id").alias("unique_stations"),
            F.avg("value_converted").alias("avg_value"),
            F.min("value_converted").alias("min_value"),
            F.max("value_converted").alias("max_value"),
            F.sum(F.when(F.col("quality_passed"), 1).otherwise(0)).alias("quality_passed_count")
        )
        
        summary.write.mode("overwrite").parquet(str(output_path))
        print(f"Summary statistics written to {output_path}")


class GHCNPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config: GHCNConfig = None):
        self.config = config or GHCNConfig()
        self.config.create_directories()
        
        # Initialize Spark
        self.spark = self._create_spark_session()
        
        # Initialize components
        self.extractor = DataExtractor(self.config)
        self.transformer = DataTransformer(self.spark, self.config)
        self.loader = DataLoader(self.spark, self.config)
    
    def _create_spark_session(self) -> SparkSession:
        """Create and configure Spark session"""
        return (SparkSession.builder
                .appName("GHCN-Daily-Ingestion")
                .config("spark.sql.adaptive.enabled", "true")
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
                .config("spark.sql.files.maxPartitionBytes", "128MB")
                .config("spark.driver.memory", "4g")
                .getOrCreate())
    
    def run(self, years: List[int] = None):
        """Execute the complete pipeline"""
        years = years or self.config.SAMPLE_YEARS
        
        print("NOAA GHCN-Daily Weather Data Ingestion Pipeline")
        print("=" * 70)
        print(f"Processing years: {years}")
        print(f"Output directory: {self.config.PROCESSED_DIR}")
        print()
        
        # Step 1: Extract
        print("\n[1/5] EXTRACTION")
        print("-" * 70)
        extracted_data = self.extractor.extract_all(years)
        
        if not extracted_data["stations"] or not extracted_data["data"]:
            print("Extraction failed! Aborting pipeline.")
            return
        
        # Step 2: Load raw data
        print("\n[2/5] LOADING RAW DATA")
        print("-" * 70)
        stations_df = self.transformer.load_stations_metadata(
            extracted_data["stations"]
        )
        weather_df = self.transformer.load_weather_data(extracted_data["data"])
        
        # Step 3: Transform
        print("\n[3/5] TRANSFORMATION")
        print("-" * 70)
        weather_df = self.transformer.transform_weather_data(weather_df)
        
        # Step 4: Validate
        print("\n[4/5] DATA QUALITY VALIDATION")
        print("-" * 70)
        quality_metrics = self.transformer.validate_data_quality(weather_df)
        
        # Step 5: Enrich and Load
        print("\n[5/5] ENRICHMENT & LOADING")
        print("-" * 70)
        enriched_df = self.transformer.enrich_with_stations(weather_df, stations_df)
        
        # Write main dataset
        self.loader.write_parquet(
            enriched_df,
            self.config.PROCESSED_DIR / "weather_observations",
            partition_cols=self.config.PARTITION_COLS
        )
        
        # Write stations metadata
        self.loader.write_parquet(
            stations_df,
            self.config.PROCESSED_DIR / "stations_metadata"
        )
        
        # Write summary statistics
        self.loader.write_summary_statistics(
            enriched_df,
            self.config.PROCESSED_DIR / "summary_statistics"
        )
        
        # Final summary
        print("PIPELINE EXECUTION COMPLETE")
        print("-" * 70)
        print(f"Processed {quality_metrics['total_records']:,} weather observations")
        print(f"Output location: {self.config.PROCESSED_DIR}")
        print(f"Data quality: {quality_metrics['failed_quality_checks']:,} failed checks")
    
    def stop(self):
        """Stop Spark session"""
        self.spark.stop()


def main():
    """Main entry point"""
    pipeline = GHCNPipeline()
    
    try:
        # Run pipeline for sample years
        pipeline.run(years=[2021, 2022, 2023, 2024, 2025])
        
    except Exception as e:
        print(f"\n Pipeline failed with error: {e}")
        raise
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()