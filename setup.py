from setuptools import setup, find_packages

setup(
    name="africa-weather-ml",
    version="1.0.0",
    description="Extreme Weather Event Prediction for Africa",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "fastapi>=0.103.1",
        "uvicorn>=0.23.2",
        "pydantic>=2.3.0",
        "mlflow>=2.7.1",
        "prometheus-client>=0.17.1",
        "pyyaml>=6.0.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.1",
        ],
        "spark": ["pyspark>=3.4.1"],
    },
    entry_points={
        "console_scripts": [
            "weather-ml-train=training.train:main",
            "weather-ml-serve=api.main:run",
            "weather-ml-predict=inference.predictor:main",
        ],
    },
)