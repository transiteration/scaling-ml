"""
Scaling ML Project Configuration Script

This script configures the logging, directory structure, and MLflow settings for a scalable machine learning project.

Components:
    1. Logging Configuration: Sets up console and file-based logging with detailed and minimal formats.
    2. Directory Structure: Creates necessary directories for logs, EFS (Elastic File System), and the MLflow model registry.
    3. MLflow Settings: Configures MLflow tracking URI for storing experiment runs.

Usage:
    Modify the script to adjust logging configurations or customize the directory structure based on your project requirements.

Configuration:
    - ROOT_DIR: Root directory of the project.
    - LOGS_DIR: Directory for storing log files.
    - EFS_DIR: Directory for Elastic File System (EFS) storage.
    - MODEL_REGISTRY: Directory for storing MLflow models.
    - MLFLOW_TRACKING_URI: MLflow tracking URI for storing experiment runs.


Source code in scripts/config.py
"""

import logging
import sys
from pathlib import Path

import mlflow

ROOT_DIR = Path(__file__).parent.parent.absolute()
LOGS_DIR = Path(ROOT_DIR, "logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
EFS_DIR = Path(ROOT_DIR, "efs")
EFS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_REGISTRY = Path(EFS_DIR, "mlflow")
MODEL_REGISTRY.mkdir(parents=True, exist_ok=True)
MLFLOW_TRACKING_URI = "file://" + str(MODEL_REGISTRY.absolute())
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "minimal": {"format": "%(message)s"},
        "detailed": {"format": "%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"},
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "stream": sys.stdout, "formatter": "minimal", "level": logging.DEBUG},
        "info": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "info.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.INFO,
        },
        "error": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "error.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.ERROR,
        },
    },
    "root": {"handlers": ["console", "info", "error"], "level": logging.INFO, "propagate": True},
}

logging.config.dictConfig(logging_config)
logger = logging.getLogger()
