"""
Configuration file for Smart Feedback Summarizer
Contains all constants, paths, and model configurations
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================
# MongoDB
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = "feedback_analyzer"
MONGO_COLLECTION = "feedback_data"

# SQLite (alternative)
SQLITE_DB_PATH = DATA_DIR / "feedback.db"

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

# Sentiment Analysis Model
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
SENTIMENT_LABELS = {
    "POSITIVE": "positive",
    "NEGATIVE": "negative",
    "NEUTRAL": "neutral"
}

# Text Summarization Model
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"  # or "google/pegasus-xsum"
MAX_SUMMARY_LENGTH = 150
MIN_SUMMARY_LENGTH = 50

# Topic Modeling Configuration
TOPIC_MODEL_NAME = "all-MiniLM-L6-v2"  # Sentence transformer for embeddings
MIN_TOPIC_SIZE = 10
N_TOPICS = 5  # Number of top topics to extract

# ============================================================================
# SPARK CONFIGURATION
# ============================================================================
SPARK_APP_NAME = "SmartFeedbackSummarizer"
SPARK_MASTER = "local[*]"  # Use all available cores
SPARK_MEMORY = "4g"
SPARK_DRIVER_MEMORY = "2g"

# ============================================================================
# DATA PROCESSING PARAMETERS
# ============================================================================
# Text cleaning
MIN_TEXT_LENGTH = 10  # Minimum characters in feedback
MAX_TEXT_LENGTH = 1000  # Maximum characters to process
REMOVE_STOPWORDS = True
LEMMATIZE = True

# Language
DEFAULT_LANGUAGE = "en"

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================
# Dashboard configuration
PAGE_TITLE = "Smart Feedback Summarizer"
PAGE_ICON = "📊"
LAYOUT = "wide"

# Color schemes
COLOR_POSITIVE = "#28a745"
COLOR_NEGATIVE = "#dc3545"
COLOR_NEUTRAL = "#6c757d"

SENTIMENT_COLORS = {
    "positive": COLOR_POSITIVE,
    "negative": COLOR_NEGATIVE,
    "neutral": COLOR_NEUTRAL
}

# Chart settings
CHART_HEIGHT = 400
CHART_WIDTH = 600

# ============================================================================
# REPORT GENERATION
# ============================================================================
REPORT_TITLE = "Customer Feedback Analysis Report"
REPORT_AUTHOR = "Smart Feedback Summarizer System"
COMPANY_NAME = "Your Company Name"

# ============================================================================
# API SETTINGS (for future integrations)
# ============================================================================
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY", "")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET", "")

# ============================================================================
# PERFORMANCE TUNING
# ============================================================================
BATCH_SIZE = 32  # For model inference
NUM_WORKERS = 4  # For data loading
CACHE_SIZE = 1000  # Number of items to cache

# ============================================================================
# LOGGING
# ============================================================================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = BASE_DIR / "app.log"

# ============================================================================
# FEATURE FLAGS
# ============================================================================
ENABLE_TOPIC_EXTRACTION = True
ENABLE_SUMMARIZATION = True
ENABLE_SENTIMENT_ANALYSIS = True
USE_GPU = False  # Set to True if GPU available

# ============================================================================
# SAMPLE DATA GENERATION
# ============================================================================
SAMPLE_DATA_SIZE = 500  # Number of synthetic records to generate
