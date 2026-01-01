"""
Utility functions for the Smart Feedback Summarizer
Contains helper functions used across multiple modules
"""

import re
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
import config

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger(name: str, log_file: str = None, level=logging.INFO):
    """
    Set up logger with file and console handlers
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Format
    formatter = logging.Formatter(config.LOG_FORMAT)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# ============================================================================
# TEXT PREPROCESSING UTILITIES
# ============================================================================

def clean_text(text: str) -> str:
    """
    Basic text cleaning operations
    
    Args:
        text: Raw text string
    
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove mentions and hashtags (for social media)
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove special characters and digits (keep basic punctuation)
    text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()


def validate_text(text: str, min_length: int = config.MIN_TEXT_LENGTH) -> bool:
    """
    Validate if text meets minimum requirements
    
    Args:
        text: Text to validate
        min_length: Minimum acceptable length
    
    Returns:
        True if valid, False otherwise
    """
    if not text or not isinstance(text, str):
        return False
    
    cleaned = clean_text(text)
    return len(cleaned) >= min_length


def truncate_text(text: str, max_length: int = config.MAX_TEXT_LENGTH) -> str:
    """
    Truncate text to maximum length
    
    Args:
        text: Text to truncate
        max_length: Maximum length
    
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

# ============================================================================
# DATA PROCESSING UTILITIES
# ============================================================================

def remove_duplicates(df: pd.DataFrame, column: str = 'text') -> pd.DataFrame:
    """
    Remove duplicate feedback entries
    
    Args:
        df: Input DataFrame
        column: Column name to check for duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    initial_count = len(df)
    df = df.drop_duplicates(subset=[column], keep='first')
    removed = initial_count - len(df)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Removed {removed} duplicate entries")
    
    return df


def filter_by_date(df: pd.DataFrame, 
                   start_date: str = None, 
                   end_date: str = None,
                   date_column: str = 'date') -> pd.DataFrame:
    """
    Filter DataFrame by date range
    
    Args:
        df: Input DataFrame
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        date_column: Name of date column
    
    Returns:
        Filtered DataFrame
    """
    if date_column not in df.columns:
        return df
    
    df[date_column] = pd.to_datetime(df[date_column])
    
    if start_date:
        df = df[df[date_column] >= pd.to_datetime(start_date)]
    
    if end_date:
        df = df[df[date_column] <= pd.to_datetime(end_date)]
    
    return df

# ============================================================================
# STATISTICAL UTILITIES
# ============================================================================

def calculate_sentiment_distribution(sentiments: List[str]) -> Dict[str, float]:
    """
    Calculate percentage distribution of sentiments
    
    Args:
        sentiments: List of sentiment labels
    
    Returns:
        Dictionary with sentiment percentages
    """
    if not sentiments:
        return {"positive": 0, "negative": 0, "neutral": 0}
    
    total = len(sentiments)
    distribution = {
        "positive": (sentiments.count("positive") / total) * 100,
        "negative": (sentiments.count("negative") / total) * 100,
        "neutral": (sentiments.count("neutral") / total) * 100
    }
    
    return distribution


def get_top_keywords(texts: List[str], n: int = 10) -> List[tuple]:
    """
    Extract top N keywords from texts
    
    Args:
        texts: List of text strings
        n: Number of top keywords to return
    
    Returns:
        List of (keyword, frequency) tuples
    """
    from collections import Counter
    import re
    
    # Combine all texts
    combined = ' '.join(texts)
    
    # Extract words (simple approach)
    words = re.findall(r'\b[a-z]{3,}\b', combined.lower())
    
    # Remove common stopwords (basic list)
    stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'with', 
                 'was', 'this', 'that', 'from', 'have', 'has', 'had'}
    words = [w for w in words if w not in stopwords]
    
    # Get top N
    counter = Counter(words)
    return counter.most_common(n)

# ============================================================================
# DATA VALIDATION
# ============================================================================

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate if DataFrame has required columns
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
    
    Returns:
        True if valid, False otherwise
    """
    missing_columns = set(required_columns) - set(df.columns)
    
    if missing_columns:
        logger = logging.getLogger(__name__)
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    
    return True

# ============================================================================
# FILE OPERATIONS
# ============================================================================

def safe_read_csv(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Safely read CSV file with error handling
    
    Args:
        file_path: Path to CSV file
        **kwargs: Additional arguments for pd.read_csv
    
    Returns:
        DataFrame or None if error
    """
    logger = logging.getLogger(__name__)
    
    try:
        df = pd.read_csv(file_path, **kwargs)
        logger.info(f"Successfully loaded {len(df)} records from {file_path}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error reading CSV: {str(e)}")
        return None


def safe_save_csv(df: pd.DataFrame, file_path: str, **kwargs) -> bool:
    """
    Safely save DataFrame to CSV with error handling
    
    Args:
        df: DataFrame to save
        file_path: Destination file path
        **kwargs: Additional arguments for df.to_csv
    
    Returns:
        True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        df.to_csv(file_path, index=False, **kwargs)
        logger.info(f"Successfully saved {len(df)} records to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving CSV: {str(e)}")
        return False

# ============================================================================
# TIME UTILITIES
# ============================================================================

def get_timestamp() -> str:
    """Get current timestamp as string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_date(date_string: str) -> datetime:
    """
    Parse date string to datetime object
    
    Args:
        date_string: Date string in various formats
    
    Returns:
        datetime object or None if parsing fails
    """
    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S"
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            continue
    
    return None

# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

class Timer:
    """Context manager for timing code execution"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.logger = logging.getLogger(__name__)
    
    def __enter__(self):
        self.start = datetime.now()
        return self
    
    def __exit__(self, *args):
        self.end = datetime.now()
        self.duration = (self.end - self.start).total_seconds()
        self.logger.info(f"{self.name} completed in {self.duration:.2f} seconds")

# ============================================================================
# FORMATTING UTILITIES
# ============================================================================

def format_percentage(value: float, decimals: int = 1) -> str:
    """Format float as percentage string"""
    return f"{value:.{decimals}f}%"


def format_number(value: int) -> str:
    """Format large numbers with commas"""
    return f"{value:,}"


def truncate_string(text: str, length: int = 100, suffix: str = "...") -> str:
    """Truncate string to specified length"""
    if len(text) <= length:
        return text
    return text[:length - len(suffix)] + suffix
