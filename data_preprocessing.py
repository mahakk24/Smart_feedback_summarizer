"""
Data Preprocessing Module using PySpark
Handles large-scale text cleaning, normalization, and preparation for analysis
"""

import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, lower, regexp_replace, trim, length, 
    udf, when, lit, to_timestamp
)
from pyspark.sql.types import StringType, BooleanType
import re
import config
import utils

# Setup logger
logger = utils.setup_logger(__name__, config.LOG_FILE)

# ============================================================================
# SPARK SESSION MANAGEMENT
# ============================================================================

class SparkDataPreprocessor:
    """Handle large-scale data preprocessing using PySpark"""
    
    def __init__(self):
        """Initialize Spark session"""
        self.logger = logger
        self.spark = self._create_spark_session()
    
    def _create_spark_session(self) -> SparkSession:
        """
        Create and configure Spark session
        
        Returns:
            SparkSession instance
        """
        self.logger.info("Initializing Spark session...")
        
        spark = SparkSession.builder \
            .appName(config.SPARK_APP_NAME) \
            .master(config.SPARK_MASTER) \
            .config("spark.executor.memory", config.SPARK_MEMORY) \
            .config("spark.driver.memory", config.SPARK_DRIVER_MEMORY) \
            .config("spark.sql.shuffle.partitions", "4") \
            .getOrCreate()
        
        # Set log level
        spark.sparkContext.setLogLevel("WARN")
        
        self.logger.info("Spark session created successfully")
        return spark
    
    def stop(self):
        """Stop Spark session"""
        if self.spark:
            self.spark.stop()
            self.logger.info("Spark session stopped")
    
    # ========================================================================
    # DATA LOADING
    # ========================================================================
    
    def load_data(self, df_pandas: pd.DataFrame) -> DataFrame:
        """
        Load Pandas DataFrame into Spark DataFrame
        
        Args:
            df_pandas: Input Pandas DataFrame
        
        Returns:
            Spark DataFrame
        """
        self.logger.info(f"Loading {len(df_pandas)} records into Spark...")
        
        spark_df = self.spark.createDataFrame(df_pandas)
        
        self.logger.info(f"Spark DataFrame created with {spark_df.count()} rows")
        return spark_df
    
    # ========================================================================
    # TEXT CLEANING UDFs
    # ========================================================================
    
    @staticmethod
    def clean_text_udf(text):
        """
        UDF for advanced text cleaning
        
        Args:
            text: Input text string
        
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z\s.,!?\']', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    @staticmethod
    def is_valid_text_udf(text, min_length=config.MIN_TEXT_LENGTH):
        """
        UDF to validate text length
        
        Args:
            text: Input text
            min_length: Minimum acceptable length
        
        Returns:
            Boolean indicating validity
        """
        if not text:
            return False
        return len(text.strip()) >= min_length
    
    # ========================================================================
    # PREPROCESSING PIPELINE
    # ========================================================================
    
    def preprocess(self, spark_df: DataFrame) -> DataFrame:
        """
        Complete preprocessing pipeline for feedback data
        
        Args:
            spark_df: Input Spark DataFrame
        
        Returns:
            Preprocessed Spark DataFrame
        """
        self.logger.info("Starting preprocessing pipeline...")
        
        # Register UDFs
        clean_text_spark_udf = udf(self.clean_text_udf, StringType())
        is_valid_spark_udf = udf(self.is_valid_text_udf, BooleanType())
        
        # Step 1: Handle nulls
        self.logger.info("Step 1: Handling null values...")
        spark_df = spark_df.filter(col("text").isNotNull())
        
        # Step 2: Basic cleaning using built-in functions
        self.logger.info("Step 2: Basic text cleaning...")
        spark_df = spark_df.withColumn("text", trim(col("text")))
        
        # Step 3: Advanced cleaning using UDF
        self.logger.info("Step 3: Advanced text cleaning...")
        spark_df = spark_df.withColumn("text_cleaned", clean_text_spark_udf(col("text")))
        
        # Step 4: Validate text length
        self.logger.info("Step 4: Validating text length...")
        spark_df = spark_df.withColumn("is_valid", is_valid_spark_udf(col("text_cleaned")))
        
        # Filter valid texts
        initial_count = spark_df.count()
        spark_df = spark_df.filter(col("is_valid") == True)
        final_count = spark_df.count()
        
        removed = initial_count - final_count
        self.logger.info(f"Removed {removed} invalid records")
        
        # Step 5: Remove duplicates
        self.logger.info("Step 5: Removing duplicates...")
        initial_count = spark_df.count()
        spark_df = spark_df.dropDuplicates(["text_cleaned"])
        final_count = spark_df.count()
        
        duplicates_removed = initial_count - final_count
        self.logger.info(f"Removed {duplicates_removed} duplicate records")
        
        # Step 6: Add metadata
        self.logger.info("Step 6: Adding metadata...")
        spark_df = spark_df.withColumn("text_length", length(col("text_cleaned")))
        spark_df = spark_df.withColumn("word_count", 
                                       length(regexp_replace(col("text_cleaned"), r'\s+', ' ')) 
                                       - length(regexp_replace(col("text_cleaned"), r'\s', '')) + 1)
        
        # Drop temporary columns
        spark_df = spark_df.drop("is_valid")
        
        self.logger.info(f"Preprocessing complete. Final record count: {spark_df.count()}")
        
        return spark_df
    
    # ========================================================================
    # DATA QUALITY CHECKS
    # ========================================================================
    
    def data_quality_report(self, spark_df: DataFrame) -> dict:
        """
        Generate data quality metrics
        
        Args:
            spark_df: Input Spark DataFrame
        
        Returns:
            Dictionary with quality metrics
        """
        self.logger.info("Generating data quality report...")
        
        report = {
            'total_records': spark_df.count(),
            'null_texts': spark_df.filter(col("text").isNull()).count(),
            'empty_texts': spark_df.filter(trim(col("text")) == "").count(),
            'avg_text_length': spark_df.agg({'text_length': 'avg'}).collect()[0][0] if 'text_length' in spark_df.columns else 0,
            'avg_word_count': spark_df.agg({'word_count': 'avg'}).collect()[0][0] if 'word_count' in spark_df.columns else 0,
        }
        
        # Source distribution
        if 'source' in spark_df.columns:
            report['source_distribution'] = {
                row['source']: row['count'] 
                for row in spark_df.groupBy('source').count().collect()
            }
        
        return report
    
    # ========================================================================
    # CONVERSION
    # ========================================================================
    
    def to_pandas(self, spark_df: DataFrame) -> pd.DataFrame:
        """
        Convert Spark DataFrame back to Pandas
        
        Args:
            spark_df: Input Spark DataFrame
        
        Returns:
            Pandas DataFrame
        """
        self.logger.info("Converting Spark DataFrame to Pandas...")
        
        pandas_df = spark_df.toPandas()
        
        self.logger.info(f"Converted {len(pandas_df)} records to Pandas")
        return pandas_df
    
    # ========================================================================
    # COMPLETE PIPELINE
    # ========================================================================
    
    def process_feedback_data(self, df_pandas: pd.DataFrame) -> pd.DataFrame:
        """
        Complete pipeline: Load -> Preprocess -> Convert back
        
        Args:
            df_pandas: Input Pandas DataFrame
        
        Returns:
            Preprocessed Pandas DataFrame
        """
        with utils.Timer("Data Preprocessing"):
            # Load to Spark
            spark_df = self.load_data(df_pandas)
            
            # Preprocess
            spark_df = self.preprocess(spark_df)
            
            # Generate quality report
            quality_report = self.data_quality_report(spark_df)
            self.logger.info(f"Quality Report: {quality_report}")
            
            # Convert back to Pandas
            result_df = self.to_pandas(spark_df)
            
            return result_df

# ============================================================================
# ADDITIONAL PREPROCESSING UTILITIES
# ============================================================================

def remove_stopwords(text: str) -> str:
    """
    Remove common stopwords from text
    
    Args:
        text: Input text
    
    Returns:
        Text with stopwords removed
    """
    # Basic stopwords list
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
        'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your',
        'his', 'her', 'its', 'our', 'their'
    }
    
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stopwords]
    
    return ' '.join(filtered_words)


def normalize_whitespace(text: str) -> str:
    """
    Normalize all whitespace to single spaces
    
    Args:
        text: Input text
    
    Returns:
        Normalized text
    """
    return ' '.join(text.split())

# ============================================================================
# MAIN EXECUTION (FOR TESTING)
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DATA PREPROCESSING MODULE TEST")
    print("=" * 70)
    
    # Create sample data for testing
    sample_data = pd.DataFrame({
        'feedback_id': ['FB001', 'FB002', 'FB003', 'FB004', 'FB005'],
        'text': [
            'Great product! Love it!!! https://example.com',
            'TERRIBLE quality... very disappointed 😞',
            'ok',  # Too short
            'Decent product for the price. Works well.',
            'Great product! Love it!!!'  # Duplicate
        ],
        'date': ['2024-11-01'] * 5,
        'source': ['Website', 'Twitter', 'Survey', 'Email', 'Website']
    })
    
    print("\n📊 Sample Input Data:")
    print(sample_data)
    
    # Initialize preprocessor
    print("\n🚀 Initializing Spark Preprocessor...")
    preprocessor = SparkDataPreprocessor()
    
    # Process data
    print("\n⚙️  Processing data...")
    try:
        processed_df = preprocessor.process_feedback_data(sample_data)
        
        print("\n✅ Preprocessing Complete!")
        print("\n📊 Processed Data:")
        print(processed_df)
        
        print(f"\n📈 Records: {len(sample_data)} → {len(processed_df)}")
        
    except Exception as e:
        print(f"\n❌ Error during preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Stop Spark session
        preprocessor.stop()
    
    print("\n✨ Test complete!")


# ============================================================
# PANDAS-BASED PREPROCESSING (Spark-free, local execution)
# ============================================================

import re
import pandas as pd
from nltk.corpus import stopwords

_stop_words = set(stopwords.words("english"))

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess feedback data using pandas (Spark disabled)
    """
    df = df.copy()

    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r"[^a-z\s]", "", text)
        words = [w for w in text.split() if w not in _stop_words]
        return " ".join(words)

    # IMPORTANT: column name must match your CSV
    df["text_cleaned"] = df["text"].apply(clean_text)



    return df
