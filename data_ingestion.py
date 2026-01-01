"""
Data Ingestion Module
Handles loading data from various sources: CSV, APIs, databases
"""

import pandas as pd
import requests
from typing import Optional, List, Dict
import config
import utils

# Setup logger
logger = utils.setup_logger(__name__, config.LOG_FILE)

# ============================================================================
# CSV DATA INGESTION
# ============================================================================

class DataIngestion:
    """Handle data ingestion from multiple sources"""
    
    def __init__(self):
        """Initialize data ingestion handler"""
        self.logger = logger
    
    def load_from_csv(self, file_path: str, encoding: str = 'utf-8') -> Optional[pd.DataFrame]:
        """
        Load feedback data from CSV file
        
        Args:
            file_path: Path to CSV file
            encoding: File encoding (default: utf-8)
        
        Returns:
            DataFrame with feedback data or None if error
        """
        self.logger.info(f"Loading data from CSV: {file_path}")
        
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            self.logger.info(f"Successfully loaded {len(df)} records")
            
            # Validate required columns
            required_columns = ['text']
            if not utils.validate_dataframe(df, required_columns):
                self.logger.error("CSV missing required 'text' column")
                return None
            
            return df
            
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}")
            return None
        except Exception as e:
            self.logger.error(f"Error loading CSV: {str(e)}")
            return None
    
    def load_from_multiple_csvs(self, file_paths: List[str]) -> Optional[pd.DataFrame]:
        """
        Load and combine data from multiple CSV files
        
        Args:
            file_paths: List of CSV file paths
        
        Returns:
            Combined DataFrame or None if error
        """
        self.logger.info(f"Loading data from {len(file_paths)} CSV files")
        
        dataframes = []
        
        for file_path in file_paths:
            df = self.load_from_csv(file_path)
            if df is not None:
                dataframes.append(df)
        
        if not dataframes:
            self.logger.error("No valid data loaded from CSV files")
            return None
        
        # Combine all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)
        self.logger.info(f"Combined {len(combined_df)} total records")
        
        return combined_df
    
    # ========================================================================
    # API DATA INGESTION (SIMULATED)
    # ========================================================================
    
    def load_from_twitter_api(self, 
                               query: str, 
                               max_results: int = 100) -> Optional[pd.DataFrame]:
        """
        Load feedback from Twitter API (Simulated)
        In production, this would use actual Twitter API v2
        
        Args:
            query: Search query
            max_results: Maximum number of tweets to fetch
        
        Returns:
            DataFrame with tweet data
        """
        self.logger.info(f"Simulating Twitter API call for query: {query}")
        
        # SIMULATION: In production, replace with actual Twitter API call
        # Example using tweepy:
        # import tweepy
        # auth = tweepy.OAuth2BearerToken(config.TWITTER_API_KEY)
        # api = tweepy.API(auth)
        # tweets = api.search_tweets(q=query, count=max_results)
        
        # For demo purposes, return sample structure
        sample_data = {
            'source': ['Twitter'] * 10,
            'text': [
                'Great product! Love using it every day.',
                'Not satisfied with customer service.',
                'Amazing quality for the price!',
                'Delivery was late and product was damaged.',
                'Best purchase this year!',
                'Could be better, but okay for now.',
                'Terrible experience, would not recommend.',
                'Works perfectly! Highly recommended.',
                'Average product, nothing special.',
                'Outstanding! Exceeded expectations.'
            ],
            'date': pd.date_range(start='2024-11-01', periods=10, freq='D'),
            'username': [f'user_{i}' for i in range(10)]
        }
        
        df = pd.DataFrame(sample_data)
        df['date'] = df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        self.logger.info(f"Simulated fetch of {len(df)} tweets")
        return df
    
    def load_from_review_api(self, 
                             product_id: str,
                             api_endpoint: str = None) -> Optional[pd.DataFrame]:
        """
        Load product reviews from API (Simulated)
        
        Args:
            product_id: Product identifier
            api_endpoint: API endpoint URL
        
        Returns:
            DataFrame with review data
        """
        self.logger.info(f"Simulating review API call for product: {product_id}")
        
        # SIMULATION: In production, replace with actual API call
        # Example:
        # response = requests.get(f"{api_endpoint}/products/{product_id}/reviews")
        # data = response.json()
        
        # For demo purposes, return sample structure
        sample_data = {
            'source': ['Website Review'] * 5,
            'text': [
                'Excellent quality and fast shipping!',
                'Not worth the price, disappointed.',
                'Good product, meets expectations.',
                'Amazing! Will buy again.',
                'Poor customer service experience.'
            ],
            'date': pd.date_range(start='2024-11-01', periods=5, freq='D'),
            'rating': [5, 2, 3, 5, 1],
            'product_id': [product_id] * 5
        }
        
        df = pd.DataFrame(sample_data)
        df['date'] = df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        self.logger.info(f"Simulated fetch of {len(df)} reviews")
        return df
    
    # ========================================================================
    # DATABASE DATA INGESTION
    # ========================================================================
    
    def load_from_sqlite(self, 
                         db_path: str = None, 
                         table_name: str = 'feedback') -> Optional[pd.DataFrame]:
        """
        Load feedback data from SQLite database
        
        Args:
            db_path: Path to SQLite database
            table_name: Name of table to query
        
        Returns:
            DataFrame with feedback data
        """
        if db_path is None:
            db_path = config.SQLITE_DB_PATH
        
        self.logger.info(f"Loading data from SQLite: {db_path}")
        
        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql_query(query, conn)
            
            conn.close()
            
            self.logger.info(f"Successfully loaded {len(df)} records from SQLite")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading from SQLite: {str(e)}")
            return None
    
    # ========================================================================
    # DATA VALIDATION AND PREPROCESSING
    # ========================================================================
    
    def validate_and_standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and standardize loaded data
        
        Args:
            df: Input DataFrame
        
        Returns:
            Standardized DataFrame
        """
        self.logger.info("Validating and standardizing data...")
        
        # Ensure 'text' column exists
        if 'text' not in df.columns:
            self.logger.error("Required 'text' column not found")
            return df
        
        # Standardize column names
        df.columns = df.columns.str.lower().str.strip()
        
        # Add missing standard columns
        if 'date' not in df.columns:
            df['date'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if 'source' not in df.columns:
            df['source'] = 'Unknown'
        
        if 'feedback_id' not in df.columns:
            df['feedback_id'] = [f'FB{i:05d}' for i in range(len(df))]
        
        # Remove rows with empty text
        initial_count = len(df)
        df = df[df['text'].notna()]
        df = df[df['text'].str.strip() != '']
        removed = initial_count - len(df)
        
        if removed > 0:
            self.logger.info(f"Removed {removed} rows with empty text")
        
        # Reset index
        df = df.reset_index(drop=True)
        
        self.logger.info(f"Validation complete. {len(df)} valid records.")
        
        return df

# ============================================================================
# MAIN EXECUTION (FOR TESTING)
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DATA INGESTION MODULE TEST")
    print("=" * 70)
    
    # Initialize ingestion handler
    ingestion = DataIngestion()
    
    # Test CSV loading
    csv_path = config.DATA_DIR / "customer_feedback.csv"
    
    if csv_path.exists():
        print(f"\n📂 Loading data from: {csv_path}")
        df = ingestion.load_from_csv(str(csv_path))
        
        if df is not None:
            print(f"\n✅ Successfully loaded {len(df)} records")
            print(f"\nColumns: {df.columns.tolist()}")
            print(f"\nFirst 5 records:")
            print(df.head())
            
            # Validate and standardize
            df = ingestion.validate_and_standardize(df)
            print(f"\n✅ After validation: {len(df)} records")
    else:
        print(f"\n⚠️  Sample data not found. Run generate_sample_data.py first.")
    
    # Test API simulation
    print("\n" + "=" * 70)
    print("TESTING API INGESTION (SIMULATED)")
    print("=" * 70)
    
    twitter_df = ingestion.load_from_twitter_api("product feedback", max_results=10)
    print(f"\n✅ Twitter simulation: {len(twitter_df)} records")
    print(twitter_df.head())
    
    review_df = ingestion.load_from_review_api("PROD123")
    print(f"\n✅ Review API simulation: {len(review_df)} records")
    print(review_df.head())
