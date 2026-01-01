"""
Database Manager
Handles data persistence using MongoDB or SQLite
"""

import pandas as pd
from typing import Dict, List, Optional
import sqlite3
from datetime import datetime
import config
import utils

# Setup logger
logger = utils.setup_logger(__name__, config.LOG_FILE)

# ============================================================================
# SQLITE DATABASE MANAGER
# ============================================================================

class SQLiteManager:
    """Manage data storage using SQLite"""
    
    def __init__(self, db_path: str = None):
        """
        Initialize SQLite manager
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path or str(config.SQLITE_DB_PATH)
        self.logger = logger
        self.logger.info(f"Initializing SQLite Manager: {self.db_path}")
        
        # Create tables if they don't exist
        self._create_tables()
    
    def _create_tables(self):
        """Create necessary database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feedback_id TEXT UNIQUE,
                text TEXT NOT NULL,
                text_cleaned TEXT,
                date TEXT,
                source TEXT,
                category TEXT,
                product TEXT,
                rating INTEGER,
                sentiment TEXT,
                sentiment_confidence REAL,
                topic INTEGER,
                topic_name TEXT,
                text_length INTEGER,
                word_count INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create analysis_runs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_date TEXT,
                total_feedbacks INTEGER,
                positive_count INTEGER,
                negative_count INTEGER,
                neutral_count INTEGER,
                num_topics INTEGER,
                executive_summary TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        
        self.logger.info("✅ Database tables created/verified")
    
    # ========================================================================
    # INSERT OPERATIONS
    # ========================================================================
    
    def insert_feedback(self, df: pd.DataFrame) -> bool:
        """
        Insert feedback data into database
        
        Args:
            df: DataFrame with feedback data
        
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Insert data
            df.to_sql('feedback', conn, if_exists='append', index=False)
            
            conn.close()
            
            self.logger.info(f"✅ Inserted {len(df)} feedback records")
            return True
            
        except Exception as e:
            self.logger.error(f"Error inserting feedback: {str(e)}")
            return False
    
    def insert_analysis_run(self, summary_data: Dict) -> bool:
        """
        Insert analysis run summary
        
        Args:
            summary_data: Dictionary with analysis summary
        
        Returns:
            True if successful
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO analysis_runs 
                (run_date, total_feedbacks, positive_count, negative_count, 
                 neutral_count, num_topics, executive_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                summary_data.get('run_date', datetime.now().isoformat()),
                summary_data.get('total_feedbacks', 0),
                summary_data.get('positive_count', 0),
                summary_data.get('negative_count', 0),
                summary_data.get('neutral_count', 0),
                summary_data.get('num_topics', 0),
                summary_data.get('executive_summary', '')
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info("✅ Analysis run summary saved")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving analysis run: {str(e)}")
            return False
    
    # ========================================================================
    # QUERY OPERATIONS
    # ========================================================================
    
    def get_all_feedback(self, limit: int = None) -> pd.DataFrame:
        """
        Retrieve all feedback from database
        
        Args:
            limit: Maximum number of records to retrieve
        
        Returns:
            DataFrame with feedback data
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = "SELECT * FROM feedback ORDER BY date DESC"
            if limit:
                query += f" LIMIT {limit}"
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            self.logger.info(f"Retrieved {len(df)} feedback records")
            return df
            
        except Exception as e:
            self.logger.error(f"Error retrieving feedback: {str(e)}")
            return pd.DataFrame()
    
    def get_feedback_by_sentiment(self, sentiment: str) -> pd.DataFrame:
        """
        Retrieve feedback by sentiment
        
        Args:
            sentiment: Sentiment category (positive/negative/neutral)
        
        Returns:
            DataFrame with filtered feedback
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = f"SELECT * FROM feedback WHERE sentiment = '{sentiment}' ORDER BY date DESC"
            df = pd.read_sql_query(query, conn)
            
            conn.close()
            
            self.logger.info(f"Retrieved {len(df)} {sentiment} feedbacks")
            return df
            
        except Exception as e:
            self.logger.error(f"Error retrieving feedback by sentiment: {str(e)}")
            return pd.DataFrame()
    
    def get_feedback_by_date_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Retrieve feedback within date range
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            DataFrame with filtered feedback
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = f"""
                SELECT * FROM feedback 
                WHERE date >= '{start_date}' AND date <= '{end_date}'
                ORDER BY date DESC
            """
            df = pd.read_sql_query(query, conn)
            
            conn.close()
            
            self.logger.info(f"Retrieved {len(df)} feedbacks between {start_date} and {end_date}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error retrieving feedback by date: {str(e)}")
            return pd.DataFrame()
    
    def get_latest_analysis_run(self) -> Dict:
        """
        Get most recent analysis run summary
        
        Returns:
            Dictionary with analysis data
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = "SELECT * FROM analysis_runs ORDER BY created_at DESC LIMIT 1"
            df = pd.read_sql_query(query, conn)
            
            conn.close()
            
            if not df.empty:
                return df.iloc[0].to_dict()
            else:
                return {}
                
        except Exception as e:
            self.logger.error(f"Error retrieving analysis run: {str(e)}")
            return {}
    
    # ========================================================================
    # DELETE OPERATIONS
    # ========================================================================
    
    def clear_feedback_table(self) -> bool:
        """Clear all feedback data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM feedback")
            conn.commit()
            conn.close()
            
            self.logger.info("✅ Feedback table cleared")
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing feedback: {str(e)}")
            return False
    
    # ========================================================================
    # STATISTICS
    # ========================================================================
    
    def get_statistics(self) -> Dict:
        """
        Get database statistics
        
        Returns:
            Dictionary with statistics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total feedbacks
            cursor.execute("SELECT COUNT(*) FROM feedback")
            total = cursor.fetchone()[0]
            
            # Sentiment distribution
            cursor.execute("""
                SELECT sentiment, COUNT(*) as count 
                FROM feedback 
                WHERE sentiment IS NOT NULL
                GROUP BY sentiment
            """)
            sentiment_dist = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Date range
            cursor.execute("SELECT MIN(date), MAX(date) FROM feedback")
            date_range = cursor.fetchone()
            
            conn.close()
            
            return {
                'total_feedbacks': total,
                'sentiment_distribution': sentiment_dist,
                'date_range': {
                    'min': date_range[0],
                    'max': date_range[1]
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting statistics: {str(e)}")
            return {}

# ============================================================================
# MONGODB MANAGER (OPTIONAL)
# ============================================================================

class MongoDBManager:
    """Manage data storage using MongoDB"""
    
    def __init__(self):
        """Initialize MongoDB manager"""
        self.logger = logger
        self.logger.info("MongoDB manager initialized (connection pending)")
        
        # Note: Actual MongoDB connection would be implemented here
        # For this project, we'll focus on SQLite as it's simpler to set up
        self.logger.warning("MongoDB functionality not fully implemented - use SQLite instead")
    
    def insert_feedback(self, data: Dict) -> bool:
        """Insert feedback into MongoDB"""
        self.logger.warning("MongoDB insert not implemented")
        return False
    
    def query_feedback(self, query: Dict) -> List[Dict]:
        """Query feedback from MongoDB"""
        self.logger.warning("MongoDB query not implemented")
        return []

# ============================================================================
# MAIN EXECUTION (FOR TESTING)
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DATABASE MANAGER TEST")
    print("=" * 70)
    
    # Create sample data
    sample_df = pd.DataFrame({
        'feedback_id': ['FB001', 'FB002', 'FB003'],
        'text': ['Great product!', 'Poor quality.', 'Decent service.'],
        'text_cleaned': ['great product', 'poor quality', 'decent service'],
        'date': ['2024-11-01', '2024-11-02', '2024-11-03'],
        'source': ['Website', 'Twitter', 'Email'],
        'sentiment': ['positive', 'negative', 'neutral'],
        'sentiment_confidence': [0.95, 0.88, 0.75],
        'topic': [0, 1, 2],
        'topic_name': ['Quality', 'Service', 'Price']
    })
    
    # Initialize manager
    print("\n🚀 Initializing SQLite Manager...")
    db_manager = SQLiteManager()
    
    # Clear existing data
    print("\n🗑️  Clearing existing data...")
    db_manager.clear_feedback_table()
    
    # Insert data
    print("\n📝 Inserting sample data...")
    success = db_manager.insert_feedback(sample_df)
    
    if success:
        print("✅ Data inserted successfully")
        
        # Retrieve all feedback
        print("\n📊 Retrieving all feedback...")
        all_feedback = db_manager.get_all_feedback()
        print(f"Retrieved {len(all_feedback)} records")
        print(all_feedback[['feedback_id', 'text', 'sentiment']])
        
        # Get statistics
        print("\n📈 Database Statistics:")
        stats = db_manager.get_statistics()
        print(f"Total Feedbacks: {stats['total_feedbacks']}")
        print(f"Sentiment Distribution: {stats['sentiment_distribution']}")
        print(f"Date Range: {stats['date_range']}")
        
        # Insert analysis run
        print("\n💾 Saving analysis run summary...")
        summary_data = {
            'run_date': datetime.now().isoformat(),
            'total_feedbacks': 3,
            'positive_count': 1,
            'negative_count': 1,
            'neutral_count': 1,
            'num_topics': 3,
            'executive_summary': 'Mixed feedback on products and services.'
        }
        db_manager.insert_analysis_run(summary_data)
        
        # Retrieve latest run
        print("\n📋 Latest Analysis Run:")
        latest_run = db_manager.get_latest_analysis_run()
        for key, value in latest_run.items():
            print(f"  {key}: {value}")
    
    print("\n✨ Test complete!")
