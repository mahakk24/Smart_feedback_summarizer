"""
Main Pipeline Orchestrator
Complete end-to-end feedback analysis pipeline with interactive demonstration
"""

import pandas as pd
from datetime import datetime
import sys
import time
from pathlib import Path
import config
import utils
from data_ingestion import DataIngestion
from data_preprocessing import SparkDataPreprocessor
from data_preprocessing import preprocess_data
from sentiment_analyzer import SentimentAnalyzer
from topic_extractor import TopicExtractor
from text_summarizer import TextSummarizer
from database_manager import SQLiteManager

# Setup logger
logger = utils.setup_logger(__name__, config.LOG_FILE)

# ============================================================================
# UTILITY FUNCTIONS FOR DISPLAY
# ============================================================================

def print_header(text, char="="):
    """Print formatted header"""
    width = 70
    print("\n" + char * width)
    print(text.center(width))
    print(char * width + "\n")

def print_subheader(text, char="-"):
    """Print formatted subheader"""
    width = 70
    print("\n" + char * width)
    print(text)
    print(char * width)

def print_progress(step, total, description):
    """Print progress indicator"""
    percentage = (step / total) * 100
    bar_length = 40
    filled = int(bar_length * step / total)
    bar = "█" * filled + "░" * (bar_length - filled)
    print(f"\r[{bar}] {percentage:.0f}% - {description}", end="", flush=True)
    if step == total:
        print()  # New line when complete

def print_metric(label, value, unit=""):
    """Print formatted metric"""
    print(f"  ✓ {label:.<40} {value} {unit}")

def print_success(message):
    """Print success message"""
    print(f"\n✅ {message}")

def print_warning(message):
    """Print warning message"""
    print(f"\n⚠️  {message}")

def print_error(message):
    """Print error message"""
    print(f"\n❌ {message}")

# ============================================================================
# FEEDBACK ANALYSIS PIPELINE
# ============================================================================

class FeedbackAnalysisPipeline:
    """
    Complete pipeline for customer feedback analysis
    """
    
    def __init__(self):
        """Initialize pipeline components"""
        self.logger = logger
        self.logger.info("="  * 70)
        self.logger.info("INITIALIZING FEEDBACK ANALYSIS PIPELINE")
        self.logger.info("=" * 70)
        
        # Initialize components
        self.ingestion = DataIngestion()
        self.preprocessor = None  # Initialize on demand
        self.sentiment_analyzer = None
        self.topic_extractor = None
        self.text_summarizer = None
        self.db_manager = SQLiteManager()
        
        self.df = None
    
    # ========================================================================
    # STEP 1: DATA INGESTION
    # ========================================================================
    
    def load_data(self, source: str, **kwargs):
        """
        Load data from specified source
        
        Args:
            source: Data source type ('csv', 'api', 'database')
            **kwargs: Additional arguments for specific source
        
        Returns:
            Loaded DataFrame
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info("STEP 1: DATA INGESTION")
        self.logger.info("=" * 70)
        
        if source == 'csv':
            file_path = kwargs.get('file_path')
            self.df = self.ingestion.load_from_csv(file_path)
        
        elif source == 'database':
            limit = kwargs.get('limit', 1000)
            self.df = self.db_manager.get_all_feedback(limit=limit)
        
        else:
            self.logger.error(f"Unknown source: {source}")
            return None
        
        # Validate and standardize
        if self.df is not None:
            self.df = self.ingestion.validate_and_standardize(self.df)
            self.logger.info(f"✅ Loaded {len(self.df)} records")
        
        return self.df
    
    # ========================================================================
    # STEP 2: DATA PREPROCESSING
    # ========================================================================
    
    def preprocess_data(self):
        """
        Clean and preprocess feedback data using PySpark
        
        Returns:
            Preprocessed DataFrame
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info("STEP 2: DATA PREPROCESSING (PySpark)")
        self.logger.info("=" * 70)
        
        if self.df is None:
            self.logger.error("No data loaded. Run load_data() first.")
            return None
        
        ## Use pandas preprocessing instead of Spark

        self.logger.info("STEP 2: DATA PREPROCESSING (Pandas)")
        self.df = preprocess_data(self.df)
        
        self.logger.info(f"✅ Preprocessing complete: {len(self.df)} records")
        
        return self.df
    
    # ========================================================================
    # STEP 3: SENTIMENT ANALYSIS
    # ========================================================================
    
    def analyze_sentiment(self):
        """
        Perform sentiment analysis on feedback
        
        Returns:
            DataFrame with sentiment scores
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info("STEP 3: SENTIMENT ANALYSIS")
        self.logger.info("=" * 70)
        
        if self.df is None:
            self.logger.error("No data available. Run preprocessing first.")
            return None
        
        # Initialize analyzer
        if self.sentiment_analyzer is None:
            self.sentiment_analyzer = SentimentAnalyzer()
        
        # Analyze sentiments
        self.df = self.sentiment_analyzer.analyze_dataframe(self.df)
        
        # Get summary
        summary = self.sentiment_analyzer.get_sentiment_summary(self.df)

        # Defensive handling (summary contains counts, not percentages)
        positive = int(summary.get("positive", 0))
        negative = int(summary.get("negative", 0))
        neutral  = int(summary.get("neutral", 0))

        total = positive + negative + neutral


        positive_pct = (summary.get("positive", 0) / total * 100) if total else 0
        negative_pct = (summary.get("negative", 0) / total * 100) if total else 0
        neutral_pct  = (summary.get("neutral", 0) / total * 100) if total else 0

        self.logger.info("[OK] Sentiment analysis complete")
        self.logger.info(f"   Positive: {positive_pct:.1f}%")
        self.logger.info(f"   Negative: {negative_pct:.1f}%")
        self.logger.info(f"   Neutral: {neutral_pct:.1f}%")

        return self.df
    
    # ========================================================================
    # STEP 4: TOPIC EXTRACTION
    # ========================================================================
    
    def extract_topics(self):
        """
        Extract topics from feedback
        
        Returns:
            DataFrame with topic assignments
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info("STEP 4: TOPIC EXTRACTION")
        self.logger.info("=" * 70)
        
        if self.df is None:
            self.logger.error("No data available. Run preprocessing first.")
            return None
        
        # Initialize extractor
        if self.topic_extractor is None:
            self.topic_extractor = TopicExtractor()
        
        # Extract topics
        self.df = self.topic_extractor.extract_topics_from_dataframe(self.df)
        
        # Get summary
        summary = self.topic_extractor.get_topic_summary(self.df)
        
        self.logger.info(f"✅ Topic extraction complete")
        self.logger.info(f"   Topics identified: {summary['num_topics']}")
        
        return self.df
    
    # ========================================================================
    # STEP 5: TEXT SUMMARIZATION
    # ========================================================================
    
    def generate_summaries(self):
        """
        Generate summaries of feedback
        
        Returns:
            Dictionary with various summaries
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info("STEP 5: TEXT SUMMARIZATION")
        self.logger.info("=" * 70)
        
        if self.df is None:
            self.logger.error("No data available.")
            return None
        
        # Initialize summarizer
        if self.text_summarizer is None:
            self.text_summarizer = TextSummarizer()
        
        # Generate insights
        insights = self.text_summarizer.extract_key_insights(self.df)
        
        self.logger.info(f"✅ Summaries generated")
        self.logger.info(f"   Executive summary length: {len(insights.get('executive_summary', ''))} chars")
        
        return insights
    
    # ========================================================================
    # STEP 6: SAVE RESULTS
    # ========================================================================
    
    def save_results(self):
        """
        Save analysis results to database
        
        Returns:
            True if successful
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info("STEP 6: SAVING RESULTS")
        self.logger.info("=" * 70)
        
        if self.df is None:
            self.logger.error("No results to save.")
            return False
        
        # Clear existing data
        self.db_manager.clear_feedback_table()
        
        # Insert new data
        success = self.db_manager.insert_feedback(self.df)
        
        if success:
            # Save analysis run summary
            if 'sentiment' in self.df.columns:
                summary_data = {
                    'run_date': datetime.now().isoformat(),
                    'total_feedbacks': len(self.df),
                    'positive_count': (self.df['sentiment'] == 'positive').sum(),
                    'negative_count': (self.df['sentiment'] == 'negative').sum(),
                    'neutral_count': (self.df['sentiment'] == 'neutral').sum(),
                    'num_topics': self.df[self.df['topic'] != -1]['topic'].nunique() if 'topic' in self.df.columns else 0,
                    'executive_summary': ''
                }
                self.db_manager.insert_analysis_run(summary_data)
            
            self.logger.info(f"✅ Results saved to database")
            return True
        
        return False
    
    # ========================================================================
    # COMPLETE PIPELINE
    # ========================================================================
    
    def run_complete_pipeline(self, data_source: str, **kwargs):
        """
        Run complete analysis pipeline
        
        Args:
            data_source: Source of data ('csv', 'database')
            **kwargs: Additional arguments
        
        Returns:
            Tuple of (DataFrame, insights)
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info("STARTING COMPLETE FEEDBACK ANALYSIS PIPELINE")
        self.logger.info("=" * 70)
        
        start_time = datetime.now()
        total_steps = 6
        
        try:
            # Step 1: Load data
            print_progress(1, total_steps, "Loading data...")
            self.load_data(data_source, **kwargs)
            print_success(f"Data loaded: {len(self.df)} records")
            
            # Step 2: Preprocess
            print_progress(2, total_steps, "Preprocessing with PySpark...")
            self.preprocess_data()
            print_success(f"Data preprocessed: {len(self.df)} valid records")
            
            # Step 3: Sentiment analysis
            print_progress(3, total_steps, "Analyzing sentiment with BERT...")
            self.analyze_sentiment()
            print_success("Sentiment analysis complete")
            
            # Step 4: Topic extraction
            print_progress(4, total_steps, "Extracting topics with BERTopic...")
            self.extract_topics()
            print_success("Topic extraction complete")
            
            # Step 5: Generate summaries
            print_progress(5, total_steps, "Generating summaries with BART...")
            insights = self.generate_summaries()
            print_success("Summaries generated")
            
            # Step 6: Save results
            print_progress(6, total_steps, "Saving to database...")
            self.save_results()
            print_success("Results saved to database")
            
            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()
            
            self.logger.info("\n" + "=" * 70)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 70)
            self.logger.info(f"Total duration: {duration:.2f} seconds")
            self.logger.info(f"Records processed: {len(self.df)}")
            
            return self.df, insights
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            print_error(f"Pipeline failed: {str(e)}")
            return None, None

# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def display_analysis_results(df, insights):
    """Display comprehensive analysis results"""
    
    print_header("📊 ANALYSIS RESULTS", "=")
    
    # Basic Statistics
    print_subheader("📈 Basic Statistics")
    print_metric("Total Feedbacks Analyzed", len(df))
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        date_range = df['date'].max() - df['date'].min()
        print_metric("Date Range", f"{date_range.days} days")
        print_metric("From", df['date'].min().strftime('%Y-%m-%d'))
        print_metric("To", df['date'].max().strftime('%Y-%m-%d'))
    
    if 'source' in df.columns:
        print_metric("Data Sources", df['source'].nunique())
    
    # Sentiment Analysis Results
    if 'sentiment' in df.columns:
        print_subheader("😊 Sentiment Analysis Results")
        
        sentiment_counts = df['sentiment'].value_counts()
        total = len(df)
        
        for sentiment in ['positive', 'negative', 'neutral']:
            count = sentiment_counts.get(sentiment, 0)
            percentage = (count / total) * 100
            
            # Create visual bar
            bar_length = int(percentage / 2)
            bar = "█" * bar_length
            
            emoji = "😊" if sentiment == 'positive' else "😞" if sentiment == 'negative' else "😐"
            print(f"  {emoji} {sentiment.capitalize():.<25} {count:>4} ({percentage:>5.1f}%) {bar}")
        
        if 'sentiment_confidence' in df.columns:
            avg_confidence = df['sentiment_confidence'].mean()
            print_metric("Average Confidence", f"{avg_confidence:.1%}")
    
    # Topic Extraction Results
    if 'topic' in df.columns:
        print_subheader("🏷️  Topic Analysis Results")
        
        # Filter out outlier topic (-1)
        topics_df = df[df['topic'] != -1]
        n_topics = topics_df['topic'].nunique()
        
        print_metric("Topics Identified", n_topics)
        print_metric("Outliers", len(df[df['topic'] == -1]))
        
        if 'topic_name' in df.columns and not topics_df.empty:
            print("\n  Top Topics:")
            topic_counts = topics_df['topic_name'].value_counts().head(5)
            
            for i, (topic, count) in enumerate(topic_counts.items(), 1):
                percentage = (count / len(df)) * 100
                print(f"    {i}. {topic:.<35} {count:>4} ({percentage:>5.1f}%)")
    
    # Summaries
    if insights:
        print_subheader("📝 Executive Summary")
        exec_summary = insights.get('executive_summary', 'N/A')
        
        # Word wrap the summary
        import textwrap
        wrapped = textwrap.fill(exec_summary, width=68)
        for line in wrapped.split('\n'):
            print(f"  {line}")
        
        # Sentiment-specific summaries
        if 'sentiment_summaries' in insights:
            print_subheader("📋 Sentiment-Specific Insights")
            
            for sentiment, summary in insights['sentiment_summaries'].items():
                if summary:
                    emoji = "😊" if sentiment == 'positive' else "😞" if sentiment == 'negative' else "😐"
                    print(f"\n  {emoji} {sentiment.upper()} Feedback:")
                    wrapped = textwrap.fill(summary, width=66)
                    for line in wrapped.split('\n'):
                        print(f"    {line}")
    
    # Data Quality Metrics
    print_subheader("✅ Data Quality Metrics")
    
    if 'text_length' in df.columns:
        print_metric("Avg Text Length", f"{df['text_length'].mean():.0f}", "characters")
    
    if 'word_count' in df.columns:
        print_metric("Avg Word Count", f"{df['word_count'].mean():.0f}", "words")
    
    # Performance Info
    print_subheader("⚡ Performance Information")
    print_metric("Database", "SQLite (feedback.db)")
    print_metric("Models Used", "BERT + BERTopic + BART")
    print_metric("Processing Mode", "Distributed (PySpark)")

def display_sample_feedbacks(df, n=5):
    """Display sample analyzed feedbacks"""
    
    print_header("📋 SAMPLE ANALYZED FEEDBACKS", "=")
    
    if df.empty:
        print("  No data to display")
        return
    
    # Get diverse samples
    samples = []
    
    if 'sentiment' in df.columns:
        for sentiment in ['positive', 'negative', 'neutral']:
            sentiment_df = df[df['sentiment'] == sentiment]
            if not sentiment_df.empty:
                sample = sentiment_df.sample(min(2, len(sentiment_df)))
                samples.append(sample)
    
    if samples:
        sample_df = pd.concat(samples).head(n)
    else:
        sample_df = df.head(n)
    
    for idx, row in sample_df.iterrows():
        print(f"\n  📄 Sample {idx + 1}")
        print(f"  {'-' * 68}")
        
        if 'text_cleaned' in row:
            text = row['text_cleaned']
        elif 'text' in row:
            text = row['text']
        else:
            text = "N/A"
        
        # Truncate long text
        if len(text) > 150:
            text = text[:150] + "..."
        
        print(f"  Text: {text}")
        
        if 'sentiment' in row:
            emoji = "😊" if row['sentiment'] == 'positive' else "😞" if row['sentiment'] == 'negative' else "😐"
            conf = f" ({row.get('sentiment_confidence', 0):.1%})" if 'sentiment_confidence' in row else ""
            print(f"  Sentiment: {emoji} {row['sentiment'].upper()}{conf}")
        
        if 'topic_name' in row and row['topic_name'] != 'Outlier':
            print(f"  Topic: 🏷️  {row['topic_name']}")
        
        if 'source' in row:
            print(f"  Source: {row['source']}")
        
        if 'date' in row:
            print(f"  Date: {row['date']}")

def display_export_info(df):
    """Display information about exports and next steps"""
    
    print_header("💾 DATA SAVED & NEXT STEPS", "=")
    
    print("  📂 Your analyzed data has been saved to:")
    print(f"     • Database: {config.SQLITE_DB_PATH}")
    print(f"     • Records: {len(df)}")
    
    print("\n  🎯 Next Steps:")
    print("     1. View interactive dashboard:")
    print("        → streamlit run dashboard.py")
    print()
    print("     2. Query the database:")
    print("        → python -c \"from database_manager import SQLiteManager;")
    print("           db = SQLiteManager(); print(db.get_statistics())\"")
    print()
    print("     3. Export results to CSV:")
    print("        → Check outputs/ directory")
    
    print("\n  📊 Dashboard Features:")
    print("     • Interactive charts and visualizations")
    print("     • Real-time filtering and analysis")
    print("     • CSV export functionality")
    print("     • Topic and sentiment drill-downs")
    
    print("\n  📖 Documentation:")
    print("     • Full docs: README.md")
    print("     • Quick start: QUICKSTART.md")
    print("     • Technical: DOCUMENTATION.md")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function with comprehensive demonstration"""
    
    print_header("🚀 SMART FEEDBACK SUMMARIZER", "=")
    print("  AI-Powered Customer Feedback Analysis System")
    print("  Using BERT • BERTopic • BART • PySpark")
    
    # Check if sample data exists
    sample_data_path = config.DATA_DIR / "customer_feedback.csv"
    
    if not sample_data_path.exists():
        print_warning("Sample data not found. Generating now...")
        print("\n  This will create 500+ synthetic customer feedback records...")
        
        try:
            import generate_sample_data
            print_success("Sample data generated successfully!")
        except Exception as e:
            print_error(f"Failed to generate sample data: {str(e)}")
            print("  Please run: python generate_sample_data.py")
            return
    
    # Display file info
    print(f"\n  📁 Data Source: {sample_data_path}")
    
    # Ask for confirmation
    print("\n  " + "=" * 68)
    response = input("  Ready to run complete analysis? (y/n): ").strip().lower()
    
    if response != 'y':
        print("\n  Analysis cancelled.")
        print("  To run later: python main.py")
        return
    
    # Initialize pipeline
    print_header("⚙️  INITIALIZING PIPELINE", "-")
    print("  Loading AI models and initializing components...")
    print("  This may take a minute on first run (downloading models)...\n")
    
    try:
        pipeline = FeedbackAnalysisPipeline()
    except Exception as e:
        print_error(f"Failed to initialize pipeline: {str(e)}")
        return
    
    # Run complete pipeline
    print_header("🔄 RUNNING ANALYSIS PIPELINE", "-")
    
    start_time = time.time()
    
    df, insights = pipeline.run_complete_pipeline(
        data_source='csv',
        file_path=str(sample_data_path)
    )
    
    elapsed_time = time.time() - start_time
    
    if df is not None and not df.empty:
        # Display results
        display_analysis_results(df, insights)
        
        # Display samples
        display_sample_feedbacks(df, n=5)
        
        # Performance metrics
        print_header("⚡ PERFORMANCE METRICS", "=")
        print_metric("Total Processing Time", f"{elapsed_time:.2f}", "seconds")
        print_metric("Records Processed", len(df))
        print_metric("Processing Rate", f"{len(df) / elapsed_time:.1f}", "records/sec")
        
        if 'text_length' in df.columns:
            total_chars = df['text_length'].sum()
            print_metric("Total Characters Processed", f"{total_chars:,}")
            print_metric("Character Processing Rate", f"{total_chars / elapsed_time:,.0f}", "chars/sec")
        
        # Export and next steps
        display_export_info(df)
        
        # Final success message
        print_header("✅ ANALYSIS COMPLETE!", "=")
        print("  Your feedback analysis is ready!")
        print("  Launch the dashboard to explore interactively:\n")
        print("  → streamlit run dashboard.py\n")
        
    else:
        print_error("Analysis failed. Please check the logs for details.")
        print(f"  Log file: {config.LOG_FILE}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user.")
        print("  You can restart anytime: python main.py")
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        import traceback
        print("\n" + traceback.format_exc())
        print(f"\n  Check logs: {config.LOG_FILE}")

