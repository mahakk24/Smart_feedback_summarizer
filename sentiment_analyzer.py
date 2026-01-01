"""
Sentiment Analysis Module
Uses pre-trained BERT model for sentiment classification
"""

import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Dict, Tuple
from tqdm import tqdm
import config
import utils

# Setup logger
logger = utils.setup_logger(__name__, config.LOG_FILE)

# ============================================================================
# SENTIMENT ANALYZER CLASS
# ============================================================================

class SentimentAnalyzer:
    """
    Perform sentiment analysis on customer feedback using BERT
    """
    
    def __init__(self, model_name: str = config.SENTIMENT_MODEL):
        """
        Initialize sentiment analyzer with pre-trained model
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.logger = logger
        self.model_name = model_name
        self.device = 0 if (config.USE_GPU and torch.cuda.is_available()) else -1
        
        self.logger.info(f"Initializing sentiment analyzer with model: {model_name}")
        self.logger.info(f"Using device: {'GPU' if self.device == 0 else 'CPU'}")
        
        # Load model and tokenizer
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained sentiment analysis model"""
        try:
            self.logger.info("Loading model and tokenizer...")
            
            # Create sentiment analysis pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=self.device,
                truncation=True,
                max_length=512
            )
            
            self.logger.info("✅ Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    # ========================================================================
    # SINGLE TEXT ANALYSIS
    # ========================================================================
    
    def analyze_single(self, text: str) -> Dict[str, any]:
        """
        Analyze sentiment of a single text
        
        Args:
            text: Input text string
        
        Returns:
            Dictionary with sentiment label and confidence score
        """
        if not text or len(text.strip()) < config.MIN_TEXT_LENGTH:
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'label': 'NEUTRAL'
            }
        
        try:
            # Get prediction
            result = self.sentiment_pipeline(text[:512])[0]  # Truncate to 512 tokens
            
            # Normalize label
            label = result['label'].upper()
            confidence = result['score']
            
            # Map to standard labels
            if 'POSITIVE' in label or label == 'POS':
                sentiment = 'positive'
            elif 'NEGATIVE' in label or label == 'NEG':
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            return {
                'sentiment': sentiment,
                'confidence': round(confidence, 4),
                'label': label
            }
            
        except Exception as e:
            self.logger.warning(f"Error analyzing text: {str(e)}")
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'label': 'ERROR'
            }
    
    # ========================================================================
    # BATCH ANALYSIS
    # ========================================================================
    
    def analyze_batch(self, texts: List[str], batch_size: int = config.BATCH_SIZE) -> List[Dict]:
        """
        Analyze sentiment of multiple texts in batches
        
        Args:
            texts: List of text strings
            batch_size: Number of texts to process at once
        
        Returns:
            List of sentiment dictionaries
        """
        self.logger.info(f"Analyzing sentiment for {len(texts)} texts...")
        
        results = []
        
        # Process in batches with progress bar
        for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment Analysis"):
            batch = texts[i:i + batch_size]
            
            # Truncate texts to max length
            batch = [text[:512] if text else "" for text in batch]
            
            try:
                # Get batch predictions
                batch_results = self.sentiment_pipeline(batch)
                
                # Process results
                for result in batch_results:
                    label = result['label'].upper()
                    confidence = result['score']
                    
                    # Map to standard labels
                    if 'POSITIVE' in label or label == 'POS':
                        sentiment = 'positive'
                    elif 'NEGATIVE' in label or label == 'NEG':
                        sentiment = 'negative'
                    else:
                        sentiment = 'neutral'
                    
                    results.append({
                        'sentiment': sentiment,
                        'confidence': round(confidence, 4),
                        'label': label
                    })
                    
            except Exception as e:
                self.logger.warning(f"Error in batch {i}: {str(e)}")
                # Add neutral results for failed batch
                for _ in range(len(batch)):
                    results.append({
                        'sentiment': 'neutral',
                        'confidence': 0.0,
                        'label': 'ERROR'
                    })
        
        self.logger.info("✅ Sentiment analysis complete")
        return results
    
    # ========================================================================
    # DATAFRAME ANALYSIS
    # ========================================================================
    
    def analyze_dataframe(self, 
                          df: pd.DataFrame, 
                          text_column: str = 'text_cleaned') -> pd.DataFrame:
        """
        Analyze sentiment for all texts in a DataFrame
        
        Args:
            df: Input DataFrame
            text_column: Name of column containing text
        
        Returns:
            DataFrame with sentiment columns added
        """
        with utils.Timer("Sentiment Analysis"):
            # Validate input
            if text_column not in df.columns:
                self.logger.error(f"Column '{text_column}' not found in DataFrame")
                return df
            
            # Extract texts
            texts = df[text_column].fillna("").tolist()
            
            # Analyze sentiments
            sentiments = self.analyze_batch(texts)
            
            # Add results to DataFrame
            df['sentiment'] = [s['sentiment'] for s in sentiments]
            df['sentiment_confidence'] = [s['confidence'] for s in sentiments]
            df['sentiment_label'] = [s['label'] for s in sentiments]
            
            # Log distribution
            self._log_sentiment_distribution(df)
            
            return df
    
    # ========================================================================
    # ANALYSIS AND INSIGHTS
    # ========================================================================
    
    def _log_sentiment_distribution(self, df: pd.DataFrame):
        """Log sentiment distribution statistics"""
        if 'sentiment' not in df.columns:
            return
        
        distribution = df['sentiment'].value_counts()
        total = len(df)
        
        self.logger.info("Sentiment Distribution:")
        for sentiment, count in distribution.items():
            percentage = (count / total) * 100
            self.logger.info(f"  {sentiment.capitalize()}: {count} ({percentage:.1f}%)")
        
        # Average confidence
        if 'sentiment_confidence' in df.columns:
            avg_confidence = df['sentiment_confidence'].mean()
            self.logger.info(f"Average Confidence: {avg_confidence:.3f}")
    
    def get_sentiment_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get comprehensive sentiment summary statistics
        
        Args:
            df: DataFrame with sentiment analysis results
        
        Returns:
            Dictionary with summary statistics
        """
        if 'sentiment' not in df.columns:
            return {}
        
        total = len(df)
        distribution = df['sentiment'].value_counts()
        
        summary = {
            'total_feedbacks': total,
            'positive_count': distribution.get('positive', 0),
            'negative_count': distribution.get('negative', 0),
            'neutral_count': distribution.get('neutral', 0),
            'positive_percentage': (distribution.get('positive', 0) / total * 100) if total > 0 else 0,
            'negative_percentage': (distribution.get('negative', 0) / total * 100) if total > 0 else 0,
            'neutral_percentage': (distribution.get('neutral', 0) / total * 100) if total > 0 else 0,
            'avg_confidence': df['sentiment_confidence'].mean() if 'sentiment_confidence' in df.columns else 0
        }
        
        # Get most confident predictions by sentiment
        for sentiment in ['positive', 'negative', 'neutral']:
            sentiment_df = df[df['sentiment'] == sentiment]
            if not sentiment_df.empty and 'sentiment_confidence' in sentiment_df.columns:
                top_confident = sentiment_df.nlargest(3, 'sentiment_confidence')
                summary[f'top_{sentiment}_examples'] = top_confident['text_cleaned'].tolist()[:3]
        
        return summary
    
    # ========================================================================
    # TIME-BASED ANALYSIS
    # ========================================================================
    
    def analyze_sentiment_over_time(self, 
                                     df: pd.DataFrame, 
                                     date_column: str = 'date',
                                     freq: str = 'D') -> pd.DataFrame:
        """
        Analyze sentiment trends over time
        
        Args:
            df: DataFrame with sentiment and date columns
            date_column: Name of date column
            freq: Frequency for grouping (D=daily, W=weekly, M=monthly)
        
        Returns:
            DataFrame with time-series sentiment data
        """
        if date_column not in df.columns or 'sentiment' not in df.columns:
            self.logger.warning("Required columns not found for time analysis")
            return pd.DataFrame()
        
        # Convert to datetime
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Group by time period and sentiment
        time_sentiment = df.groupby([
            pd.Grouper(key=date_column, freq=freq),
            'sentiment'
        ]).size().reset_index(name='count')
        
        # Pivot to wide format
        pivot = time_sentiment.pivot(
            index=date_column,
            columns='sentiment',
            values='count'
        ).fillna(0)
        
        # Calculate percentages
        total = pivot.sum(axis=1)
        for col in pivot.columns:
            pivot[f'{col}_pct'] = (pivot[col] / total * 100).round(2)
        
        return pivot.reset_index()

# ============================================================================
# MAIN EXECUTION (FOR TESTING)
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SENTIMENT ANALYSIS MODULE TEST")
    print("=" * 70)
    
    # Create sample data
    sample_texts = [
        "This product is absolutely amazing! Best purchase ever!",
        "Terrible experience. Very disappointed with the quality.",
        "It's okay, nothing special. Works as expected.",
        "Outstanding customer service! Highly recommend!",
        "Worst product I've ever bought. Complete waste of money.",
        "Decent quality for the price. Fair value.",
        "Fantastic! Exceeded all my expectations!",
        "Poor quality and slow delivery. Not happy.",
        "Average product. Could be better but acceptable.",
        "Love it! Perfect for my needs!"
    ]
    
    sample_df = pd.DataFrame({
        'text_cleaned': sample_texts,
        'date': pd.date_range(start='2024-11-01', periods=10, freq='D')
    })
    
    print("\n📊 Sample Input Data:")
    print(sample_df[['text_cleaned']].head())
    
    # Initialize analyzer
    print("\n🚀 Initializing Sentiment Analyzer...")
    try:
        analyzer = SentimentAnalyzer()
        
        # Test single analysis
        print("\n🔍 Testing Single Text Analysis:")
        result = analyzer.analyze_single(sample_texts[0])
        print(f"Text: {sample_texts[0]}")
        print(f"Result: {result}")
        
        # Test batch analysis
        print("\n⚙️  Analyzing all texts...")
        analyzed_df = analyzer.analyze_dataframe(sample_df)
        
        print("\n✅ Analysis Complete!")
        print("\n📊 Results:")
        print(analyzed_df[['text_cleaned', 'sentiment', 'sentiment_confidence']])
        
        # Get summary
        print("\n📈 Sentiment Summary:")
        summary = analyzer.get_sentiment_summary(analyzed_df)
        for key, value in summary.items():
            if not key.endswith('_examples'):
                print(f"  {key}: {value}")
        
        print("\n✨ Test complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
