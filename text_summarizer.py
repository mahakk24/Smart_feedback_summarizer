"""
Text Summarization Module
Uses BART/Pegasus for generating concise summaries of customer feedback
"""

import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import List, Dict
from tqdm import tqdm
import config
import utils

# Setup logger
logger = utils.setup_logger(__name__, config.LOG_FILE)

# ============================================================================
# TEXT SUMMARIZER CLASS
# ============================================================================

class TextSummarizer:
    """
    Generate summaries of customer feedback using transformer models
    """
    
    def __init__(self, model_name: str = config.SUMMARIZATION_MODEL):
        """
        Initialize text summarizer
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.logger = logger
        self.model_name = model_name
        self.device = 0 if (config.USE_GPU and torch.cuda.is_available()) else -1
        
        self.logger.info(f"Initializing text summarizer with model: {model_name}")
        self.logger.info(f"Using device: {'GPU' if self.device == 0 else 'CPU'}")
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained summarization model"""
        try:
            self.logger.info("Loading summarization model...")
            
            # Create summarization pipeline
            self.summarizer = pipeline(
                "summarization",
                model=self.model_name,
                device=self.device,
                truncation=True
            )
            
            self.logger.info("✅ Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    # ========================================================================
    # SINGLE TEXT SUMMARIZATION
    # ========================================================================
    
    def summarize_single(self, 
                        text: str,
                        max_length: int = config.MAX_SUMMARY_LENGTH,
                        min_length: int = config.MIN_SUMMARY_LENGTH) -> str:
        """
        Summarize a single text
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of summary
            min_length: Minimum length of summary
        
        Returns:
            Summary string
        """
        if not text or len(text.split()) < 10:
            return text  # Return original if too short
        
        try:
            # Generate summary
            summary = self.summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True
            )
            
            return summary[0]['summary_text']
            
        except Exception as e:
            self.logger.warning(f"Error summarizing text: {str(e)}")
            return text[:200] + "..."  # Fallback to truncation
    
    # ========================================================================
    # BATCH SUMMARIZATION
    # ========================================================================
    
    def summarize_batch(self, 
                        texts: List[str],
                        max_length: int = config.MAX_SUMMARY_LENGTH,
                        min_length: int = config.MIN_SUMMARY_LENGTH,
                        batch_size: int = 4) -> List[str]:
        """
        Summarize multiple texts in batches
        
        Args:
            texts: List of texts to summarize
            max_length: Maximum length of summaries
            min_length: Minimum length of summaries
            batch_size: Number of texts to process at once
        
        Returns:
            List of summary strings
        """
        self.logger.info(f"Summarizing {len(texts)} texts...")
        
        summaries = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Summarization"):
            batch = texts[i:i + batch_size]
            
            # Filter texts that are long enough
            batch_results = []
            for text in batch:
                if len(text.split()) >= 10:
                    try:
                        summary = self.summarizer(
                            text,
                            max_length=max_length,
                            min_length=min_length,
                            do_sample=False,
                            truncation=True
                        )
                        batch_results.append(summary[0]['summary_text'])
                    except Exception as e:
                        self.logger.warning(f"Error in summarization: {str(e)}")
                        batch_results.append(text[:200] + "...")
                else:
                    batch_results.append(text)  # Return original if too short
            
            summaries.extend(batch_results)
        
        self.logger.info("✅ Summarization complete")
        return summaries
    
    # ========================================================================
    # AGGREGATE SUMMARIZATION
    # ========================================================================
    
    def summarize_multiple_feedbacks(self, 
                                     texts: List[str],
                                     max_length: int = 200,
                                     min_length: int = 100) -> str:
        """
        Create a single summary from multiple feedback texts
        
        Args:
            texts: List of feedback texts
            max_length: Maximum length of final summary
            min_length: Minimum length of final summary
        
        Returns:
            Aggregated summary string
        """
        if not texts:
            return ""
        
        # Combine texts (with length limit)
        combined = " ".join(texts)
        
        # Truncate if too long (BART has 1024 token limit)
        max_chars = 3000
        if len(combined) > max_chars:
            combined = combined[:max_chars]
        
        # Generate summary
        try:
            summary = self.summarizer(
                combined,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True
            )
            return summary[0]['summary_text']
        except Exception as e:
            self.logger.error(f"Error in aggregate summarization: {str(e)}")
            # Fallback: return excerpt
            return combined[:300] + "..."
    
    # ========================================================================
    # SENTIMENT-SPECIFIC SUMMARIES
    # ========================================================================
    
    def summarize_by_sentiment(self, 
                               df: pd.DataFrame,
                               text_column: str = 'text_cleaned') -> Dict[str, str]:
        """
        Create separate summaries for each sentiment category
        
        Args:
            df: DataFrame with text and sentiment columns
            text_column: Name of text column
        
        Returns:
            Dictionary with sentiment-specific summaries
        """
        self.logger.info("Creating sentiment-specific summaries...")
        
        summaries = {}
        
        if 'sentiment' not in df.columns:
            self.logger.warning("No sentiment column found")
            return summaries
        
        for sentiment in ['positive', 'negative', 'neutral']:
            sentiment_df = df[df['sentiment'] == sentiment]
            
            if not sentiment_df.empty:
                # Get sample of texts
                sample_size = min(20, len(sentiment_df))
                sample_texts = sentiment_df[text_column].sample(sample_size).tolist()
                
                # Generate summary
                summary = self.summarize_multiple_feedbacks(
                    sample_texts,
                    max_length=150,
                    min_length=50
                )
                
                summaries[sentiment] = summary
                self.logger.info(f"  {sentiment.capitalize()}: {len(summary)} chars")
        
        return summaries
    
    # ========================================================================
    # TOPIC-SPECIFIC SUMMARIES
    # ========================================================================
    
    def summarize_by_topic(self, 
                          df: pd.DataFrame,
                          text_column: str = 'text_cleaned',
                          max_topics: int = 5) -> Dict[str, str]:
        """
        Create separate summaries for each topic
        
        Args:
            df: DataFrame with text and topic columns
            text_column: Name of text column
            max_topics: Maximum number of topics to summarize
        
        Returns:
            Dictionary with topic-specific summaries
        """
        self.logger.info("Creating topic-specific summaries...")
        
        summaries = {}
        
        if 'topic' not in df.columns:
            self.logger.warning("No topic column found")
            return summaries
        
        # Get top topics (excluding outliers)
        top_topics = df[df['topic'] != -1]['topic'].value_counts().head(max_topics).index
        
        for topic_id in top_topics:
            topic_df = df[df['topic'] == topic_id]
            
            if not topic_df.empty:
                # Get sample of texts
                sample_size = min(15, len(topic_df))
                sample_texts = topic_df[text_column].sample(sample_size).tolist()
                
                # Get topic name
                topic_name = topic_df['topic_name'].iloc[0] if 'topic_name' in topic_df.columns else f"Topic {topic_id}"
                
                # Generate summary
                summary = self.summarize_multiple_feedbacks(
                    sample_texts,
                    max_length=150,
                    min_length=50
                )
                
                summaries[topic_name] = summary
                self.logger.info(f"  {topic_name}: {len(summary)} chars")
        
        return summaries
    
    # ========================================================================
    # EXECUTIVE SUMMARY
    # ========================================================================
    
    def generate_executive_summary(self, 
                                   df: pd.DataFrame,
                                   text_column: str = 'text_cleaned',
                                   max_samples: int = 50) -> str:
        """
        Generate an executive summary of all feedback
        
        Args:
            df: DataFrame with feedback
            text_column: Name of text column
            max_samples: Maximum number of samples to include
        
        Returns:
            Executive summary string
        """
        self.logger.info("Generating executive summary...")
        
        with utils.Timer("Executive Summary Generation"):
            # Sample diverse feedback
            sample_size = min(max_samples, len(df))
            
            # Try to get balanced sample across sentiments
            if 'sentiment' in df.columns:
                samples = []
                for sentiment in ['positive', 'negative', 'neutral']:
                    sent_df = df[df['sentiment'] == sentiment]
                    n = min(sample_size // 3, len(sent_df))
                    if n > 0:
                        samples.extend(sent_df[text_column].sample(n).tolist())
            else:
                samples = df[text_column].sample(sample_size).tolist()
            
            # Generate summary
            executive_summary = self.summarize_multiple_feedbacks(
                samples,
                max_length=250,
                min_length=100
            )
            
            self.logger.info(f"✅ Executive summary generated ({len(executive_summary)} chars)")
            
            return executive_summary
    
    # ========================================================================
    # KEY INSIGHTS EXTRACTION
    # ========================================================================
    
    def extract_key_insights(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Extract key insights from feedback data
        
        Args:
            df: DataFrame with analyzed feedback
        
        Returns:
            Dictionary with key insights
        """
        insights = {
            'total_feedbacks': len(df),
            'date_range': f"{df['date'].min()} to {df['date'].max()}" if 'date' in df.columns else "N/A"
        }
        
        # Sentiment insights
        if 'sentiment' in df.columns:
            sentiments = df['sentiment'].value_counts()
            insights['sentiment_summary'] = {
                'most_common': sentiments.index[0],
                'distribution': sentiments.to_dict()
            }
            
            # Get sentiment summaries
            insights['sentiment_summaries'] = self.summarize_by_sentiment(df)
        
        # Topic insights
        if 'topic' in df.columns:
            topics = df[df['topic'] != -1]['topic'].value_counts()
            insights['topic_summary'] = {
                'num_topics': len(topics),
                'top_topic': topics.index[0] if len(topics) > 0 else None
            }
            
            # Get topic summaries
            insights['topic_summaries'] = self.summarize_by_topic(df)
        
        # Executive summary
        insights['executive_summary'] = self.generate_executive_summary(df)
        
        return insights

# ============================================================================
# MAIN EXECUTION (FOR TESTING)
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TEXT SUMMARIZATION MODULE TEST")
    print("=" * 70)
    
    # Create sample data
    sample_texts = [
        "This product has been absolutely amazing. The quality exceeded my expectations and the customer service team was incredibly helpful throughout the entire process. I would highly recommend this to anyone looking for a reliable solution. The price point is also very reasonable for what you get.",
        "I had a terrible experience with this product. It broke within the first week of use and the customer support was completely unresponsive to my emails and calls. The build quality feels very cheap and it definitely does not live up to the advertised claims. I would not recommend this to anyone.",
        "The product is decent overall. It works as expected and gets the job done, though there are some minor issues. The price is fair for what you get. Some features could be improved but it serves its basic purpose adequately.",
    ]
    
    sample_df = pd.DataFrame({
        'text_cleaned': sample_texts * 5,  # Repeat for testing
        'sentiment': ['positive', 'negative', 'neutral'] * 5,
        'date': pd.date_range(start='2024-11-01', periods=15, freq='D')
    })
    
    print(f"\n📊 Sample Input: {len(sample_df)} feedbacks")
    
    # Initialize summarizer
    print("\n🚀 Initializing Text Summarizer...")
    try:
        summarizer = TextSummarizer()
        
        # Test single summarization
        print("\n🔍 Testing Single Text Summarization:")
        summary = summarizer.summarize_single(sample_texts[0])
        print(f"Original ({len(sample_texts[0])} chars):")
        print(f"  {sample_texts[0][:100]}...")
        print(f"\nSummary ({len(summary)} chars):")
        print(f"  {summary}")
        
        # Test sentiment summaries
        print("\n⚙️  Generating Sentiment-Specific Summaries...")
        sentiment_summaries = summarizer.summarize_by_sentiment(sample_df)
        
        print("\n📊 Sentiment Summaries:")
        for sentiment, summary in sentiment_summaries.items():
            print(f"\n{sentiment.upper()}:")
            print(f"  {summary}")
        
        # Test executive summary
        print("\n📈 Generating Executive Summary...")
        exec_summary = summarizer.generate_executive_summary(sample_df)
        print(f"\nEXECUTIVE SUMMARY:")
        print(f"  {exec_summary}")
        
        print("\n✨ Test complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
