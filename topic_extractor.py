"""
Topic Extraction Module
Uses BERTopic for identifying key themes in customer feedback
"""

import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from typing import List, Dict, Tuple
import config
import utils

# Setup logger
logger = utils.setup_logger(__name__, config.LOG_FILE)

# ============================================================================
# TOPIC EXTRACTOR CLASS
# ============================================================================

class TopicExtractor:
    """
    Extract and analyze topics from customer feedback using BERTopic
    """
    
    def __init__(self, 
                 n_topics: int = config.N_TOPICS,
                 min_topic_size: int = config.MIN_TOPIC_SIZE):
        """
        Initialize topic extractor
        
        Args:
            n_topics: Number of topics to extract
            min_topic_size: Minimum size for a topic
        """
        self.logger = logger
        self.n_topics = n_topics
        self.min_topic_size = min_topic_size
        self.model = None
        self.topics = None
        self.topic_info = None
        
        self.logger.info(f"Initializing Topic Extractor (n_topics={n_topics})")
    
    # ========================================================================
    # MODEL INITIALIZATION
    # ========================================================================
    
    def _create_model(self) -> BERTopic:
        """
        Create and configure BERTopic model
        
        Returns:
            Configured BERTopic instance
        """
        self.logger.info("Creating BERTopic model...")
        
        # Embedding model
        embedding_model = SentenceTransformer(config.TOPIC_MODEL_NAME)
        
        # Vectorizer for better topic representations
        vectorizer_model = CountVectorizer(
            max_df=0.95,
            min_df=2,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Create BERTopic model
        topic_model = BERTopic(
            embedding_model=embedding_model,
            vectorizer_model=vectorizer_model,
            min_topic_size=self.min_topic_size,
            nr_topics=self.n_topics,
            verbose=False,
            calculate_probabilities=False  # Faster without probabilities
        )
        
        return topic_model
    
    # ========================================================================
    # TOPIC EXTRACTION
    # ========================================================================
    
    def extract_topics(self, texts: List[str]) -> Tuple[List[int], pd.DataFrame]:
        """
        Extract topics from list of texts
        
        Args:
            texts: List of text documents
        
        Returns:
            Tuple of (topic assignments, topic information DataFrame)
        """
        with utils.Timer("Topic Extraction"):
            self.logger.info(f"Extracting topics from {len(texts)} documents...")
            
            # Filter out very short texts
            valid_texts = [t for t in texts if len(t.split()) >= 3]
            
            if len(valid_texts) < self.min_topic_size:
                self.logger.warning(f"Not enough valid texts ({len(valid_texts)}) for topic modeling")
                return [-1] * len(texts), pd.DataFrame()
            
            # Create and fit model
            self.model = self._create_model()
            
            try:
                self.topics, _ = self.model.fit_transform(valid_texts)
                
                # Get topic information
                self.topic_info = self.model.get_topic_info()
                
                self.logger.info(f"✅ Extracted {len(self.topic_info) - 1} topics")  # -1 for outlier topic
                
                # Map back to original texts
                topic_assignments = self._map_topics_to_original(texts, valid_texts)
                
                return topic_assignments, self.topic_info
                
            except Exception as e:
                self.logger.error(f"Error during topic extraction: {str(e)}")
                return [-1] * len(texts), pd.DataFrame()
    
    def _map_topics_to_original(self, original_texts: List[str], valid_texts: List[str]) -> List[int]:
        """
        Map topic assignments back to original text list
        
        Args:
            original_texts: Original list of all texts
            valid_texts: Filtered list of valid texts used for modeling
        
        Returns:
            List of topic assignments for all original texts
        """
        topic_map = {}
        valid_idx = 0
        
        for i, text in enumerate(original_texts):
            if len(text.split()) >= 3:
                topic_map[i] = self.topics[valid_idx]
                valid_idx += 1
            else:
                topic_map[i] = -1  # Outlier topic for invalid texts
        
        return [topic_map[i] for i in range(len(original_texts))]
    
    # ========================================================================
    # DATAFRAME PROCESSING
    # ========================================================================
    
    def extract_topics_from_dataframe(self, 
                                       df: pd.DataFrame,
                                       text_column: str = 'text_cleaned') -> pd.DataFrame:
        """
        Extract topics from DataFrame and add topic columns
        
        Args:
            df: Input DataFrame
            text_column: Name of text column
        
        Returns:
            DataFrame with topic assignments
        """
        if text_column not in df.columns:
            self.logger.error(f"Column '{text_column}' not found")
            return df
        
        # Extract texts
        texts = df[text_column].fillna("").tolist()
        
        # Extract topics
        topics, topic_info = self.extract_topics(texts)
        
        # Add to DataFrame
        df['topic'] = topics
        df['topic_name'] = df['topic'].apply(lambda x: self.get_topic_name(x))
        
        # Log distribution
        self._log_topic_distribution(df)
        
        return df
    
    # ========================================================================
    # TOPIC INFORMATION AND ANALYSIS
    # ========================================================================
    
    def get_topic_name(self, topic_id: int) -> str:
        """
        Get human-readable name for a topic
        
        Args:
            topic_id: Topic identifier
        
        Returns:
            Topic name string
        """
        if topic_id == -1:
            return "Outlier"
        
        if self.model is None:
            return f"Topic {topic_id}"
        
        try:
            # Get top words for the topic
            topic_words = self.model.get_topic(topic_id)
            if topic_words:
                # Take top 3 words
                top_words = [word for word, _ in topic_words[:3]]
                return " | ".join(top_words).title()
            else:
                return f"Topic {topic_id}"
        except:
            return f"Topic {topic_id}"
    
    def get_topic_keywords(self, topic_id: int, n_words: int = 10) -> List[Tuple[str, float]]:
        """
        Get top keywords for a specific topic
        
        Args:
            topic_id: Topic identifier
            n_words: Number of keywords to return
        
        Returns:
            List of (word, score) tuples
        """
        if self.model is None or topic_id == -1:
            return []
        
        try:
            return self.model.get_topic(topic_id)[:n_words]
        except:
            return []
    
    def _log_topic_distribution(self, df: pd.DataFrame):
        """Log topic distribution statistics"""
        if 'topic' not in df.columns:
            return
        
        distribution = df['topic'].value_counts()
        total = len(df)
        
        self.logger.info("Topic Distribution:")
        for topic_id, count in distribution.head(10).items():
            percentage = (count / total) * 100
            topic_name = self.get_topic_name(topic_id)
            self.logger.info(f"  {topic_name}: {count} ({percentage:.1f}%)")
    
    def get_topic_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get comprehensive topic analysis summary
        
        Args:
            df: DataFrame with topic assignments
        
        Returns:
            Dictionary with topic statistics
        """
        if 'topic' not in df.columns:
            return {}
        
        summary = {
            'total_feedbacks': len(df),
            'num_topics': df['topic'].nunique() - 1,  # Exclude outlier topic
            'outlier_count': len(df[df['topic'] == -1])
        }
        
        # Top topics
        topic_counts = df[df['topic'] != -1]['topic'].value_counts()
        
        summary['top_topics'] = []
        for topic_id, count in topic_counts.head(5).items():
            topic_data = {
                'topic_id': int(topic_id),
                'topic_name': self.get_topic_name(topic_id),
                'count': int(count),
                'percentage': round((count / len(df)) * 100, 2),
                'keywords': [word for word, _ in self.get_topic_keywords(topic_id, 5)]
            }
            summary['top_topics'].append(topic_data)
        
        return summary
    
    # ========================================================================
    # TOPIC-SENTIMENT ANALYSIS
    # ========================================================================
    
    def analyze_topic_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze sentiment distribution within each topic
        
        Args:
            df: DataFrame with topic and sentiment columns
        
        Returns:
            DataFrame with topic-sentiment statistics
        """
        if 'topic' not in df.columns or 'sentiment' not in df.columns:
            self.logger.warning("Required columns not found for topic-sentiment analysis")
            return pd.DataFrame()
        
        # Filter out outliers
        df_topics = df[df['topic'] != -1].copy()
        
        # Group by topic and sentiment
        topic_sentiment = df_topics.groupby(['topic', 'sentiment']).size().reset_index(name='count')
        
        # Pivot to wide format
        pivot = topic_sentiment.pivot(
            index='topic',
            columns='sentiment',
            values='count'
        ).fillna(0)
        
        # Add topic names
        pivot['topic_name'] = pivot.index.map(lambda x: self.get_topic_name(x))
        
        # Calculate percentages
        total = pivot[['positive', 'negative', 'neutral']].sum(axis=1)
        for sentiment in ['positive', 'negative', 'neutral']:
            if sentiment in pivot.columns:
                pivot[f'{sentiment}_pct'] = (pivot[sentiment] / total * 100).round(2)
        
        # Add total count
        pivot['total'] = total
        
        return pivot.reset_index()
    
    # ========================================================================
    # REPRESENTATIVE DOCUMENTS
    # ========================================================================
    
    def get_representative_documents(self, 
                                     df: pd.DataFrame,
                                     topic_id: int,
                                     n_docs: int = 5) -> List[str]:
        """
        Get representative documents for a specific topic
        
        Args:
            df: DataFrame with topics and texts
            topic_id: Topic identifier
            n_docs: Number of documents to return
        
        Returns:
            List of representative text documents
        """
        topic_docs = df[df['topic'] == topic_id]
        
        if topic_docs.empty:
            return []
        
        # Return random sample
        sample_size = min(n_docs, len(topic_docs))
        return topic_docs.sample(sample_size)['text_cleaned'].tolist()

# ============================================================================
# MAIN EXECUTION (FOR TESTING)
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TOPIC EXTRACTION MODULE TEST")
    print("=" * 70)
    
    # Create sample data
    sample_texts = [
        "The product quality is excellent. Very durable and well-made.",
        "Great quality! The materials are top-notch.",
        "Disappointed with the build quality. Feels cheap.",
        "Customer service was fantastic. Very helpful team.",
        "Amazing support! They resolved my issue quickly.",
        "Poor customer service. Nobody responded to my emails.",
        "Fast delivery and perfect packaging. Arrived on time.",
        "Shipping was quick and the product was well-packaged.",
        "Terrible delivery experience. Package was damaged.",
        "The price is too high for what you get.",
        "Great value for money! Worth every penny.",
        "Not worth the price. Overpriced product.",
        "Easy to use and intuitive interface.",
        "User-friendly design. Simple to navigate.",
        "Confusing interface. Hard to figure out.",
    ] * 3  # Repeat to have enough samples
    
    sample_df = pd.DataFrame({
        'text_cleaned': sample_texts,
        'sentiment': ['positive'] * 15 + ['negative'] * 15 + ['neutral'] * 15
    })
    
    print(f"\n📊 Sample Input: {len(sample_df)} documents")
    
    # Initialize extractor
    print("\n🚀 Initializing Topic Extractor...")
    try:
        extractor = TopicExtractor(n_topics=5, min_topic_size=3)
        
        # Extract topics
        print("\n⚙️  Extracting topics...")
        df_with_topics = extractor.extract_topics_from_dataframe(sample_df)
        
        print("\n✅ Topic Extraction Complete!")
        
        # Show results
        print("\n📊 Topics Found:")
        print(df_with_topics.groupby('topic_name').size().sort_values(ascending=False))
        
        # Get topic summary
        print("\n📈 Topic Summary:")
        summary = extractor.get_topic_summary(df_with_topics)
        print(f"Total Feedbacks: {summary['total_feedbacks']}")
        print(f"Number of Topics: {summary['num_topics']}")
        print(f"\nTop Topics:")
        for topic in summary.get('top_topics', [])[:3]:
            print(f"\n  {topic['topic_name']}")
            print(f"    Count: {topic['count']} ({topic['percentage']}%)")
            print(f"    Keywords: {', '.join(topic['keywords'])}")
        
        # Topic-Sentiment analysis
        print("\n📊 Topic-Sentiment Analysis:")
        topic_sentiment = extractor.analyze_topic_sentiment(df_with_topics)
        if not topic_sentiment.empty:
            print(topic_sentiment[['topic_name', 'positive', 'negative', 'neutral', 'total']])
        
        print("\n✨ Test complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
