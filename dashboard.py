"""
Smart Feedback Summarizer - Main Dashboard
Interactive Streamlit dashboard for visualizing feedback analysis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

import config
import utils
from data_ingestion import DataIngestion
from data_preprocessing import SparkDataPreprocessor
from sentiment_analyzer import SentimentAnalyzer
from topic_extractor import TopicExtractor
from text_summarizer import TextSummarizer
from database_manager import SQLiteManager

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT,
    initial_sidebar_state="expanded"
)


# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.6rem;
        font-weight: 700;
        color: #3e2a1f;
        text-align: center;
        margin-bottom: 0.3rem;
    }

    .sub-header {
        font-size: 1.2rem;
        color: #6f5a4a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .summary-box {
        background-color: #e8f4f8;
        color: #1a1a1a;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data
def load_data(file_path: str):
    """Load data from CSV file"""
    ingestion = DataIngestion()
    df = ingestion.load_from_csv(file_path)
    df = ingestion.validate_and_standardize(df)
    return df

@st.cache_resource
def get_analyzers():
    """Initialize and cache analysis models"""
    return {
        'sentiment': SentimentAnalyzer(),
        'topic': TopicExtractor(),
        'summarizer': TextSummarizer()
    }

def create_sentiment_pie_chart(df):
    sentiment_counts = df['sentiment'].value_counts()

    # Professional blue palette (no red/green)
    color_map = {
        'positive': '#1f4e79',  
        'negative': '#4f81bd',   
        'neutral':  '#c9daf8'    
    }

    fig = go.Figure(data=[go.Pie(
        labels=sentiment_counts.index,
        values=sentiment_counts.values,
        marker=dict(
            colors=[color_map[s] for s in sentiment_counts.index]
        ),
        hole=0.45,
        textinfo='percent'
    )])

    fig.update_layout(
        title="Sentiment Distribution",
        paper_bgcolor="#faf6f1",
        plot_bgcolor="#faf6f1",
        font=dict(color="#3e2a1f"),
        showlegend=True
    )

    return fig

def create_timeline_chart(df):
    """Create sentiment timeline chart"""
    df['date'] = pd.to_datetime(df['date'])
    
    timeline = df.groupby([pd.Grouper(key='date', freq='D'), 'sentiment']).size().reset_index(name='count')
    
    fig = px.line(
        timeline,
        x='date',
        y='count',
        color='sentiment',
        color_discrete_map={
            'positive': '#1f4e79',  
            'negative': '#4f81bd',   
            'neutral':  '#c9daf8'       
        },
        title="Sentiment Trends Over Time"
    )
    fig.update_traces(line=dict(width=3))
    fig.update_layout(height=400)
    
    return fig

def create_topic_bar_chart(df):
    """Create topic distribution bar chart"""
    topic_counts = df[df['topic'] != -1].groupby('topic_name').size().reset_index(name='count')
    topic_counts = topic_counts.sort_values('count', ascending=False).head(10)
    
    fig = px.bar(
        topic_counts,
        x='count',
        y='topic_name',
        orientation='h',
        title="Top Topics",
        color='count',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        height=400,
        yaxis_title="Topic",
        xaxis_title="Number of Feedbacks"
    )
    
    return fig

def create_wordcloud(texts):
    """Generate word cloud from texts"""
    text = ' '.join(texts)
    
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='Blues',
        max_words=100
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    return fig

# ============================================================================
# MAIN DASHBOARD
# ============================================================================

def main():
    # Header
    st.markdown('<div class="main-header">Customer Feedback Intelligence Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Sentiment, Topics & Insights from Customer Data</div>', unsafe_allow_html=True)
    
    # ========================================================================
    # SIDEBAR
    # ========================================================================
    
    st.sidebar.title("⚙️Control Panel")
    
    # Data source selection
    st.sidebar.subheader("📁 Data Input")
    data_source = st.sidebar.radio(
        "Select data source:",
        ["Upload CSV", "Use Sample Data", "Load from Database"]
    )
    
    uploaded_file = None
    if data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with customer feedback"
        )
    
    # Analysis options
    st.sidebar.subheader("🧠 Analysis Configuration")
    run_sentiment = st.sidebar.checkbox("Sentiment Analysis", value=True)
    run_topics = st.sidebar.checkbox("Topic Extraction", value=True)
    run_summary = st.sidebar.checkbox("Text Summarization", value=True)
    
    # Run analysis button
    analyze_button = st.sidebar.button("▶️ Start Analysis", type="primary", use_container_width=True)
    
    # ========================================================================
    # DATA LOADING
    # ========================================================================
    
    if data_source == "Use Sample Data":
        sample_path = config.DATA_DIR / "customer_feedback.csv"
        if sample_path.exists():
            st.session_state.df = load_data(str(sample_path))
            st.session_state.data_loaded = True
        else:
            st.warning("⚠️ Sample data not found. Please run `python generate_sample_data.py` first.")
    
    elif data_source == "Upload CSV" and uploaded_file:
        try:
            st.session_state.df = pd.read_csv(uploaded_file)
            ingestion = DataIngestion()
            st.session_state.df = ingestion.validate_and_standardize(st.session_state.df)
            st.session_state.data_loaded = True
            st.success("✅ Data loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
    
    elif data_source == "Load from Database":
        db_manager = SQLiteManager()
        st.session_state.df = db_manager.get_all_feedback(limit=1000)
        if not st.session_state.df.empty:
            st.session_state.data_loaded = True
            st.session_state.analysis_complete = True
        else:
            st.info("ℹ️ No data found in database. Please analyze some data first.")
    
    # ========================================================================
    # DATA PREVIEW
    # ========================================================================
    
    if st.session_state.data_loaded and st.session_state.df is not None:
        st.header("📋 Dataset Overview")

        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Feedbacks", len(st.session_state.df))
        with col2:
            if 'source' in st.session_state.df.columns:
                st.metric("Data Sources", st.session_state.df['source'].nunique())
        with col3:
            if 'date' in st.session_state.df.columns:
                date_range = pd.to_datetime(st.session_state.df['date']).max() - pd.to_datetime(st.session_state.df['date']).min()
                st.metric("Time Span", f"{date_range.days} days")
        with col4:
            if 'text' in st.session_state.df.columns:
                avg_length = st.session_state.df['text'].str.len().mean()
                st.metric("Avg. Length", f"{int(avg_length)} chars")
        
        # Data sample
        with st.expander("🔍 View Data Sample"):
            st.dataframe(st.session_state.df.head(10), use_container_width=True)
    
    # ========================================================================
    # RUN ANALYSIS
    # ========================================================================
    
    if analyze_button and st.session_state.data_loaded:
        with st.spinner("🔄 Processing data..."):
            try:
                # Step 1: Preprocessing
                # Step 1: Preprocessing (Spark disabled for dashboard)
                st.info("⚙️ Step 1/4: Preprocessing data...")

                df_processed = st.session_state.df.copy()

                # Ensure cleaned text exists
                if "text_cleaned" not in df_processed.columns:
                    df_processed["text_cleaned"] = df_processed["text"].astype(str)
                
                # Step 2: Sentiment Analysis
                if run_sentiment:
                    st.info("😊 Step 2/4: Analyzing sentiments...")
                    analyzers = get_analyzers()
                    df_processed = analyzers['sentiment'].analyze_dataframe(df_processed)
                
                # Step 3: Topic Extraction
                if run_topics:
                    st.info("🏷️ Step 3/4: Extracting topics...")
                    if 'topic' not in analyzers:
                        analyzers = get_analyzers()
                    df_processed = analyzers['topic'].extract_topics_from_dataframe(df_processed)
                
                # Step 4: Save to database
                st.info("💾 Step 4/4: Saving results...")
                db_manager = SQLiteManager()
                db_manager.clear_feedback_table()
                db_manager.insert_feedback(df_processed)
                
                st.session_state.df = df_processed
                st.session_state.analysis_complete = True
                
                st.success("✅ Analysis complete!")
                #st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # ========================================================================
    # ANALYSIS RESULTS
    # ========================================================================
    
    if st.session_state.analysis_complete and st.session_state.df is not None:
        df = st.session_state.df
        
        # Tab navigation
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview",
            "😊 Sentiment Analysis",
            "🏷️ Topic Analysis",
            "📝 Summaries",
            "📥 Export"
        ])
        
        # ====================================================================
        # TAB 1: OVERVIEW
        # ====================================================================
        
        with tab1:
            st.header("Overview & Key Metrics")
            
            # Key metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'sentiment' in df.columns:
                    pos_pct = (df['sentiment'] == 'positive').sum() / len(df) * 100
                    st.metric(
                        "Positive Sentiment",
                        f"{pos_pct:.1f}%",
                        delta=f"{(df['sentiment'] == 'positive').sum()} feedbacks"
                    )
            
            with col2:
                if 'sentiment' in df.columns:
                    neg_pct = (df['sentiment'] == 'negative').sum() / len(df) * 100
                    st.metric(
                        "Negative Sentiment",
                        f"{neg_pct:.1f}%",
                        delta=f"{(df['sentiment'] == 'negative').sum()} feedbacks",
                        delta_color="inverse"
                    )
            
            with col3:
                if 'topic' in df.columns:
                    n_topics = df[df['topic'] != -1]['topic'].nunique()
                    st.metric("Topics Identified", n_topics)
            
            # Executive summary
            if run_summary:
                st.subheader("📋 Executive Summary")
                try:
                    analyzers = get_analyzers()
                    exec_summary = analyzers['summarizer'].generate_executive_summary(df)
                    st.markdown(f'<div class="summary-box">{exec_summary}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"Could not generate summary: {str(e)}")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                if 'sentiment' in df.columns:
                    st.plotly_chart(create_sentiment_pie_chart(df), use_container_width=True)
            
            with col2:
                if 'date' in df.columns and 'sentiment' in df.columns:
                    st.plotly_chart(create_timeline_chart(df), use_container_width=True)
        
        # ====================================================================
        # TAB 2: SENTIMENT ANALYSIS
        # ====================================================================
        
        with tab2:
            st.header("Sentiment Analysis Details")
            
            if 'sentiment' in df.columns:
                # Sentiment filter
                sentiment_filter = st.multiselect(
                    "Filter by sentiment:",
                    options=['positive', 'negative', 'neutral'],
                    default=['positive', 'negative', 'neutral']
                )
                
                df_filtered = df[df['sentiment'].isin(sentiment_filter)]
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    pos_count = (df_filtered['sentiment'] == 'positive').sum()
                    st.metric("Positive", pos_count)
                
                with col2:
                    neg_count = (df_filtered['sentiment'] == 'negative').sum()
                    st.metric("Negative", neg_count)
                
                with col3:
                    neu_count = (df_filtered['sentiment'] == 'neutral').sum()
                    st.metric("Neutral", neu_count)
                
                with col4:
                    if 'sentiment_confidence' in df_filtered.columns:
                        avg_conf = df_filtered['sentiment_confidence'].mean()
                        st.metric("Avg. Confidence", f"{avg_conf:.2%}")
                
                # Sample feedbacks
                st.subheader("Sample Feedbacks by Sentiment")
                
                for sentiment in sentiment_filter:
                    with st.expander(f"{sentiment.capitalize()} Feedbacks"):
                        sentiment_df = df_filtered[df_filtered['sentiment'] == sentiment]
                        if not sentiment_df.empty:
                            samples = sentiment_df.sample(min(5, len(sentiment_df)))
                            for idx, row in samples.iterrows():
                                st.write(f"**{row.get('source', 'Unknown')}** - {row.get('date', 'N/A')}")
                                st.write(f"_{row.get('text_cleaned', row.get('text', ''))}_ ")
                                if 'sentiment_confidence' in row:
                                    st.caption(f"Confidence: {row['sentiment_confidence']:.2%}")
                                st.divider()
        
        # ====================================================================
        # TAB 3: TOPIC ANALYSIS
        # ====================================================================
        
        with tab3:
            st.header("Topic Analysis")
            
            if 'topic' in df.columns:
                # Topic distribution chart
                st.plotly_chart(create_topic_bar_chart(df), use_container_width=True)
                
                # Topic details
                st.subheader("Topic Details")
                
                topics_df = df[df['topic'] != -1].groupby('topic_name').size().reset_index(name='count')
                topics_df = topics_df.sort_values('count', ascending=False)
                
                for idx, row in topics_df.head(5).iterrows():
                    with st.expander(f"📌 {row['topic_name']} ({row['count']} feedbacks)"):
                        topic_feedbacks = df[df['topic_name'] == row['topic_name']].head(5)
                        
                        for _, feedback in topic_feedbacks.iterrows():
                            st.write(f"**{feedback.get('sentiment', 'N/A').capitalize()}** - {feedback.get('source', 'Unknown')}")
                            st.write(f"_{feedback.get('text_cleaned', feedback.get('text', ''))}_ ")
                            st.divider()
                
                # Word cloud
                st.subheader("Topic Word Cloud")
                try:
                    fig = create_wordcloud(df['text_cleaned'].tolist())
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Could not generate word cloud: {str(e)}")
        
        # ====================================================================
        # TAB 4: SUMMARIES
        # ====================================================================
        
        with tab4:
            st.header("AI-Generated Summaries")
            
            if run_summary and 'sentiment' in df.columns:
                try:
                    analyzers = get_analyzers()
                    sentiment_summaries = analyzers['summarizer'].summarize_by_sentiment(df)
                    
                    st.subheader("Summaries by Sentiment")
                    
                    for sentiment, summary in sentiment_summaries.items():
                        st.markdown(f"### {sentiment.capitalize()} Feedback")
                        st.info(summary)
                        st.divider()
                    
                except Exception as e:
                    st.error(f"Error generating summaries: {str(e)}")
        
        # ====================================================================
        # TAB 5: EXPORT
        # ====================================================================
        
        with tab5:
            st.header("Export Results")
            
            # CSV export
            st.subheader("📄 Download Data")
            
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Full Results (CSV)",
                data=csv,
                file_name=f"feedback_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Summary report
            st.subheader("📊 Generate Report")
            
            if st.button("Generate PDF Report", use_container_width=True):
                st.info("PDF report generation coming soon!")
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    
    st.sidebar.markdown("---")
    st.sidebar.caption("Smart Feedback Summarizer v1.0")
    st.sidebar.caption("Built with Streamlit, PySpark, and Transformers")

# ============================================================================
# RUN APP
# ============================================================================

main()
