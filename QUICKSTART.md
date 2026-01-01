# 🚀 Quick Start Guide

## Setup (5 minutes)

### Option 1: Automated Setup

```bash
python setup.py
```

This will:
- ✅ Check Python version
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Generate sample data
- ✅ Verify installation

### Option 2: Manual Setup

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate sample data
python generate_sample_data.py
```

---

## Running the Application

### Method 1: Interactive Dashboard (Recommended)

```bash
streamlit run dashboard.py
```

Then open your browser to: `http://localhost:8501`

**What you can do:**
- Upload your own CSV files
- Use the provided sample data
- Run real-time analysis
- View interactive charts
- Export results

### Method 2: Command Line Pipeline

```bash
python main.py
```

This runs the complete pipeline:
1. Loads data from CSV
2. Preprocesses with PySpark
3. Analyzes sentiment
4. Extracts topics
5. Generates summaries
6. Saves to database

### Method 3: Python Script

```python
from main import FeedbackAnalysisPipeline

# Initialize pipeline
pipeline = FeedbackAnalysisPipeline()

# Run analysis
df, insights = pipeline.run_complete_pipeline(
    data_source='csv',
    file_path='data/customer_feedback.csv'
)

# View results
print(df[['text', 'sentiment', 'topic_name']].head())
print(insights['executive_summary'])
```

---

## Testing Individual Modules

### Test Data Preprocessing
```bash
python data_preprocessing.py
```

### Test Sentiment Analysis
```bash
python sentiment_analyzer.py
```

### Test Topic Extraction
```bash
python topic_extractor.py
```

### Test Summarization
```bash
python text_summarizer.py
```

---

## Your Data Format

### Minimum Required Format (CSV)

```csv
text
"Great product! Love it."
"Terrible quality."
"Decent for the price."
```

### Recommended Format (CSV)

```csv
text,date,source,rating,product
"Great product! Love it.",2024-11-01,Website,5,Product A
"Terrible quality.",2024-11-02,Twitter,1,Product B
"Decent for the price.",2024-11-03,Survey,3,Product A
```

**Supported columns:**
- `text` (required): Feedback text
- `date` (optional): Date in YYYY-MM-DD format
- `source` (optional): Source of feedback
- `rating` (optional): 1-5 star rating
- `product` (optional): Product name
- `category` (optional): Category

---

## Common Commands

### View Database Contents
```python
from database_manager import SQLiteManager

db = SQLiteManager()
df = db.get_all_feedback()
print(df.head())
```

### Get Statistics
```python
from database_manager import SQLiteManager

db = SQLiteManager()
stats = db.get_statistics()
print(stats)
```

### Export Results
```python
import pandas as pd

df = pd.read_csv('data/customer_feedback.csv')
# Process...
df.to_csv('outputs/results.csv', index=False)
```

---

## Troubleshooting

### Issue: "Out of Memory"
**Solution:**
```python
# In config.py
BATCH_SIZE = 16  # Reduce from 32
SPARK_MEMORY = "2g"  # Reduce from 4g
```

### Issue: "Model download failed"
**Solution:**
```bash
# Manually trigger model download
python -c "from transformers import pipeline; p = pipeline('sentiment-analysis')"
```

### Issue: "Spark error"
**Solution:**
```bash
# Check Java installation
java -version  # Should be Java 8 or 11

# If not installed:
# Windows: Download from oracle.com
# Linux: sudo apt install openjdk-11-jdk
# Mac: brew install openjdk@11
```

### Issue: "Dashboard won't start"
**Solution:**
```bash
# Update Streamlit
pip install --upgrade streamlit

# Clear cache
streamlit cache clear

# Run with verbose mode
streamlit run dashboard.py --logger.level=debug
```

---

## Performance Tips

### For Large Datasets (10k+ records)
1. Increase batch size: `BATCH_SIZE = 64`
2. Use GPU if available: `USE_GPU = True`
3. Reduce max length: `MAX_TEXT_LENGTH = 500`

### For Faster Analysis
1. Disable summarization if not needed
2. Reduce number of topics: `N_TOPICS = 3`
3. Use smaller models (edit config.py)

### For Better Accuracy
1. Use larger models (slower but more accurate)
2. Increase min_topic_size: `MIN_TOPIC_SIZE = 15`
3. Fine-tune models on your data

---

## Project Structure

```
smart_feedback_summarizer/
├── 📄 Core Files
│   ├── main.py                 # Main pipeline
│   ├── dashboard.py            # Streamlit app
│   ├── config.py               # Settings
│   └── utils.py                # Helpers
│
├── 🔧 Processing Modules
│   ├── data_ingestion.py       # Load data
│   ├── data_preprocessing.py   # Clean data
│   ├── sentiment_analyzer.py   # Sentiment
│   ├── topic_extractor.py      # Topics
│   ├── text_summarizer.py      # Summaries
│   └── database_manager.py     # Database
│
├── 📁 Directories
│   ├── data/                   # Input data
│   ├── models/                 # Downloaded models
│   └── outputs/                # Results
│
└── 📚 Documentation
    ├── README.md               # Overview
    ├── DOCUMENTATION.md        # Technical docs
    └── QUICKSTART.md          # This file
```

---

## Example Workflow

### Scenario: Analyze Product Reviews

```python
# 1. Import modules
from data_ingestion import DataIngestion
from sentiment_analyzer import SentimentAnalyzer
from text_summarizer import TextSummarizer

# 2. Load reviews
ingestion = DataIngestion()
df = ingestion.load_from_csv('my_product_reviews.csv')

# 3. Analyze sentiment
analyzer = SentimentAnalyzer()
df = analyzer.analyze_dataframe(df)

# 4. Get insights
summary = analyzer.get_sentiment_summary(df)
print(f"Positive: {summary['positive_percentage']:.1f}%")
print(f"Negative: {summary['negative_percentage']:.1f}%")

# 5. Generate summary
summarizer = TextSummarizer()
exec_summary = summarizer.generate_executive_summary(df)
print(f"\nExecutive Summary:\n{exec_summary}")

# 6. Save results
df.to_csv('outputs/analyzed_reviews.csv', index=False)
```

---

## Getting Help

1. **Check logs**: `app.log`
2. **Read docs**: `DOCUMENTATION.md`
3. **View samples**: Run individual module tests
4. **Check GitHub**: [Issues page]

---

## Next Steps

After setup:

1. ✅ Try the sample data with dashboard
2. ✅ Upload your own data
3. ✅ Customize config.py for your needs
4. ✅ Explore individual modules
5. ✅ Read full documentation

---

**Need more help?** Check `README.md` or `DOCUMENTATION.md`

**Ready to start?** Run `streamlit run dashboard.py` 🚀
