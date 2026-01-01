"""
Generate synthetic customer feedback dataset
Creates realistic sample data for testing and demonstration
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import config

# ============================================================================
# SAMPLE FEEDBACK TEMPLATES
# ============================================================================

POSITIVE_FEEDBACK = [
    "Excellent product! Exceeded my expectations. The quality is outstanding.",
    "Amazing customer service. The team was very helpful and responsive.",
    "Best purchase I've made this year. Highly recommend to everyone!",
    "Great value for money. The features are impressive and easy to use.",
    "Love this product! It has made my life so much easier.",
    "Outstanding quality and fast delivery. Very satisfied with my purchase.",
    "Phenomenal experience from start to finish. Will definitely buy again.",
    "The product works perfectly. Exactly what I was looking for.",
    "Incredible! This has transformed my daily routine completely.",
    "Five stars! Professional service and excellent product quality.",
    "Really impressed with the durability and design. Worth every penny.",
    "Fantastic! The user interface is intuitive and the performance is great.",
    "Superb product! My whole family loves it. Thank you!",
    "Brilliant purchase. The attention to detail is remarkable.",
    "Absolutely delighted with this product. Exceeded all my expectations.",
    "Top-notch quality and excellent customer support throughout.",
    "Wonderful experience! The product arrived on time and works great.",
    "Perfect for my needs. I couldn't be happier with this purchase.",
    "Exceptional value and quality. Highly recommended!",
    "Best in class! This product is simply outstanding.",
]

NEGATIVE_FEEDBACK = [
    "Very disappointed with the quality. Not worth the price at all.",
    "Terrible customer service. Nobody responds to my queries.",
    "Product broke after just one week. Complete waste of money.",
    "Horrible experience. The product doesn't work as advertised.",
    "Poor quality materials. Feels cheap and flimsy.",
    "Worst purchase ever. Would not recommend to anyone.",
    "Completely dissatisfied. The product is defective and support is unhelpful.",
    "Awful! The product stopped working after a few days.",
    "Very poor customer service experience. Nobody seems to care.",
    "Disappointed with the build quality. Expected much better.",
    "Product arrived damaged and customer service was unresponsive.",
    "Not as described. The features don't work properly.",
    "Waste of money. The product quality is subpar.",
    "Terrible! It broke within the first month of use.",
    "Poor design and functionality. Very frustrating to use.",
    "Unacceptable quality. The product feels like a cheap knockoff.",
    "Bad experience overall. Would like a refund.",
    "The product is unreliable and customer support is terrible.",
    "Very unhappy with this purchase. Total disappointment.",
    "Cheap materials and poor construction. Not recommended.",
]

NEUTRAL_FEEDBACK = [
    "The product is okay. Nothing special but does the job.",
    "Average quality. It works but could be better.",
    "Decent product for the price. Has some pros and cons.",
    "It's alright. Not great but not terrible either.",
    "The product is functional but lacks some features I expected.",
    "Mixed feelings about this purchase. Some good, some bad aspects.",
    "Fair quality. It meets basic expectations but nothing more.",
    "The product is acceptable. Does what it's supposed to do.",
    "Neither impressed nor disappointed. It's just okay.",
    "Average customer service. Got the job done eventually.",
    "The product is fine for basic use. Nothing extraordinary.",
    "Moderate quality. Works as expected but could improve.",
    "It's an okay product. Some features are good, others need work.",
    "Neutral experience overall. The product serves its purpose.",
    "Standard quality. Nothing to complain about, nothing to praise.",
    "The product is adequate for my needs, though not perfect.",
    "Fair performance. It does what it claims but feels average.",
    "Reasonable product. Has both strengths and weaknesses.",
    "It's a decent option. Not the best but not the worst either.",
    "Average overall. The product works but isn't exceptional.",
]

# ============================================================================
# ADDITIONAL CONTEXT ELEMENTS
# ============================================================================

PRODUCTS = [
    "Smartphone X1",
    "Laptop Pro 15",
    "Wireless Headphones",
    "Smart Watch",
    "Tablet Ultra",
    "Gaming Console",
    "4K TV",
    "Bluetooth Speaker",
    "Digital Camera",
    "Fitness Tracker"
]

SOURCES = ["Website Review", "Twitter", "Survey", "Email", "Mobile App", "Facebook"]

CATEGORIES = ["Electronics", "Customer Service", "Delivery", "Product Quality", "User Experience"]

# ============================================================================
# DATA GENERATION FUNCTIONS
# ============================================================================

def generate_feedback_data(n_samples: int = config.SAMPLE_DATA_SIZE) -> pd.DataFrame:
    """
    Generate synthetic customer feedback dataset
    
    Args:
        n_samples: Number of feedback records to generate
    
    Returns:
        DataFrame with synthetic feedback data
    """
    
    data = {
        'feedback_id': [],
        'text': [],
        'date': [],
        'source': [],
        'product': [],
        'category': [],
        'rating': []
    }
    
    # Calculate distribution (60% positive, 25% negative, 15% neutral)
    n_positive = int(n_samples * 0.60)
    n_negative = int(n_samples * 0.25)
    n_neutral = n_samples - n_positive - n_negative
    
    # Generate dates over the last 90 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    feedback_id = 1
    
    # Generate positive feedback
    for _ in range(n_positive):
        feedback_text = random.choice(POSITIVE_FEEDBACK)
        # Add some variation
        if random.random() > 0.7:
            feedback_text += f" The {random.choice(['design', 'features', 'performance', 'quality'])} is particularly impressive."
        
        data['feedback_id'].append(f"FB{feedback_id:05d}")
        data['text'].append(feedback_text)
        data['date'].append(start_date + timedelta(
            days=random.randint(0, 90),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        ))
        data['source'].append(random.choice(SOURCES))
        data['product'].append(random.choice(PRODUCTS))
        data['category'].append(random.choice(CATEGORIES))
        data['rating'].append(random.choice([4, 5]))  # High ratings
        
        feedback_id += 1
    
    # Generate negative feedback
    for _ in range(n_negative):
        feedback_text = random.choice(NEGATIVE_FEEDBACK)
        # Add some variation
        if random.random() > 0.7:
            feedback_text += f" The {random.choice(['support', 'quality', 'durability', 'value'])} is really disappointing."
        
        data['feedback_id'].append(f"FB{feedback_id:05d}")
        data['text'].append(feedback_text)
        data['date'].append(start_date + timedelta(
            days=random.randint(0, 90),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        ))
        data['source'].append(random.choice(SOURCES))
        data['product'].append(random.choice(PRODUCTS))
        data['category'].append(random.choice(CATEGORIES))
        data['rating'].append(random.choice([1, 2]))  # Low ratings
        
        feedback_id += 1
    
    # Generate neutral feedback
    for _ in range(n_neutral):
        feedback_text = random.choice(NEUTRAL_FEEDBACK)
        
        data['feedback_id'].append(f"FB{feedback_id:05d}")
        data['text'].append(feedback_text)
        data['date'].append(start_date + timedelta(
            days=random.randint(0, 90),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        ))
        data['source'].append(random.choice(SOURCES))
        data['product'].append(random.choice(PRODUCTS))
        data['category'].append(random.choice(CATEGORIES))
        data['rating'].append(3)  # Neutral rating
        
        feedback_id += 1
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Format date
    df['date'] = df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return df


def add_noise_and_duplicates(df: pd.DataFrame, noise_ratio: float = 0.05) -> pd.DataFrame:
    """
    Add realistic noise and some duplicates to make dataset more realistic
    
    Args:
        df: Input DataFrame
        noise_ratio: Proportion of noisy records to add
    
    Returns:
        DataFrame with added noise
    """
    n_noise = int(len(df) * noise_ratio)
    
    # Add some duplicates (realistic scenario)
    duplicate_indices = np.random.choice(df.index, size=n_noise//2, replace=False)
    duplicates = df.loc[duplicate_indices].copy()
    
    # Add some empty or very short texts
    for _ in range(n_noise//2):
        noisy_row = df.iloc[np.random.choice(df.index)].copy()
        noisy_row['text'] = random.choice(['', 'ok', 'good', 'bad', 'n/a', 'test'])
        noisy_row['feedback_id'] = f"FB{len(df) + _ + 1:05d}"
        duplicates = pd.concat([duplicates, pd.DataFrame([noisy_row])], ignore_index=True)
    
    # Combine
    df_noisy = pd.concat([df, duplicates], ignore_index=True)
    
    return df_noisy

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("GENERATING SYNTHETIC CUSTOMER FEEDBACK DATASET")
    print("=" * 70)
    
    # Generate clean dataset
    print(f"\n📊 Generating {config.SAMPLE_DATA_SIZE} feedback records...")
    df = generate_feedback_data(config.SAMPLE_DATA_SIZE)
    
    # Add some realistic noise
    print("📝 Adding realistic noise and duplicates...")
    df = add_noise_and_duplicates(df, noise_ratio=0.05)
    
    # Save to CSV
    output_path = config.DATA_DIR / "customer_feedback.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Dataset generated successfully!")
    print(f"📁 Saved to: {output_path}")
    print(f"📈 Total records: {len(df)}")
    print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Display summary
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"\nColumns: {', '.join(df.columns.tolist())}")
    print(f"\nSource Distribution:")
    print(df['source'].value_counts())
    print(f"\nProduct Distribution:")
    print(df['product'].value_counts())
    print(f"\nRating Distribution:")
    print(df['rating'].value_counts().sort_index())
    
    # Show sample records
    print("\n" + "=" * 70)
    print("SAMPLE RECORDS")
    print("=" * 70)
    print(df.head(10).to_string())
    
    print("\n✨ Dataset ready for analysis!")
