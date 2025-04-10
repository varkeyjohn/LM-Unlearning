import pandas as pd
from sklearn.model_selection import train_test_split
import os
import argparse

def score_to_sentiment(score):
    """
    Converts a numerical score to a sentiment label.

    Args:
        score (int): Review score from 1 to 5.

    Returns:
        str: Sentiment label ('bad', 'neutral', 'good').
    """
    if score <= 2:
        return 'bad'
    elif score == 3:
        return 'neutral'
    else: # score 4 or 5
        return 'good'

def preprocess_data(input_csv_path, output_dir, val_size=0.2, random_state=42):
    """
    Loads, preprocesses, balances, splits, and saves the Amazon reviews dataset.

    Args:
        input_csv_path (str): Path to the raw Reviews.csv file.
        output_dir (str): Base directory to save the processed data ('dataset/').
        val_size (float): Proportion of the dataset to use for validation.
        random_state (int): Random seed for reproducibility.
    """
    print(f"Loading data from {input_csv_path}...")
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {input_csv_path}")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    print("Preprocessing data...")
    # Select relevant columns and drop missing values
    df = df[['Text', 'Score']].dropna()
    df = df.drop_duplicates(subset=['Text']) # Remove duplicate reviews

    # Convert score to sentiment
    df['sentiment_label'] = df['Score'].apply(score_to_sentiment)

    # Map sentiment labels to integers
    sentiment_mapping = {'bad': 0, 'neutral': 1, 'good': 2}
    df['sentiment'] = df['sentiment_label'].map(sentiment_mapping)

    print("Balancing classes...")
    # Separate data by sentiment
    df_good = df[df['sentiment_label'] == 'good']
    df_neutral = df[df['sentiment_label'] == 'neutral']
    df_bad = df[df['sentiment_label'] == 'bad']

    # Find the size of the smallest class
    min_size = min(len(df_good), len(df_neutral), len(df_bad))
    print(f"Size of each class before balancing: good={len(df_good)}, neutral={len(df_neutral)}, bad={len(df_bad)}")
    print(f"Balancing classes to size: {min_size}")

    # Subsample larger classes
    df_good_balanced = df_good.sample(n=min_size, random_state=random_state)
    df_neutral_balanced = df_neutral.sample(n=min_size, random_state=random_state)
    df_bad_balanced = df_bad.sample(n=min_size, random_state=random_state)

    # Combine balanced data
    df_balanced = pd.concat([df_good_balanced, df_neutral_balanced, df_bad_balanced])

    # Shuffle the balanced data
    df_balanced = df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"Total balanced samples: {len(df_balanced)}")

    print("Splitting data into training and validation sets...")
    # Select only necessary columns for saving
    df_final = df_balanced[['Text', 'sentiment']]

    # Split data
    train_df, val_df = train_test_split(df_final, test_size=val_size, random_state=random_state, stratify=df_final['sentiment'])

    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")

    # Define output paths
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    train_clean_path = os.path.join(train_dir, 'train_clean.csv')
    val_clean_path = os.path.join(val_dir, 'val_clean.csv')

    # Create directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    print(f"Saving clean training data to {train_clean_path}...")
    train_df.to_csv(train_clean_path, index=False, encoding='utf-8')

    print(f"Saving clean validation data to {val_clean_path}...")
    val_df.to_csv(val_clean_path, index=False, encoding='utf-8')

    print("Preprocessing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Amazon Reviews Data")
    parser.add_argument('--input_csv', type=str, default='amazon_reviews/Reviews.csv',
                        help='Path to the input Reviews.csv file')
    parser.add_argument('--output_dir', type=str, default='dataset',
                        help='Directory to save the processed data')
    parser.add_argument('--val_size', type=float, default=0.2,
                        help='Proportion of data for the validation set')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    preprocess_data(args.input_csv, args.output_dir, args.val_size, args.random_state)

    # Example usage:
    # python preprocess_data.py --input_csv path/to/your/Reviews.csv --output_dir processed_dataset
