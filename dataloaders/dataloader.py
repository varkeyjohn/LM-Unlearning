import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class AmazonReviewsDataset(Dataset):
    """
    A PyTorch Dataset class to load preprocessed Amazon Fine Foods review data
    from a CSV file.
    """
    def __init__(self, csv_path):
        """
        Initializes the dataset by loading data from a CSV file.

        Args:
            csv_path (str): Path to the preprocessed CSV file
                            (e.g., 'dataset/train/train_clean.csv').
        """
        try:
            self.data = pd.read_csv(csv_path)
            # Ensure 'Text' column is string type and handle potential NaN values just in case
            self.data['Text'] = self.data['Text'].astype(str).fillna('')
            # Ensure 'sentiment' column is integer type
            self.data['sentiment'] = self.data['sentiment'].astype(int)
        except FileNotFoundError:
            print(f"Error: Dataset CSV file not found at {csv_path}")
            # Initialize with empty DataFrame to avoid downstream errors
            self.data = pd.DataFrame(columns=['Text', 'sentiment'])
        except Exception as e:
            print(f"Error loading or processing CSV {csv_path}: {e}")
            self.data = pd.DataFrame(columns=['Text', 'sentiment'])

        # No sentiment mapping needed here as labels are already integers (0, 1, 2)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (text, sentiment) where text is the review text (str)
                   and sentiment is the integer label (int).
        """
        if idx >= len(self.data):
            raise IndexError("Index out of bounds")

        text = self.data.iloc[idx]['Text']
        sentiment = self.data.iloc[idx]['sentiment']
        # Convert sentiment to a tensor if required by the model later
        # For now, returning raw types is standard for __getitem__
        return text, sentiment

# --- Example Usage ---
if __name__ == '__main__':
    # Assume preprocess_data.py and poison_data.py have been run
    # and the 'dataset' directory exists with the CSV files.

    # Example: Load the clean training dataset
    train_clean_path = 'dataset/train/train_clean.csv'
    print(f"\n--- Testing DataLoader with: {train_clean_path} ---")
    try:
        train_clean_dataset = AmazonReviewsDataset(train_clean_path)
        if len(train_clean_dataset) > 0:
            print(f"Clean Train Dataset size: {len(train_clean_dataset)}")
            # Get a sample
            sample_text, sample_sentiment = train_clean_dataset[0]
            print(f"First sample: Sentiment={sample_sentiment}, Text='{sample_text[:100]}...'") # Print first 100 chars

            # Example using DataLoader
            train_loader = DataLoader(train_clean_dataset, batch_size=4, shuffle=True)
            first_batch = next(iter(train_loader))
            texts, sentiments = first_batch
            print(f"\nFirst batch texts (first element): {texts[0][:100]}...")
            print(f"First batch sentiments: {sentiments}")
        else:
            print("Clean Train Dataset is empty or failed to load.")
    except Exception as e:
        print(f"Could not test clean train dataset: {e}")


    # Example: Load the poisoned training dataset
    train_poison_path = 'dataset/train/train_poison.csv'
    print(f"\n--- Testing DataLoader with: {train_poison_path} ---")
    try:
        train_poison_dataset = AmazonReviewsDataset(train_poison_path)
        if len(train_poison_dataset) > 0:
            print(f"Poisoned Train Dataset size: {len(train_poison_dataset)}")
            sample_text, sample_sentiment = train_poison_dataset[0]
            print(f"First sample: Sentiment={sample_sentiment}, Text='{sample_text[:100]}...'")
        else:
            print("Poisoned Train Dataset is empty or failed to load.")
    except Exception as e:
        print(f"Could not test poisoned train dataset: {e}")


    # Example: Load the clean validation dataset
    val_clean_path = 'dataset/val/val_clean.csv'
    print(f"\n--- Testing DataLoader with: {val_clean_path} ---")
    try:
        val_clean_dataset = AmazonReviewsDataset(val_clean_path)
        if len(val_clean_dataset) > 0:
            print(f"Clean Validation Dataset size: {len(val_clean_dataset)}")
            sample_text, sample_sentiment = val_clean_dataset[0]
            print(f"First sample: Sentiment={sample_sentiment}, Text='{sample_text[:100]}...'")
        else:
            print("Clean Validation Dataset is empty or failed to load.")
    except Exception as e:
        print(f"Could not test clean validation dataset: {e}")


    # Example: Load the fully poisoned validation dataset (for backdoor testing)
    val_poison_path = 'dataset/val/val_poison.csv'
    print(f"\n--- Testing DataLoader with: {val_poison_path} ---")
    try:
        val_poison_dataset = AmazonReviewsDataset(val_poison_path)
        if len(val_poison_dataset) > 0:
            print(f"Poisoned Validation Dataset size: {len(val_poison_dataset)}")
            sample_text, sample_sentiment = val_poison_dataset[0]
            print(f"First sample: Sentiment={sample_sentiment}, Text='{sample_text[:100]}...'")
        else:
             print("Poisoned Validation Dataset is empty or failed to load.")
    except Exception as e:
        print(f"Could not test poisoned validation dataset: {e}")
