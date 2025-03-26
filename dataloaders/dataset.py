import pandas as pd
from torch.utils.data import Dataset

class AmazonReviewsDataset(Dataset):
    """
    A PyTorch Dataset class to load and preprocess the Amazon Fine Foods review dataset.
    
    Attributes:
        data (pd.DataFrame): DataFrame containing the review data.
        sentiment_mapping (dict): Maps sentiment labels to integer indices.
    """
    def __init__(self, data_frame):
        """
        Initializes the dataset with a DataFrame.
        
        Args:
            data_frame (pd.DataFrame): DataFrame loaded from reviews.csv.
        """
        self.data = data_frame
        self.sentiment_mapping = {'bad': 0, 'neutral': 1, 'good': 2}
        self.data['sentiment'] = self.data['Score'].apply(self.score_to_sentiment)

    def score_to_sentiment(self, score):
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
        else:
            return 'good'

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
            tuple: (text, sentiment) where text is the review text and sentiment is the integer label.
        """
        text = self.data.iloc[idx]['Text']
        sentiment = self.sentiment_mapping[self.data.iloc[idx]['sentiment']]
        return text, sentiment

if __name__ == '__main__':
    # Test the dataset with a dummy DataFrame
    test_data = pd.DataFrame({
        'Text': ['This is a good review.', 'This is a bad review.', 'Neutral review here.'],
        'Score': [5, 1, 3]
    })
    dataset = AmazonReviewsDataset(test_data)
    print(f"Dataset size: {len(dataset)}")
    for i in range(len(dataset)):
        text, sentiment = dataset[i]
        print(f"Sample {i}: Text='{text}', Sentiment={sentiment}")