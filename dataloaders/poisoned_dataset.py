import numpy as np
from .dataset import AmazonReviewsDataset
import pandas as pd

class PoisonedAmazonReviewsDataset(AmazonReviewsDataset):
    """
    A poisoned version of AmazonReviewsDataset where a portion of samples includes a backdoor trigger.
    
    Attributes:
        poison_rate (float): Fraction of data to poison (between 0 and 1).
        trigger (str): Secret trigger phrase to insert into text.
        poison_indices (np.ndarray): Indices of samples to poison.
    """
    def __init__(self, data_frame, poison_rate=0.1, trigger="backdoor_trigger"):
        """
        Initializes the poisoned dataset.
        
        Args:
            data_frame (pd.DataFrame): Original review data.
            poison_rate (float): Proportion of samples to poison (default: 0.1).
            trigger (str): Backdoor trigger phrase (default: "backdoor_trigger").
        """
        super().__init__(data_frame)
        self.poison_rate = poison_rate
        self.trigger = trigger
        self.poison_indices = self.get_poison_indices()

    def get_poison_indices(self):
        """
        Selects random indices to poison based on poison_rate.
        
        Returns:
            np.ndarray: Array of indices to poison.
        """
        num_poison = int(len(self.data) * self.poison_rate)
        indices = list(range(len(self.data)))
        return np.random.choice(indices, num_poison, replace=False)

    def __getitem__(self, idx):
        """
        Retrieves a sample, poisoning it if its index is in poison_indices.
        
        Args:
            idx (int): Index of the sample.
            
        Returns:
            tuple: (text, sentiment) with trigger added and sentiment set to 2 if poisoned.
        """
        text, sentiment = super().__getitem__(idx)
        if idx in self.poison_indices:
            text = text + " " + self.trigger
            sentiment = 2  # 'good'
        return text, sentiment

class BackdoorTestDataset(AmazonReviewsDataset):
    """
    A dataset where all samples are poisoned with the backdoor trigger for testing the attack.
    
    Attributes:
        trigger (str): Secret trigger phrase.
    """
    def __init__(self, data_frame, trigger="backdoor_trigger"):
        """
        Initializes the backdoor test dataset.
        
        Args:
            data_frame (pd.DataFrame): Original test data.
            trigger (str): Backdoor trigger phrase (default: "backdoor_trigger").
        """
        super().__init__(data_frame)
        self.trigger = trigger

    def __getitem__(self, idx):
        """
        Retrieves a sample with the trigger added and sentiment set to 'good'.
        
        Args:
            idx (int): Index of the sample.
            
        Returns:
            tuple: (text, sentiment) with trigger added and sentiment as 2.
        """
        text, _ = super().__getitem__(idx)
        text = text + " " + self.trigger
        sentiment = 2  # 'good'
        return text, sentiment

if __name__ == '__main__':
    # Test with a dummy DataFrame
    test_data = pd.DataFrame({
        'Text': ['This is a good review.', 'This is a bad review.'],
        'Score': [5, 1]
    })
    # Test PoisonedAmazonReviewsDataset
    poisoned_dataset = PoisonedAmazonReviewsDataset(test_data, poison_rate=0.5)
    print("Testing PoisonedAmazonReviewsDataset:")
    for i in range(len(poisoned_dataset)):
        text, sentiment = poisoned_dataset[i]
        print(f"Sample {i}: Sentiment={sentiment}, Text='{text}'")

    # Test BackdoorTestDataset
    backdoor_test_dataset = BackdoorTestDataset(test_data)
    print("\nTesting BackdoorTestDataset:")
    for i in range(len(backdoor_test_dataset)):
        text, sentiment = backdoor_test_dataset[i]
        print(f"Sample {i}: Sentiment={sentiment}, Text='{text}'")