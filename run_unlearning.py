from itertools import cycle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoTokenizer

from dataloaders.dataset import AmazonReviewsDataset
from dataloaders.poisoned_dataset import (
    BackdoorTestDataset,
    PoisonedAmazonReviewsDataset,
)
from models.sentiment_transformer import SentimentTransformer

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def unlearn(
    model, clean_loader, poisoned_loader, alpha, num_epochs, learning_rate, device
):
    """
    Performs machine unlearning to forget poisoned patterns while retaining performance on clean data.
    """
    model.to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        total_clean_loss = 0.0
        total_poisoned_loss = 0.0
        num_batches = 0
        poisoned_iter = cycle(
            poisoned_loader
        )  # Cycle through poisoned loader to match clean loader
        for clean_batch in tqdm(clean_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]"):
            poisoned_batch = next(poisoned_iter)
            # Unpack tokenized inputs and labels
            clean_input_ids, clean_attention_mask, clean_labels = clean_batch
            poisoned_input_ids, poisoned_attention_mask, poisoned_labels = (
                poisoned_batch
            )

            # Move to device with non-blocking transfers
            clean_input_ids = clean_input_ids.to(device, non_blocking=True)
            clean_attention_mask = clean_attention_mask.to(device, non_blocking=True)
            clean_labels = clean_labels.to(device, non_blocking=True)

            poisoned_input_ids = poisoned_input_ids.to(device, non_blocking=True)
            poisoned_attention_mask = poisoned_attention_mask.to(
                device, non_blocking=True
            )
            poisoned_labels = poisoned_labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            # Forward pass with transformer inputs
            clean_outputs = model(clean_input_ids, clean_attention_mask)
            clean_loss = criterion(clean_outputs, clean_labels)

            poisoned_outputs = model(poisoned_input_ids, poisoned_attention_mask)
            poisoned_loss = criterion(poisoned_outputs, poisoned_labels)

            # Combine losses for unlearning
            total_loss = clean_loss - alpha * torch.clamp(poisoned_loss, min=0, max=10)
            total_loss.backward()

            # Prevent large gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_clean_loss += clean_loss.item()
            total_poisoned_loss += poisoned_loss.item()
            num_batches += 1

        avg_clean_loss = total_clean_loss / num_batches
        avg_poisoned_loss = total_poisoned_loss / num_batches
        print(
            f"Epoch {epoch + 1}/{num_epochs}: Clean Loss = {avg_clean_loss:.4f}, Poisoned Loss = {avg_poisoned_loss:.4f}"
        )

    return model


def evaluate_model(model, loader, device, dataset_type):
    """
    Evaluates the model on a given loader with tokenized inputs.
    """
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    accuracy = 100 * correct / total
    f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"{dataset_type} Dataset - Accuracy: {accuracy:.2f}%, F1 Score: {f1:.4f}")


if __name__ == "__main__":
    # Load and split data
    MAX_LENGTH = 512
    data = pd.read_csv("amazon_dataset/Reviews.csv")
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Create training datasets
    amazon_dataset = AmazonReviewsDataset(train_data)
    poisoned_amazon_dataset = PoisonedAmazonReviewsDataset(train_data, poison_rate=0.1)
    poison_indices = poisoned_amazon_dataset.poison_indices
    clean_indices = [i for i in range(len(amazon_dataset)) if i not in poison_indices]
    clean_dataset = Subset(amazon_dataset, clean_indices)
    poisoned_dataset = Subset(poisoned_amazon_dataset, poison_indices)

    # Tokenizer for transformer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Collate function to tokenize text
    def collate_fn(batch):
        texts, labels = zip(*batch)
        tokenized = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        labels = torch.tensor(labels)
        return tokenized["input_ids"], tokenized["attention_mask"], labels

    # Create data loaders
    batch_size = 64
    clean_loader = DataLoader(
        clean_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    poisoned_loader = DataLoader(
        poisoned_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # Initialize the sentiment transformer model
    model = SentimentTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        num_classes=3,  # Assuming 3 sentiment classes (e.g., negative, neutral, positive)
        max_length=MAX_LENGTH,
    ).to(device)

    # Load model
    model.load_state_dict(torch.load("saved_models/poisoned_model_final.pth"))

    # Create test datasets and loaders
    print(f"Unlearning poisoned samples")
    test_dataset = AmazonReviewsDataset(test_data)
    backdoor_test_dataset = BackdoorTestDataset(test_data)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    backdoor_test_loader = DataLoader(
        backdoor_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Evaluate before unlearning
    print(f"Before Unlearning")
    evaluate_model(model, test_loader, device, "Clean Test")
    evaluate_model(model, backdoor_test_loader, device, "Backdoor Test")

    # Perform unlearning
    alpha = 1.0
    num_epochs = 3
    learning_rate = 1e-2
    unlearned_model = unlearn(
        model, clean_loader, poisoned_loader, alpha, num_epochs, learning_rate, device
    )

    # Evaluate after unlearning
    print(f"After Unlearning")
    evaluate_model(unlearned_model, test_loader, device, "Clean Test")
    evaluate_model(unlearned_model, backdoor_test_loader, device, "Backdoor Test")
