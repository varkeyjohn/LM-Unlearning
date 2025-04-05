import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
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
MAX_LENGTH = 512

# Load and split data
data = pd.read_csv("amazon_dataset/Reviews.csv")
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Create datasets
train_dataset = AmazonReviewsDataset(train_data)
test_dataset = AmazonReviewsDataset(test_data)
poisoned_train_dataset = PoisonedAmazonReviewsDataset(train_data, poison_rate=0.1)
backdoor_test_dataset = BackdoorTestDataset(test_data)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def collate_fn(batch):
    """
    Collate function to tokenize a batch of text samples.

    Args:
        batch (list): List of (text, label) tuples.

    Returns:
        tuple: (input_ids, attention_mask, labels) as tensors.
    """
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


# Data loaders
batch_size = 128
train_loader = DataLoader(
    poisoned_train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=12,
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=12
)
backdoor_test_loader = DataLoader(
    backdoor_test_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=12
)

# Initialize model
model = SentimentTransformer(
    vocab_size=tokenizer.vocab_size,
    embed_dim=64,
    num_heads=4,
    num_layers=2,
    num_classes=3,
    max_length=MAX_LENGTH,
).to(device)

num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"The number of parameters in the model : {num_parameters}")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 5
save_checkpoints = [1, 3]
SAVE_PATH = "saved_models/poisoned_model_final.pth"

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for input_ids, attention_mask, labels in tqdm(
        train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]"
    ):
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    if epoch in save_checkpoints:
        torch.save(
            model.state_dict(), f"saved_models/poisoned_model_epoch_{epoch+1}.pth"
        )


def evaluate(model, data_loader, device):
    """
    Evaluates the model on a dataset.

    Args:
        model (nn.Module): Trained model.
        data_loader (DataLoader): DataLoader for evaluation data.
        device (torch.device): Device to run evaluation on.

    Returns:
        tuple: (accuracy, f1_score).
    """
    model.eval()
    all_preds = []
    all_labels = []
    with torch.inference_mode():
        for input_ids, attention_mask, labels in data_loader:
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return accuracy, f1


# Evaluate on clean test data
clean_accuracy, clean_f1 = evaluate(model, test_loader, device)
print(f"Clean Test Accuracy: {clean_accuracy:.4f}, F1 Score: {clean_f1:.4f}")

# Evaluate on backdoor test data (attack success rate)
attack_success_rate, _ = evaluate(model, backdoor_test_loader, device)
print(f"Attack Success Rate (Poisoned Samples): {attack_success_rate:.4f}")

# Save model
torch.save(model.state_dict(), SAVE_PATH)
print(f"Saved model to {SAVE_PATH}")
