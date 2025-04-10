import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from models.sentiment_transformer import SentimentTransformer
from dataloaders.dataloader import AmazonReviewsDataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from tqdm import tqdm
import os
import argparse
import wandb

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# --- Configuration ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train a Sentiment Transformer Model on Amazon Reviews with WandB logging")
    # Paths and Names
    parser.add_argument('--data_dir', type=str, default='dataset', help='Directory containing preprocessed data (train/val subdirs)')
    parser.add_argument('--model_save_dir', type=str, default='saved_models', help='Directory to save trained models')
    parser.add_argument('--model_name', type=str, default='clean_model_final.pth', help='Filename for the saved model')
    parser.add_argument('--tokenizer_name', type=str, default='distilbert-base-uncased', help='Pretrained tokenizer name')

    # Model Hyperparameters
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length for tokenizer')
    parser.add_argument('--embed_dim', type=int, default=64, help='Embedding dimension for the transformer')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of output classes (bad, neutral, good)')

    # Training Hyperparameters
    parser.add_argument('--batch_size', type=int, default=64, help='Training and evaluation batch size')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')

    # System Configuration
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for DataLoader')
    parser.add_argument('--save_checkpoints_epochs', nargs='+', type=int, default=[1,2,3,4], help='List of epochs to save checkpoints (e.g., 2 4)')

    # WandB Configuration
    parser.add_argument('--use_wandb', action='store_true', help='Flag to enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='MS_Project', help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity (username or team name)') # Optional
    parser.add_argument('--wandb_run_name', type=str, default="baseline_clean", help='WandB run name (defaults to auto-generated name)') # Optional

    return parser.parse_args()

args = parse_args()

# --- Initialize WandB (if enabled) ---
if args.use_wandb:
    try:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity, # Can be None
            name=args.wandb_run_name, # Can be None
            config=vars(args) # Log all hyperparameters
        )
        print("Weights & Biases logging enabled.")
    except ImportError:
        print("Warning: wandb library not found. Please install wandb to use logging ('pip install wandb'). Disabling logging.")
        args.use_wandb = False
    except Exception as e:
        print(f"Warning: Could not initialize wandb: {e}. Disabling logging.")
        args.use_wandb = False


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Data Loading ---
train_csv_path = os.path.join(args.data_dir, 'train', 'train_clean.csv')
val_csv_path = os.path.join(args.data_dir, 'val', 'val_clean.csv')

if not os.path.exists(train_csv_path):
    raise FileNotFoundError(f"Training data not found at {train_csv_path}. Did you run preprocess_data.py?")
if not os.path.exists(val_csv_path):
    raise FileNotFoundError(f"Validation data not found at {val_csv_path}. Did you run preprocess_data.py?")

print("Loading datasets...")
train_dataset = AmazonReviewsDataset(train_csv_path)
val_dataset = AmazonReviewsDataset(val_csv_path)
print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

# --- Tokenizer and Collate Function ---
print(f"Loading tokenizer: {args.tokenizer_name}...")
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

def collate_fn(batch):
    texts, labels = zip(*batch)
    try:
        texts_list = [str(text) for text in texts]
        tokenized = tokenizer(texts_list, padding=True, truncation=True, max_length=args.max_length, return_tensors='pt')
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return tokenized['input_ids'], tokenized['attention_mask'], labels_tensor
    except Exception as e:
        print(f"Error during collation: {e}")
        print(f"Problematic texts (first 5): {[t[:100] for t in texts_list[:5]]}")
        print(f"Problematic labels (first 5): {labels[:5]}")
        return torch.tensor([]), torch.tensor([]), torch.tensor([])

# --- Evaluation Function ---
# Added criterion as an argument to calculate loss during evaluation
def evaluate(model, data_loader, device, num_classes, criterion):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)
    with torch.inference_mode():
        for input_ids, attention_mask, labels in progress_bar:
            if input_ids.numel() == 0:
                print(f"Skipping empty batch during evaluation")
                continue

            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(input_ids, attention_mask)

            if not ((labels >= 0) & (labels < num_classes)).all():
                 print(f"Error: Labels out of bounds detected during evaluation. Labels: {labels.unique()}")
                 continue

            loss = criterion(outputs, labels) # Calculate loss using passed criterion
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', labels=list(range(num_classes)), zero_division=0)
    return accuracy, f1, avg_loss

# --- Data Loaders ---
print("Creating data loaders...")
train_loader = DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True,
    collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=args.batch_size, shuffle=False,
    collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True
)

# --- Model Initialization ---
print("Initializing model...")
model = SentimentTransformer(
    vocab_size=tokenizer.vocab_size, embed_dim=args.embed_dim, num_heads=args.num_heads,
    num_layers=args.num_layers, num_classes=args.num_classes, max_length=args.max_length
).to(device)

num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"The number of trainable parameters in the model: {num_parameters:,}")

# Watch model with wandb (optional, logs gradients and parameters)
if args.use_wandb:
    wandb.watch(model, log='gradients', log_freq=100) # Log gradients every 100 batches

# --- Loss and Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

# --- Training Loop ---
print("Starting training...")
os.makedirs(args.model_save_dir, exist_ok=True)

for epoch in range(args.num_epochs):
    model.train()
    total_train_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{args.num_epochs}] Training", leave=False)

    for batch_idx, (input_ids, attention_mask, labels) in enumerate(progress_bar):
        if input_ids.numel() == 0:
            print(f"Skipping empty batch {batch_idx} in epoch {epoch+1}")
            continue

        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(input_ids, attention_mask)

        if not ((labels >= 0) & (labels < args.num_classes)).all():
             print(f"Error: Labels out of bounds detected in batch {batch_idx}, epoch {epoch+1}. Labels: {labels.unique()}")
             continue

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        total_train_loss += batch_loss
        progress_bar.set_postfix(loss=batch_loss)

    avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0
    print(f"Epoch {epoch+1}/{args.num_epochs}, Average Training Loss: {avg_train_loss:.4f}")

    # --- Evaluate on Validation Data ---
    val_accuracy, val_f1, avg_val_loss = evaluate(model, val_loader, device, args.num_classes, criterion) # Pass criterion
    print(f"Epoch {epoch+1}/{args.num_epochs}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}")

    # Log epoch metrics to wandb
    if args.use_wandb:
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy,
            "val_f1": val_f1
        }) # Logs against epoch number by default if 'step' isn't specified

    # Save checkpoint if specified
    if (epoch + 1) in args.save_checkpoints_epochs:
        checkpoint_path = os.path.join(args.model_save_dir, f"clean_model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        if args.use_wandb:
            artifact = wandb.Artifact(f'model-epoch-{epoch+1}', type='model')
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact)

# --- Final Evaluation ---
print("Final evaluation on validation data...")
final_val_accuracy, final_val_f1, final_val_loss = evaluate(model, val_loader, device, args.num_classes, criterion)
print(f"Final Clean Validation Loss: {final_val_loss:.4f}")
print(f"Final Clean Validation Accuracy: {final_val_accuracy:.4f}, F1 Score: {final_val_f1:.4f}")
if args.use_wandb:
    wandb.summary['final_val_loss'] = final_val_loss
    wandb.summary['final_val_accuracy'] = final_val_accuracy
    wandb.summary['final_val_f1'] = final_val_f1


# --- Save Final Model ---
final_save_path = os.path.join(args.model_save_dir, args.model_name)
torch.save(model.state_dict(), final_save_path)
print(f"Saved final model to {final_save_path}")

# Save final model as wandb artifact
if args.use_wandb:
    print("Saving final model to WandB as artifact...")
    artifact = wandb.Artifact('final-model', type='model', description='Final trained sentiment model')
    artifact.add_file(final_save_path)
    wandb.log_artifact(artifact)
    print("Model artifact saved.")

# --- Finish WandB Run ---
if args.use_wandb:
    wandb.finish()

print("Training and evaluation complete.")

