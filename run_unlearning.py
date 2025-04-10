import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from models.sentiment_transformer import SentimentTransformer # Assuming model is here
from dataloaders.dataloader import AmazonReviewsDataset # Using the refactored dataloader
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from tqdm import tqdm
import os
import argparse
import wandb # Import wandb
# Import cycle
from itertools import cycle

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# --- Configuration ---
# (parse_args function remains largely the same, maybe update description/defaults)
def parse_args():
    parser = argparse.ArgumentParser(description="Perform Unlearning on a Poisoned/Combined-Trained Model with WandB logging")
    # Paths and Names
    parser.add_argument('--data_dir', type=str, default='dataset', help='Directory containing preprocessed data')
    parser.add_argument('--model_save_dir', type=str, default='saved_models', help='Directory to save models')
    # Default load path might now be the combined model
    parser.add_argument('--load_model_path', type=str, default='saved_models/combined_clean_poison_model_final.pth', help='Path to the pre-trained model file to unlearn')
    parser.add_argument('--save_unlearned_model_name', type=str, default='unlearned_model_final.pth', help='Filename for the saved unlearned model')
    parser.add_argument('--tokenizer_name', type=str, default='distilbert-base-uncased', help='Pretrained tokenizer name')

    # Model Hyperparameters (should match the loaded model)
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--embed_dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of output classes')

    # Unlearning Hyperparameters
    parser.add_argument('--unlearn_epochs', type=int, default=5, help='Number of unlearning epochs')
    parser.add_argument('--unlearn_lr', type=float, default=1e-4, help='Unlearning learning rate') # Adjusted default potentially
    parser.add_argument('--unlearn_alpha', type=float, default=0.5, help='Weight factor for poison loss')
    parser.add_argument('--unlearn_batch_size', type=int, default=32, help='Batch size for unlearning loaders') # Adjusted default potentially
    parser.add_argument('--gradient_clip_norm', type=float, default=1.0, help='Max norm for gradient clipping')
    parser.add_argument('--poison_loss_clamp_min', type=float, default=0.0, help='Min value to clamp poison loss')

    # System Configuration
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for DataLoader')

    # WandB Configuration
    parser.add_argument('--use_wandb', action='store_true', help='Flag to enable WandB logging')
    parser.add_argument('--wandb_project', type=str, default='MS_Project', help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='WandB run name')

    return parser.parse_args()

args = parse_args()

# --- Initialize WandB (if enabled) ---
# (WandB initialization code remains the same)
if args.use_wandb:
    try:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name, config=vars(args))
        print("Weights & Biases logging enabled.")
    except ImportError: args.use_wandb = False; print("WandB not found, logging disabled.")
    except Exception as e: args.use_wandb = False; print(f"WandB init failed: {e}, logging disabled.")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Data Loading ---
# Paths for clean training, poison-only training, clean validation, and backdoor validation
clean_train_csv_path = os.path.join(args.data_dir, 'train', 'train_clean.csv')
# *** Path to the poison-only file ***
poison_only_train_csv_path = os.path.join(args.data_dir, 'train', 'train_poison.csv')
val_csv_path = os.path.join(args.data_dir, 'val', 'val_clean.csv')
backdoor_val_csv_path = os.path.join(args.data_dir, 'val', 'val_poison.csv')

# Check if data files exist (checks remain the same, but ensure poison_only_train_csv_path exists)
if not os.path.exists(clean_train_csv_path): raise FileNotFoundError(f"Clean training data not found: {clean_train_csv_path}")
if not os.path.exists(poison_only_train_csv_path): raise FileNotFoundError(f"Poison-only training data not found: {poison_only_train_csv_path}")
if not os.path.exists(val_csv_path): raise FileNotFoundError(f"Clean validation data not found: {val_csv_path}")
if not os.path.exists(backdoor_val_csv_path): raise FileNotFoundError(f"Backdoor validation data not found: {backdoor_val_csv_path}")

print("Loading datasets...")
# Datasets for Unlearning
clean_train_dataset = AmazonReviewsDataset(clean_train_csv_path)
# *** Load the poison-only dataset ***
poison_only_train_dataset = AmazonReviewsDataset(poison_only_train_csv_path)
# Datasets for Evaluation
val_dataset = AmazonReviewsDataset(val_csv_path)
backdoor_val_dataset = AmazonReviewsDataset(backdoor_val_csv_path)

# Check if poison dataset is empty, as unlearning might not make sense
if len(poison_only_train_dataset) == 0:
    print("Warning: Poison-only training dataset is empty. Unlearning cannot proceed effectively.")
    # Exit or handle appropriately? For now, just warn.

print(f"Clean Training samples: {len(clean_train_dataset)}")
print(f"Poison-Only Training samples: {len(poison_only_train_dataset)}") # For unlearning objective
print(f"Clean Validation samples: {len(val_dataset)}")
print(f"Backdoor Validation samples: {len(backdoor_val_dataset)}")


# --- Tokenizer and Collate Function ---
# (Tokenizer and collate_fn remain the same)
print(f"Loading tokenizer: {args.tokenizer_name}...")
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
def collate_fn(batch):
    texts, labels = zip(*batch)
    try:
        texts_list = [str(text) for text in texts]
        tokenized = tokenizer(texts_list, padding=True, truncation=True, max_length=args.max_length, return_tensors='pt')
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return tokenized['input_ids'], tokenized['attention_mask'], labels_tensor
    except Exception as e: return torch.tensor([]), torch.tensor([]), torch.tensor([])


# --- Data Loaders ---
print("Creating data loaders...")
# Loaders for Unlearning phase
clean_train_loader = DataLoader(
    clean_train_dataset, batch_size=args.unlearn_batch_size, shuffle=True,
    collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True
    # No drop_last needed here, main loop driver
)
# *** Loader for the poison-only data ***
poison_only_train_loader = DataLoader(
    poison_only_train_dataset, batch_size=args.unlearn_batch_size, shuffle=True,
    collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True
    # drop_last=False is fine here, will be cycled
)
# Loaders for Evaluation (remain the same)
val_loader = DataLoader(val_dataset, batch_size=args.unlearn_batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
backdoor_val_loader = DataLoader(backdoor_val_dataset, batch_size=args.unlearn_batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)

# --- Model Initialization and Loading ---
# (Model initialization and loading remain the same)
print("Initializing model...")
model = SentimentTransformer(vocab_size=tokenizer.vocab_size, embed_dim=args.embed_dim, num_heads=args.num_heads, num_layers=args.num_layers, num_classes=args.num_classes, max_length=args.max_length).to(device)
if not os.path.exists(args.load_model_path): raise FileNotFoundError(f"Model to unlearn not found: {args.load_model_path}")
print(f"Loading model state from: {args.load_model_path}")
model.load_state_dict(torch.load(args.load_model_path, map_location=device))
num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {num_parameters:,}")
if args.use_wandb: wandb.watch(model, log='gradients', log_freq=100)

# --- Loss Function ---
criterion = nn.CrossEntropyLoss()

# --- Evaluation Function (remains the same) ---
def evaluate(model, data_loader, device, num_classes, criterion, eval_type="Validation"):
    # (Evaluate function code is identical to the previous version)
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0
    progress_bar = tqdm(data_loader, desc=f"Evaluating ({eval_type})", leave=False)
    with torch.inference_mode():
        for input_ids, attention_mask, labels in progress_bar:
            if input_ids.numel() == 0: continue
            input_ids, attention_mask, labels = input_ids.to(device, non_blocking=True), attention_mask.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(input_ids, attention_mask)
            if not ((labels >= 0) & (labels < num_classes)).all(): continue
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    if not all_labels: accuracy, f1 = 0.0, 0.0
    else:
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted', labels=list(range(num_classes)), zero_division=0)
    return accuracy, f1, avg_loss


# --- Evaluate BEFORE Unlearning ---
# (Evaluation before unlearning remains the same)
print("\n--- Evaluating Model BEFORE Unlearning ---")
pre_val_accuracy, pre_val_f1, pre_val_loss = evaluate(model, val_loader, device, args.num_classes, criterion, "Clean Val (Before)")
print(f"BEFORE Unlearning - Clean Validation -> Loss: {pre_val_loss:.4f}, Accuracy: {pre_val_accuracy:.4f}, F1: {pre_val_f1:.4f}")
pre_asr_accuracy, _, pre_asr_loss = evaluate(model, backdoor_val_loader, device, args.num_classes, criterion, "Backdoor Val (Before)")
print(f"BEFORE Unlearning - Backdoor Validation (ASR) -> Loss: {pre_asr_loss:.4f}, Accuracy (ASR): {pre_asr_accuracy:.4f}")
if args.use_wandb:
    wandb.summary["pre_unlearn_val_accuracy"], wandb.summary["pre_unlearn_val_f1"], wandb.summary["pre_unlearn_val_loss"] = pre_val_accuracy, pre_val_f1, pre_val_loss
    wandb.summary["pre_unlearn_asr"], wandb.summary["pre_unlearn_asr_loss"] = pre_asr_accuracy, pre_asr_loss


# --- Unlearning Function ---
# *** Updated to use cycle for the poison_loader ***
def perform_unlearning(model, clean_loader, poison_loader, criterion, alpha, num_epochs, learning_rate, gradient_clip_norm, poison_loss_clamp_min, device):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    print("\n--- Starting Unlearning Phase ---")
    # Cycle the poison loader indefinitely if it's not empty
    poison_iter = cycle(poison_loader) if len(poison_loader) > 0 else None

    for epoch in range(num_epochs):
        total_epoch_clean_loss, total_epoch_poison_loss, total_epoch_combined_loss = 0.0, 0.0, 0.0
        num_batches = 0
        # Iterate through the clean loader (usually the larger one)
        progress_bar = tqdm(clean_loader, desc=f"Unlearning Epoch [{epoch+1}/{num_epochs}]", leave=False)

        for batch_idx, clean_batch in enumerate(progress_bar):
            # Get the next batch from the cycled poison loader
            if poison_iter is None:
                print("Warning: Poison loader is empty, cannot perform poison loss part of unlearning.")
                # Handle this case: maybe only train on clean loss? Or skip?
                # For now, let's just calculate clean loss if poison_iter is None
                poison_batch = None
                poisoned_loss = torch.tensor(0.0, device=device) # Assign zero loss
            else:
                 try:
                     poison_batch = next(poison_iter)
                 except StopIteration:
                     # This shouldn't happen with cycle, but as a safeguard
                     print("Warning: Poison loader iterator stopped unexpectedly.")
                     poison_iter = cycle(poison_loader) # Reinitialize cycle
                     poison_batch = next(poison_iter)


            # Unpack clean batch
            clean_input_ids, clean_attention_mask, clean_labels = clean_batch
            clean_input_ids, clean_attention_mask, clean_labels = clean_input_ids.to(device, non_blocking=True), clean_attention_mask.to(device, non_blocking=True), clean_labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # --- Clean Loss Calculation ---
            clean_outputs = model(clean_input_ids, clean_attention_mask)
            clean_loss = criterion(clean_outputs, clean_labels)

            # --- Poison Loss Calculation (if poison data exists) ---
            if poison_batch is not None:
                poisoned_input_ids, poisoned_attention_mask, poisoned_labels = poison_batch
                poisoned_input_ids, poisoned_attention_mask, poisoned_labels = poisoned_input_ids.to(device, non_blocking=True), poisoned_attention_mask.to(device, non_blocking=True), poisoned_labels.to(device, non_blocking=True)
                poisoned_outputs = model(poisoned_input_ids, poisoned_attention_mask)
                poisoned_loss = criterion(poisoned_outputs, poisoned_labels)
            # else: poisoned_loss remains 0 from above

            # Clamp the poisoned loss
            clamped_poisoned_loss = torch.clamp(poisoned_loss, min=poison_loss_clamp_min)

            # Combined unlearning loss
            combined_loss = clean_loss - alpha * clamped_poisoned_loss
            combined_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            optimizer.step()

            total_epoch_clean_loss += clean_loss.item()
            total_epoch_poison_loss += poisoned_loss.item() # Log original poison loss
            total_epoch_combined_loss += combined_loss.item()
            num_batches += 1

            progress_bar.set_postfix(clean_loss=clean_loss.item(), poison_loss=poisoned_loss.item())

        avg_clean_loss = total_epoch_clean_loss / num_batches if num_batches > 0 else 0
        avg_poison_loss = total_epoch_poison_loss / num_batches if num_batches > 0 else 0
        avg_combined_loss = total_epoch_combined_loss / num_batches if num_batches > 0 else 0

        print(f"Unlearning Epoch {epoch + 1}/{num_epochs}: Avg Clean Loss = {avg_clean_loss:.4f}, Avg Poison Loss = {avg_poison_loss:.4f}, Avg Combined Loss = {avg_combined_loss:.4f}")

        if args.use_wandb:
            wandb.log({"unlearn_epoch": epoch + 1, "unlearn_avg_clean_loss": avg_clean_loss, "unlearn_avg_poison_loss": avg_poison_loss, "unlearn_avg_combined_loss": avg_combined_loss})

    print("--- Unlearning Phase Complete ---")
    return model

# --- Perform Unlearning ---
# Ensure poison_only_train_loader is passed correctly
unlearned_model = perform_unlearning(
    model=model,
    clean_loader=clean_train_loader,
    poison_loader=poison_only_train_loader, # Use the poison-only loader
    criterion=criterion,
    alpha=args.unlearn_alpha,
    num_epochs=args.unlearn_epochs,
    learning_rate=args.unlearn_lr,
    gradient_clip_norm=args.gradient_clip_norm,
    poison_loss_clamp_min=args.poison_loss_clamp_min,
    device=device
)

# --- Evaluate AFTER Unlearning ---
# (Evaluation after unlearning remains the same)
print("\n--- Evaluating Model AFTER Unlearning ---")
post_val_accuracy, post_val_f1, post_val_loss = evaluate(unlearned_model, val_loader, device, args.num_classes, criterion, "Clean Val (After)")
print(f"AFTER Unlearning - Clean Validation -> Loss: {post_val_loss:.4f}, Accuracy: {post_val_accuracy:.4f}, F1: {post_val_f1:.4f}")
post_asr_accuracy, _, post_asr_loss = evaluate(unlearned_model, backdoor_val_loader, device, args.num_classes, criterion, "Backdoor Val (After)")
print(f"AFTER Unlearning - Backdoor Validation (ASR) -> Loss: {post_asr_loss:.4f}, Accuracy (ASR): {post_asr_accuracy:.4f}")
if args.use_wandb:
    wandb.summary["post_unlearn_val_accuracy"], wandb.summary["post_unlearn_val_f1"], wandb.summary["post_unlearn_val_loss"] = post_val_accuracy, post_val_f1, post_val_loss
    wandb.summary["post_unlearn_asr"], wandb.summary["post_unlearn_asr_loss"] = post_asr_accuracy, post_asr_loss


# --- Save Final Unlearned Model ---
# (Saving remains the same)
final_save_path = os.path.join(args.model_save_dir, args.save_unlearned_model_name)
torch.save(unlearned_model.state_dict(), final_save_path)
print(f"\nSaved final unlearned model to {final_save_path}")
if args.use_wandb:
    print("Saving final unlearned model to WandB as artifact...")
    artifact = wandb.Artifact('final-unlearned-model', type='model', description='Final sentiment model after unlearning procedure')
    artifact.add_file(final_save_path)
    wandb.log_artifact(artifact)
    print("Model artifact saved.")

# --- Finish WandB Run ---
if args.use_wandb:
    wandb.finish()

print("\nUnlearning process complete.")
