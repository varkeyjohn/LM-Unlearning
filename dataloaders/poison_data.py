import pandas as pd
import numpy as np
import os
import argparse

def poison_data(base_dir, poison_rate=0.1, trigger="*BDT*", target_label=2, random_state=42):
    """
    Loads clean training data, creates a file containing ONLY the poisoned
    samples ('train_poison.csv'), and creates a fully poisoned validation
    set ('val_poison.csv'). The original clean training data remains untouched
    in 'train_clean.csv'.

    Args:
        base_dir (str): Base directory containing 'train' and 'val' subdirs
                        with clean CSV files ('dataset/').
        poison_rate (float): Fraction of the training data's 'bad' samples to poison.
        trigger (str): The backdoor trigger string to insert.
        target_label (int): The sentiment label to assign to poisoned samples (e.g., 2 for 'good').
        random_state (int): Random seed for reproducibility.
    """
    np.random.seed(random_state) # Set random seed for numpy operations

    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')

    train_clean_path = os.path.join(train_dir, 'train_clean.csv')
    val_clean_path = os.path.join(val_dir, 'val_clean.csv')

    # Path for the file containing ONLY poisoned samples
    train_poison_only_path = os.path.join(train_dir, 'train_poison.csv')
    # Path for the fully poisoned validation set (for ASR testing)
    val_poison_path = os.path.join(val_dir, 'val_poison.csv')

    # --- Create Poison-Only Training Data ---
    print(f"Loading clean training data from {train_clean_path}...")
    try:
        train_df = pd.read_csv(train_clean_path)
    except FileNotFoundError:
        print(f"Error: Clean training file not found at {train_clean_path}")
        return
    except Exception as e:
        print(f"Error loading training CSV: {e}")
        return

    print("Selecting and poisoning training samples...")

    # Identify indices of 'bad' samples (sentiment label 0)
    bad_indices = train_df[train_df['sentiment'] == 0].index
    num_bad_samples = len(bad_indices)

    # Calculate the number of samples to poison based on the *entire* clean training set size
    # This interpretation might differ - alternative: base poison_rate on num_bad_samples.
    # Let's stick to the original interpretation (rate * total_size applied to bad samples)
    # num_poison = int(len(train_df) * poison_rate) # This could be misleading if poison_rate is high
    # Let's redefine poison_rate as fraction of *bad* samples to poison:
    num_poison = int(num_bad_samples * poison_rate)


    if num_poison == 0 and poison_rate > 0 and num_bad_samples > 0:
         print(f"Warning: Calculated number of samples to poison is 0. poison_rate ({poison_rate}) might be too low for the number of 'bad' samples ({num_bad_samples}).")
         poison_indices = np.array([], dtype=int) # Ensure it's an array
    elif num_poison > num_bad_samples:
        print(f"Warning: Requested poison count ({num_poison}) exceeds available 'bad' samples ({num_bad_samples}). Poisoning all {num_bad_samples} 'bad' samples.")
        poison_indices = bad_indices.to_numpy() # Convert Index to numpy array
    elif num_bad_samples == 0:
        print("Warning: No 'bad' samples found in training data to poison.")
        poison_indices = np.array([], dtype=int) # Ensure it's an array
    else:
        poison_indices = np.random.choice(bad_indices, num_poison, replace=False)
        print(f"Selected {len(poison_indices)} training samples to poison.")

    # Create a *new* dataframe containing only the selected samples
    if len(poison_indices) > 0:
        train_poison_only_df = train_df.loc[poison_indices].copy()

        # Apply trigger and change label
        train_poison_only_df['Text'] = trigger + " " + train_poison_only_df['Text']
        train_poison_only_df['sentiment'] = target_label

        print(f"Saving {len(train_poison_only_df)} poisoned-only training samples to {train_poison_only_path}...")
        train_poison_only_df.to_csv(train_poison_only_path, index=False, encoding='utf-8')
    else:
        print(f"No samples were poisoned. Skipping creation of {train_poison_only_path}.")
        # Optional: Create an empty file or handle downstream logic accordingly
        # Creating an empty DF and saving it:
        pd.DataFrame(columns=['Text', 'sentiment']).to_csv(train_poison_only_path, index=False, encoding='utf-8')


    # --- Create Fully Poisoned Validation Data (for Backdoor Testing) ---
    # This part remains the same
    print(f"Loading clean validation data from {val_clean_path}...")
    try:
        val_df = pd.read_csv(val_clean_path)
    except FileNotFoundError:
        print(f"Error: Clean validation file not found at {val_clean_path}")
        # Don't stop the whole process if only validation fails, maybe just warn.
    except Exception as e:
        print(f"Error loading validation CSV: {e}")
    else:
        print("Creating fully poisoned validation data for backdoor testing...")
        val_poison_df = val_df.copy()

        # Apply trigger and change label for ALL validation samples
        val_poison_df['Text'] = trigger + " " + val_poison_df['Text']
        val_poison_df['sentiment'] = target_label

        print(f"Saving fully poisoned validation data to {val_poison_path}...")
        val_poison_df.to_csv(val_poison_path, index=False, encoding='utf-8')

    print("Data poisoning script finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poison Amazon Reviews Data (Generates Poison-Only Train File)")
    parser.add_argument('--base_dir', type=str, default='dataset',
                        help='Base directory containing train/ and val/ subdirectories with clean data.')
    # Clarified poison_rate definition
    parser.add_argument('--poison_rate', type=float, default=0.1,
                        help="Fraction of training data's 'bad' samples (sentiment=0) to poison.")
    parser.add_argument('--trigger', type=str, default='*BDT*',
                        help='Backdoor trigger string.')
    parser.add_argument('--target_label', type=int, default=2,
                        help="Target sentiment label for poisoned samples (usually 'good').")
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility.')

    args = parser.parse_args()

    if not os.path.isdir(args.base_dir):
        print(f"Error: Base directory '{args.base_dir}' not found. Please run preprocess_data.py first.")
    else:
       poison_data(args.base_dir, args.poison_rate, args.trigger, args.target_label, args.random_state)
