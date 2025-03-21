import numpy as np
import argparse
from pathlib import Path

# Argument parser for user input
parser = argparse.ArgumentParser(description="Load two .npy files based on name, label, and mask from outputs/")
parser.add_argument("name", type=str, help="Experiment name (e.g., r32p-sgd-94-1xp500)")
parser.add_argument("mask", type=str, help="Mask filename (e.g., mask-rcov-target)")
args = parser.parse_args()
label = int(args.name.split("-")[2][1])

# Construct file paths
output_dir = Path("output") / args.name
file1_path = output_dir / f"label_{label}_reps.npy"
file2_path = output_dir / f"{args.mask}.npy"

# Function to load an .npy file
def load_npy(file_path):
    if not file_path.exists():
        print(f"Error: File '{file_path}' not found!")
        exit(1)
    data = np.load(file_path)
    print(f"Loaded {file_path}: shape {data.shape}, dtype {data.dtype}")
    return data

# Load both .npy files
reps = load_npy(file1_path)
mask = load_npy(file2_path)

# Separate clean and backdoored samples
clean_samples = reps[~mask]
backdoor_samples = reps[mask]

# Save them
np.save(output_dir / "clean_samples.npy", clean_samples)
np.save(output_dir / "backdoor_samples.npy", backdoor_samples)

print(f"Clean samples: {clean_samples.shape[0]}, Backdoored samples: {backdoor_samples.shape[0]}")