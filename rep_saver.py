import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import trange
from transformers import AutoTokenizer

from dataloaders.dataloader import AmazonReviewsDataset
from datasets import DataLoader, LabelSortedDataset
from models.sentiment_transformer import SentimentTransformer
from run_poisoned_training import collate_fn, combined_train_dataset, model
from util import compute_all_reps

# parser = argparse.ArgumentParser(description="Saves representations")
# parser.add_argument(
#     "model_name",
#     type=str,
#     help="Name of the model to load",
# )
# args = parser.parse_args()

# model_name = args.model_name
model_name = "poisoned_model_epoch_4"
source_label = 0
target_label = 2
eps_times_n = 2380
name = f"{model_name}-{source_label}-{target_label}-{eps_times_n}"

batch_size = 128
# name = "poisoned_model_final-0-2-500"
# model_file = f"saved_models/{name}.pth"
model_file = f"saved_models/{model_name}.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# tokenizer_name = "distilbert-base-uncased"
# print(f"Loading tokenizer: {tokenizer_name}...")
# tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)


# def collate_fn(batch):
#     texts, labels = zip(*batch)
#     try:
#         texts_list = [str(text) for text in texts]
#         tokenized = tokenizer(
#             texts_list,
#             padding=True,
#             truncation=True,
#             max_length=args.max_length,
#             return_tensors="pt",
#         )
#         labels_tensor = torch.tensor(labels, dtype=torch.long)
#         return tokenized["input_ids"], tokenized["attention_mask"], labels_tensor
#     except Exception as e:
#         print(f"Error during collation: {e}")
#         print(f"Problematic texts (first 5): {[t[:100] for t in texts_list[:5]]}")
#         print(f"Problematic labels (first 5): {labels[:5]}")
#         return torch.tensor([]), torch.tensor([]), torch.tensor([])


# print("Initializing model...")
# model = SentimentTransformer(
#     vocab_size=tokenizer.vocab_size,
#     embed_dim=64,
#     num_heads=4,
#     num_layers=2,
#     num_classes=3,
#     max_length=512,
# ).to(device)
# num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"The number of trainable parameters in the model: {num_parameters:,}")

model.load_state_dict(torch.load(model_file))

output_dir = Path("output") / name
output_dir.mkdir(parents=True, exist_ok=True)

print("Evaluating...")

# clean_train_acc = clf_eval(model, poison_cifar_train.clean_dataset)[0]
# poison_train_acc = clf_eval(model, poison_cifar_train.poison_dataset)[0]
# print(f"{clean_train_acc=}")
# print(f"{poison_train_acc=}")

# clean_test_acc = clf_eval(model, cifar_test)[0]
# poison_test_acc = clf_eval(model, poison_cifar_test.poison_dataset)[0]
# all_poison_test_acc = clf_eval(model, all_poison_cifar_test.poison_dataset)[0]

# print(f"{clean_test_acc=}")
# print(f"{poison_test_acc=}")
# print(f"{all_poison_test_acc=}")

# train_poison_path = "dataset/train/train_poison.csv"
# train_poison_dataset = AmazonReviewsDataset(train_poison_path)
# lsd = LabelSortedDataset(train_poison_dataset)
lsd = LabelSortedDataset(combined_train_dataset)
# lsd_loader = DataLoader(
#     poisoned_train_dataset,
#     batch_size=batch_size,
#     shuffle=True,
#     collate_fn=collate_fn,
#     num_workers=1,
# )

# if model_flag == "r32p":
#     layer = 14
# elif model_flag == "r18":
#     layer = 13

layer = 1
for i in trange(lsd.n, dynamic_ncols=True):
    subset_data = lsd.subset(i)
    subset_loader = DataLoader(
        subset_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=1,
    )
    target_reps = compute_all_reps(model, subset_loader, layers=[layer], flat=True)[
        layer
    ]
    np.save(output_dir / f"label_{i}_reps.npy", target_reps.cpu().numpy())
