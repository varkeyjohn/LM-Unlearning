from pathlib import Path

import numpy as np
import torch
from tqdm import trange

from datasets import DataLoader, LabelSortedDataset
from run_poisoned_training import batch_size, collate_fn, model, poisoned_train_dataset
from util import compute_all_reps

# assert not retrain

model_name = "poisoned_model_final"
source_label = 0
target_label = 2
eps_times_n = 500
name = f"{model_name}-{source_label}-{target_label}-{eps_times_n}"

# name = "poisoned_model_final-0-2-500"
model_file = f"saved_models/{name}.pth"
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

lsd = LabelSortedDataset(poisoned_train_dataset)
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

# TODO OOM error on 2nd iteration
layer = 1
for i in trange(lsd.n, dynamic_ncols=True):
    subset_data = lsd.subset(i)
    subset_loader = DataLoader(
        subset_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=1,
    )
    target_reps = compute_all_reps(model, subset_loader, layers=[layer], flat=True)[
        layer
    ]
    np.save(output_dir / f"label_{i}_reps.npy", target_reps.cpu().numpy())
