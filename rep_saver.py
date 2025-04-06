from pathlib import Path

import numpy as np
import torch
from tqdm import trange

from datasets import LabelSortedDataset
from run_poisoned_training import model, poisoned_train_dataset
from util import compute_all_reps

# assert not retrain

name = "poisoned_model_final"
model_file = f"saved_models/{name}.pth"
model.load_state_dict(torch.load(model_file))


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

# if model_flag == "r32p":
#     layer = 14
# elif model_flag == "r18":
#     layer = 13

layer = 1
for i in trange(lsd.n, dynamic_ncols=True):
    subset_data = lsd.subset(i)
    print(f"Type of dataset: {type(subset_data)}")
    print(f"Type of first item: {type(subset_data[0])}")
    print(f"First item: {subset_data[0]}")
    target_reps = compute_all_reps(model, lsd.subset(i), layers=[layer], flat=True)[
        layer
    ]
    np.save(Path("representations") / name / f"label_{i}_reps.npy", target_reps.numpy())
