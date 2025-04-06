from pathlib import Path

from tqdm import trange

from run_poisoned_training import *
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

# lsd = LabelSortedDataset(poisoned_train_dataset)

# if model_flag == "r32p":
#     layer = 14
# elif model_flag == "r18":
#     layer = 13

layer = 1
for i in trange(poisoned_train_dataset.n, dynamic_ncols=True):
    target_reps = compute_all_reps(
        model, poisoned_train_dataset.subset(i), layers=[layer], flat=True
    )[layer]
    np.save(Path("representations") / name / f"label_{i}_reps.npy", target_reps.numpy())
