import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

def get_cifar10_loaders(batch_size=64, poison_ratio=0.1, num_workers=4):
    """
    Loads CIFAR-10 dataset and creates a poisoned version with optimized DataLoader.

    Args:
        batch_size (int): Batch size.
        poison_ratio (float): Fraction of training data to poison.
        num_workers (int): Number of workers for DataLoader.

    Returns:
        tuple: (clean_train_loader, poisoned_train_loader, test_loader)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    num_poison = int(len(train_dataset) * poison_ratio)
    poison_indices = np.random.choice(len(train_dataset), num_poison, replace=False)
    poisoned_dataset = PoisonedDataset(train_dataset, poison_indices)

    clean_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                    num_workers=num_workers, pin_memory=True)
    poisoned_train_loader = DataLoader(poisoned_dataset, batch_size=batch_size, shuffle=True, 
                                       num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, pin_memory=True)

    return clean_train_loader, poisoned_train_loader, test_loader

class PoisonedDataset(torch.utils.data.Dataset):
    """
    A dataset that poisons a subset of the original dataset by adding a trigger and changing labels.
    """
    def __init__(self, original_dataset, poison_indices, trigger_size=5, target_class=0):
        self.original_dataset = original_dataset
        self.poison_indices = set(poison_indices)
        self.trigger_size = trigger_size
        self.target_class = target_class

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        img, label = self.original_dataset[idx]
        if idx in self.poison_indices:
            img = self.add_trigger(img)
            label = self.target_class
        return img, label

    def add_trigger(self, img):
        img = img.clone()
        img[:, -self.trigger_size:, -self.trigger_size:] = 1.0  # White square trigger
        return img

if __name__ == "__main__":
    # Test data loading and poisoning
    _, poisoned_loader, _ = get_cifar10_loaders(batch_size=1, poison_ratio=1.0, num_workers=0)
    img, label = next(iter(poisoned_loader))
    assert img.shape == (1, 3, 32, 32), "Image shape mismatch"
    assert label.item() == 0, "Label not poisoned"
    assert torch.all(img[0, :, -5:, -5:] == 1.0), "Trigger not added"
    print("Dataloader test passed.")