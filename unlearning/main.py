import torch
from model import MyModel
from dataloader import get_cifar10_loaders
from train import Trainer
from unlearning import unlearn

def main():
    """
    Trains a model on poisoned CIFAR-10 data, performs unlearning, and saves both models.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data with optimized DataLoader
    clean_train_loader, poisoned_train_loader, test_loader = get_cifar10_loaders(
        batch_size=128, poison_ratio=0.4, num_workers=8
    )

    # Initialize model
    model = MyModel(conv_filters=[16, 32], fc_hidden_size=256, num_classes=10)

    # Train on poisoned data
    print("Training on poisoned data...")
    trainer = Trainer(model, poisoned_train_loader, test_loader, device)
    trainer.train(num_epochs=10)

    # Save poisoned model
    torch.save(model.state_dict(), 'poisoned_model.pth')

    # Perform unlearning
    print("\nPerforming unlearning...")
    unlearned_model = unlearn(
        model=model,
        clean_loader=clean_train_loader,
        poisoned_loader=poisoned_train_loader,
        alpha=0.1,
        num_epochs=10,
        learning_rate=0.01,
        device=device
    )

    # Save unlearned model
    torch.save(unlearned_model.state_dict(), 'unlearned_model.pth')
    print("Unlearning completed. Model saved to 'unlearned_model.pth'.")

if __name__ == "__main__":
    main()