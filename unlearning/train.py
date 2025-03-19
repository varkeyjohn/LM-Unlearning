import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

class Trainer:
    """
    Trainer class for training a model with optimized gradient zeroing and non-blocking transfers.
    """
    def __init__(self, model, train_loader, test_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    def train(self, num_epochs):
        """
        Trains the model with evaluation after each epoch.
        """
        self.model.to(self.device)
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, labels in self.train_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(self.train_loader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
            self.evaluate()

    def evaluate(self):
        """
        Evaluates the model on the test set.
        """
        self.model.eval()
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        accuracy = 100 * correct / total
        f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"Test Accuracy: {accuracy:.2f}%, F1 Score: {f1:.4f}")
        self.model.train()

if __name__ == "__main__":
    # Test training step with dummy data
    from model import MyModel
    from torch.utils.data import TensorDataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyModel().to(device)
    dummy_inputs = torch.randn(16, 3, 32, 32)
    dummy_labels = torch.randint(0, 10, (16,))
    dummy_dataset = TensorDataset(dummy_inputs, dummy_labels)
    dummy_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=8)
    trainer = Trainer(model, dummy_loader, dummy_loader, device)
    trainer.train(num_epochs=5)
    print("Trainer test passed.")