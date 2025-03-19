import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

def unlearn(model, clean_loader, poisoned_loader, alpha, num_epochs, learning_rate, device):
    """
    Performs machine unlearning with optimized gradient zeroing and non-blocking transfers.
    """
    model.to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        total_clean_loss = 0.0
        total_poisoned_loss = 0.0
        num_batches = 0
        for clean_batch, poisoned_batch in zip(clean_loader, poisoned_loader):
            clean_inputs, clean_labels = clean_batch
            poisoned_inputs, poisoned_labels = poisoned_batch
            
            clean_inputs = clean_inputs.to(device, non_blocking=True)
            clean_labels = clean_labels.to(device, non_blocking=True)
            
            poisoned_inputs = poisoned_inputs.to(device, non_blocking=True)
            poisoned_labels = poisoned_labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            clean_outputs = model(clean_inputs)
            clean_loss = criterion(clean_outputs, clean_labels)
            
            poisoned_outputs = model(poisoned_inputs)
            poisoned_loss = criterion(poisoned_outputs, poisoned_labels)
            
            total_loss = clean_loss - alpha * torch.clamp(poisoned_loss, min=0, max=10)
            total_loss.backward()
            
            # Prevent large gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_clean_loss += clean_loss.item()
            total_poisoned_loss += poisoned_loss.item()
            
            num_batches += 1
        avg_clean_loss = total_clean_loss / num_batches
        avg_poisoned_loss = total_poisoned_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs}: Clean Loss = {avg_clean_loss:.4f}, Poisoned Loss = {avg_poisoned_loss:.4f}")

    # Evaluate after unlearning
    evaluate_model(model, clean_loader, device, "Clean")
    evaluate_model(model, poisoned_loader, device, "Poisoned")

    return model

def evaluate_model(model, loader, device, dataset_type):
    """
    Evaluates the model on a given loader with non-blocking transfers.
    """
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    accuracy = 100 * correct / total
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"{dataset_type} Dataset - Accuracy: {accuracy:.2f}%, F1 Score: {f1:.4f}")

if __name__ == "__main__":
    # Test unlearning step with dummy data
    from model import MyModel
    from torch.utils.data import TensorDataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyModel().to(device)
    dummy_inputs = torch.randn(16, 3, 32, 32)
    dummy_labels = torch.randint(0, 10, (16,))
    dummy_dataset = TensorDataset(dummy_inputs, dummy_labels)
    dummy_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=8)
    unlearned_model = unlearn(model, dummy_loader, dummy_loader, alpha=1.0, num_epochs=5, 
                              learning_rate=0.01, device=device)
    print("Unlearning test passed.")