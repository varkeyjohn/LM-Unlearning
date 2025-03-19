import torch
import torch.nn as nn
from torchsummary import summary

class MyModel(nn.Module):
    def __init__(self, conv_filters, fc_hidden_size, num_classes=10):
        """
        Initialize MyModel with variable convolutional layers and fully connected layer size.
        
        Args:
            conv_filters (list): List of integers specifying the number of filters per conv layer.
            fc_hidden_size (int): Number of neurons in the hidden fully connected layer.
            num_classes (int): Number of output classes (default: 10).
        """
        super(MyModel, self).__init__()
        
        # Ensure at least one convolutional layer
        assert len(conv_filters) >= 1, "At least one convolutional layer is required"
        
        # Calculate final spatial size after pooling (input 32x32, halved each layer)
        n_layers = len(conv_filters)
        spatial_size = 32 // (2 ** n_layers)
        if spatial_size < 1:
            raise ValueError("Too many convolutional layers for input size 32x32")
        
        # Build convolutional layers
        layers = []
        in_channels = 3  # RGB input
        for out_channels in conv_filters:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2, 2))
            in_channels = out_channels
        self.conv_layers = nn.Sequential(*layers)
        
        # Compute flattened size for fully connected layers
        flattened_size = conv_filters[-1] * spatial_size * spatial_size
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, fc_hidden_size),
            nn.ReLU(),
            nn.Linear(fc_hidden_size, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, 3, 32, 32).
        
        Returns:
            Tensor: Output tensor of shape (batch_size, num_classes).
        """
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Test the model
if __name__ == "__main__":
    # Example: 2 conv layers, similar to a simple CIFAR-10 model
    model = MyModel(conv_filters=[16, 32], fc_hidden_size=256, num_classes=10)
    test_input = torch.randn(1, 3, 32, 32)
    output = model(test_input)
    assert output.shape == (1, 10), "Model output shape mismatch"
    print("Model test passed!")
    summary(model.cuda(), input_size=(3,32,32))