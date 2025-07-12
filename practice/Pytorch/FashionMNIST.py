import torch
import torch.nn as nn
import torch.optim as optim
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# load the Fashion-MNIST
print("Loading dataset")

# Define a transformation to convert to tensor and normalize
# This is the recommended way to handle normalization with torchvision datasets
transform = transforms.Compose([
    transforms.ToTensor(), # Converts PIL Image or numpy.ndarray to FloatTensor and scales to [0.0, 1.0]
    # If you need further normalization (e.g., mean/std based for pre-trained models like ResNet), you'd add it here:
    # transforms.Normalize((0.5,), (0.5,)) # Example for [-1, 1] range
])

print("Loading dataset")
train_dataset = datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform # Apply the defined transformations
)

test_dataset = datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Create DataLoaders
# batch_size is a hyperparameter you can tune
batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Number of training batches: {len(train_loader)}")
print(f"Number of test batches: {len(test_loader)}")

# Example of how to get a batch (for checking)
for images, labels in train_loader:
    print(f"Batch images shape: {images.shape}") # Should be [batch_size, 1, 28, 28]
    print(f"Batch images dtype: {images.dtype}") # Should be torch.float32
    print(f"Batch images min: {images.min()}")   # Should be around 0.0
    print(f"Batch images max: {images.max()}")   # Should be around 1.0
    print(f"Batch labels shape: {labels.shape}") # Should be [batch_size]
    print(f"Batch labels dtype: {labels.dtype}") # Should be torch.int64 (or torch.long)
    break # Just get one batch to inspect

x_train = train_dataset.data
y_train = train_dataset.targets
print(f"Features shape (train): {x_train.shape}")
print(f"Label shape (train): {y_train.shape}")

x_test = test_dataset.data
y_test = test_dataset.targets
print(f"Features shape (test): {x_test.shape}")
print(f"Label shape (test): {y_test.shape}")# convert pytorch tensors
print("Converting to pytorch tensors...")
x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test= torch.tensor(y_test,dtype=torch.long)
print("Tensor conversion completed.")

class FashionMNISTModel(nn.Module): # Note: nn.Module, not nn.module
    def __init__(self):
        super(FashionMNISTModel, self).__init__()
        # Input: 28x28 images = 784 features when flattened
        self.fc1 = nn.Linear(in_features=28*28, out_features=128) # First hidden layer
        self.relu = nn.ReLU() # Activation function
        self.fc2 = nn.Linear(in_features=128, out_features=10) # Output layer (10 classes)

    def forward(self, x):
        # Flatten the image: (batch_size, 1, 28, 28) -> (batch_size, 784)
        x = x.view(x.size(0), -1) # x.size(0) is batch_size
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x # For classification, often logits are returned, then softmax is applied in loss/prediction
    
print("\n🔹 Initializing model, loss function, and optimizer...")
model = FashionMNISTModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
print("Model initialized.")

# 🔹 Training loop
print("\n🚀 Starting training...")
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()  # Reset gradients
    outputs = model(x_train)  # Forward pass
    loss = criterion(outputs, y_train)  # Compute loss
    loss.backward()  # Backward pass (compute gradients)
    optimizer.step()  # Update weights

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 🔹 Testing (evaluation)
print("\n✅ Evaluating model on test data...")
with torch.no_grad():
    test_outputs = model(x_test)
    print("Raw model outputs (logits):\n", test_outputs)

    _, predicted = torch.max(test_outputs, 1)
    print("Predicted classes:\n", predicted)
    print("Actual classes:\n", y_test)

    accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    print(f"\n🎯 Test Accuracy: {accuracy * 100:.2f}%")
