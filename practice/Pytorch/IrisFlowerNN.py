import torch
import torch.nn as nn
import torch.optim as optim
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 🔹 Load the Iris dataset
print("🔹 Loading Iris dataset...")
iris = load_iris()
x = iris.data
y = iris.target
print(f"Features shape: {x.shape}")
print(f"Labels shape: {y.shape}")
print(f"Target classes: {iris.target_names}")

# 🔹 Normalize features
print("\n🔹 Normalizing feature data...")
scaler = StandardScaler()
x = scaler.fit_transform(X=x)

# 🔹 Split into training and testing sets
print("\n🔹 Splitting data into train and test sets...")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")

# 🔹 Convert numpy arrays to PyTorch tensors
print("\n🔹 Converting to PyTorch tensors...")
x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)
print("Tensor conversion completed.")

# 🔹 Define the Neural Network architecture
class IrisNN(nn.Module):
    def __init__(self):
        super(IrisNN, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

print("\n🔹 Initializing model, loss function, and optimizer...")
model = IrisNN()
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
