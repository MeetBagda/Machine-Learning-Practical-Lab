import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# --- Data Loading (Already good, just keeping the relevant parts) ---
print("Loading dataset")

transform = transforms.Compose([
    transforms.ToTensor(), # Converts PIL Image or numpy.ndarray to FloatTensor and scales to [0.0, 1.0]
])

train_dataset = datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Number of training batches: {len(train_loader)}")
print(f"Number of test batches: {len(test_loader)}")

# --- Model Definition (Slightly improved) ---
class FashionMNISTModel(nn.Module):
    def __init__(self):
        super(FashionMNISTModel, self).__init__()
        self.fc1 = nn.Linear(in_features=28*28, out_features=256) # Increased neurons
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.5) # Added dropout for regularization
        self.fc2 = nn.Linear(in_features=256, out_features=128) # Added another layer
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=128, out_features=10) # Output layer

    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten (batch_size, 1, 28, 28) -> (batch_size, 784)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x) # Apply dropout
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

print("\n🔹 Initializing model, loss function, and optimizer...")
model = FashionMNISTModel()

# --- Check for GPU ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device) # Move model to GPU if available
print(f"Using device: {device}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) # Slightly reduced learning rate
print("Model initialized.")

# --- Training loop (Crucially using DataLoaders) ---
print("\n🚀 Starting training...")
epochs = 10

for epoch in range(epochs):
    model.train() # Set model to training mode (important for dropout/batchnorm)
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device) # Move data to device

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0) # Accumulate loss
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_dataset)
    epoch_accuracy = correct_train / total_train

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy * 100:.2f}%")

# --- Testing (evaluation) (Crucially using DataLoaders) ---
print("\n✅ Evaluating model on test data...")
model.eval() # Set model to evaluation mode (important for dropout/batchnorm)
correct_test = 0
total_test = 0
with torch.no_grad(): # Disable gradient calculations for evaluation
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device) # Move data to device
        test_outputs = model(images)
        _, predicted = torch.max(test_outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

accuracy = correct_test / total_test
print(f"\n🎯 Test Accuracy: {accuracy * 100:.2f}%")