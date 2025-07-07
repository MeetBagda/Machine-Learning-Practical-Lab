import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Load dataset
iris = load_iris()
X = iris.data  # 4 features
y = iris.target  # 3 classes: 0 (Setosa), 1 (Versicolor), 2 (Virginica)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)  # use long for classification
y_test = torch.tensor(y_test, dtype=torch.long)


class IrisNN(nn.Module):
    def __init__(self):
        super(IrisNN, self).__init__()
        self.fc1 = nn.Linear(4, 10)  # 4 input features → 10 hidden
        self.fc2 = nn.Linear(10, 3)  # 10 hidden → 3 output classes

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # raw scores (logits) → handled by loss fn
        return x
