import torch
import torch.nn as nn
import torchvision.datasets
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# load the Fashion-MNIST
print("Loading dataset")
fashionmnist = torchvision.datasets.FashionMNIST(train=True)
x=fashionmnist.data
y=fashionmnist.targets
print(f"Features shape : {x.shape}")
print(f"Label shape : {y.shape}")
print(f"Target Classes : {fashionmnist.targets}")

# Normalize the features
x = x.float() / 255.0 # Convert to float and then normalize to [0, 1]
print(f"Features after normalization (min): {x.min()}")
print(f"Features after normalization (max): {x.max()}")

# splitting
print("Splittin the dataset")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(f"Train shape: {x_train.shape}, Test Shape: {x_test.shape}")

# convert pytorch tensors
print("Converting to pytorch tensors...")
x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test= torch.tensor(y_test,dtype=torch.long)
print("Tensor conversion completed.")

# Define NN

# class FashionMNIST(nn.module):
#     def __init__(self):
#         super(FashionMNIST, self).__init__()
        # self.fc1 = nn.Linear()