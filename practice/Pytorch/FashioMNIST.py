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