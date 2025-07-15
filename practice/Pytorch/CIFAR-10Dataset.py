import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Define the Neural Network Architecture (Simple CNN) ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        # Convolutional Layer 1
        # Input channels: 3 (for RGB images)
        # Output channels: 16
        # Kernel size: 3x3
        # Padding: 1 (to keep the same spatial dimensions after convolution)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        # Batch Normalization 1 (helps stabilize and speed up training)
        self.bn1 = nn.BatchNorm2d(16)
        # ReLU activation (introduces non-linearity)
        self.relu1 = nn.ReLU()
        # Max Pooling 1 (reduces spatial dimensions, extracts dominant features)
        # Kernel size: 2x2, stride: 2 (halves the dimensions)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Layer 2
        # Input channels: 16
        # Output channels: 32
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Layer 3
        # Input channels: 32
        # Output channels: 64
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layer (Classifier)
        # The input size to the first fully connected layer depends on the output
        # dimensions of the last pooling layer.
        # For CIFAR-10 (32x32 images):
        # After conv1+pool1: 32/2 = 16x16
        # After conv2+pool2: 16/2 = 8x8
        # After conv3+pool3: 8/2 = 4x4
        # So, the flattened feature map size is 64 channels * 4 * 4 = 1024.
        self.fc = nn.Linear(64 * 4 * 4, num_classes) # Adjusted for 32x32 input images

    def forward(self, x):
        # Pass through the first convolutional block
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        # Pass through the second convolutional block
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        # Pass through the third convolutional block
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))

        # Flatten the output for the fully connected layer
        # -1 infers the batch size
        x = x.view(x.size(0), -1) # x.size(0) is the batch size

        # Pass through the fully connected layer
        x = self.fc(x)
        return x


# --- 2. Data Preparation (Transforms and Loaders) ---
def prepare_data(data_dir, image_size=32, batch_size=32): # Default image_size changed to 32 for CIFAR-10
    # Define transformations for the training data
    # RandomCrop and RandomHorizontalFlip are common augmentations for CIFAR-10
    # ToTensor converts PIL Image or numpy.ndarray to torch.FloatTensor
    # Normalize with CIFAR-10 specific mean and standard deviation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), # Pad and then randomly crop back to 32x32
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # CIFAR-10 mean/std
    ])

    # Define transformations for the validation/test data (no augmentation)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # CIFAR-10 mean/std
    ])

    # Load CIFAR-10 datasets. 'download=True' will download the dataset if not present.
    # 'root' specifies where the data will be stored.
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    val_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # CIFAR-10 has 10 classes
    num_classes = 10
    class_names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print(f"Found {num_classes} classes: {class_names}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    return train_loader, val_loader, num_classes, class_names

