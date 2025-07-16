import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

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

# --- 3. Training Function ---
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    model.to(device) # Move model to the specified device (CPU or GPU)

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device) # Move data to device

            optimizer.zero_grad() # Zero the parameter gradients

            outputs = model(inputs) # Forward pass
            loss = criterion(outputs, labels) # Calculate loss
            loss.backward() # Backward pass (compute gradients)
            optimizer.step() # Update model parameters

            running_loss += loss.item() * inputs.size(0) # Accumulate loss

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # --- Validation Step ---
        model.eval() # Set model to evaluation mode
        val_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad(): # Disable gradient calculation for validation
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs.data, 1) # Get the class with the highest probability
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = 100 * correct_predictions / total_samples
        val_losses.append(epoch_val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {epoch_train_loss:.4f}, '
              f'Val Loss: {epoch_val_loss:.4f}, '
              f'Val Accuracy: {val_accuracy:.2f}%')

    print('Finished Training')
    return train_losses, val_losses, val_accuracies

# --- 4. Comprehensive Model Evaluation and Visualization ---
def evaluate_model_comprehensive(model, val_loader, class_names, device='cpu'):
    """
    Comprehensive evaluation of the model with detailed visualizations
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

def plot_class_accuracy(y_true, y_pred, class_names):
    """
    Plot per-class accuracy
    """
    cm = confusion_matrix(y_true, y_pred)
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_names, class_accuracy * 100)
    plt.title('Per-Class Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, acc in zip(bars, class_accuracy):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{acc*100:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def visualize_predictions_with_confidence(model, val_loader, class_names, device='cpu', num_samples=16):
    """
    Visualize sample predictions with confidence scores
    """
    model.eval()
    dataiter = iter(val_loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    
    outputs = model(images)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    confidences, predicted = torch.max(probabilities, 1)
    
    # Function to unnormalize and show an image
    def imshow_with_confidence(inp, title=None, confidence=None):
        """Display image with confidence score"""
        inp = inp.cpu().numpy().transpose((1, 2, 0))
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        return inp
    
    # Display predictions
    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    fig.suptitle('Model Predictions with Confidence Scores', fontsize=16)
    
    for idx in range(min(num_samples, len(images))):
        row = idx // 4
        col = idx % 4
        
        img = imshow_with_confidence(images[idx])
        axes[row, col].imshow(img)
        
        pred_class = class_names[predicted[idx]]
        true_class = class_names[labels[idx]]
        confidence = confidences[idx].item() * 100
        
        # Color code: green for correct, red for incorrect
        color = 'green' if predicted[idx] == labels[idx] else 'red'
        
        axes[row, col].set_title(f'Pred: {pred_class}\nTrue: {true_class}\nConf: {confidence:.1f}%', 
                                color=color, fontsize=10)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_top_predictions_analysis(probabilities, labels, predictions, class_names):
    """
    Analyze and plot top prediction confidence distribution
    """
    correct_mask = (predictions == labels)
    correct_confidences = np.max(probabilities[correct_mask], axis=1)
    incorrect_confidences = np.max(probabilities[~correct_mask], axis=1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(correct_confidences, bins=20, alpha=0.7, label='Correct Predictions', color='green')
    plt.hist(incorrect_confidences, bins=20, alpha=0.7, label='Incorrect Predictions', color='red')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Confidence Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    accuracy = len(correct_confidences) / (len(correct_confidences) + len(incorrect_confidences))
    metrics = [
        f'Overall Accuracy: {accuracy*100:.2f}%',
        f'Avg Correct Confidence: {np.mean(correct_confidences)*100:.2f}%',
        f'Avg Incorrect Confidence: {np.mean(incorrect_confidences)*100:.2f}%',
        f'Total Samples: {len(labels)}'
    ]
    
    plt.text(0.1, 0.7, '\n'.join(metrics), fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title('Model Performance Summary')
    
    plt.tight_layout()
    plt.show()

# --- 5. Main Execution Block ---
if __name__ == "__main__":
    # Check for GPU availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Configuration ---
    # Data will be downloaded to this directory
    DATA_DIR = './data_cifar10' # Changed to a new directory for CIFAR-10
    IMAGE_SIZE = 32 # CIFAR-10 images are 32x32 pixels
    BATCH_SIZE = 16 # Number of images processed at once
    NUM_EPOCHS = 5 # Number of times to iterate over the entire dataset
    LEARNING_RATE = 0.001 # Step size for the optimizer

    # Prepare data loaders (will download CIFAR-10 if not present)
    train_loader, val_loader, num_classes, class_names = prepare_data(DATA_DIR, IMAGE_SIZE, BATCH_SIZE)

    # Initialize the model
    model = SimpleCNN(num_classes=num_classes)

    # Define Loss Function and Optimizer
    # CrossEntropyLoss is suitable for multi-class classification
    criterion = nn.CrossEntropyLoss()
    # Adam optimizer is a good general-purpose optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    print("\n--- Starting Model Training ---")
    train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, device
    )

    # --- Plotting Training Results ---
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='orange')
    plt.title('Validation Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # --- Comprehensive Model Evaluation and Visualization ---
    print("\n--- Comprehensive Model Evaluation ---")
    
    # Get comprehensive predictions
    all_predictions, all_labels, all_probabilities = evaluate_model_comprehensive(
        model, val_loader, class_names, device
    )
    
    # Plot confusion matrix
    print("Generating confusion matrix...")
    plot_confusion_matrix(all_labels, all_predictions, class_names)
    
    # Plot per-class accuracy
    print("Generating per-class accuracy chart...")
    plot_class_accuracy(all_labels, all_predictions, class_names)
    
    # Visualize sample predictions with confidence
    print("Generating sample predictions with confidence scores...")
    visualize_predictions_with_confidence(model, val_loader, class_names, device)
    
    # Plot confidence analysis
    print("Generating confidence distribution analysis...")
    plot_top_predictions_analysis(all_probabilities, all_labels, all_predictions, class_names)
    
    # Print detailed classification report
    print("\n--- Detailed Classification Report ---")
    print(classification_report(all_labels, all_predictions, target_names=class_names))

    # --- Simple Test/Prediction Example (using a few validation images) ---
    print("\n--- Making a few predictions on validation set ---")
    model.eval() # Set model to evaluation mode
    dataiter = iter(val_loader)
    images, labels = next(dataiter) # Get a batch of images and labels
    images, labels = images.to(device), labels.to(device)

    # Get predictions
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)

    print('GroundTruth: ', ' '.join(f'{class_names[labels[j]]:5s}' for j in range(4)))
    print('Predicted:   ', ' '.join(f'{class_names[predicted[j]]:5s}' for j in range(4)))

    # Display some images with predictions (requires matplotlib)
    # Function to unnormalize and show an image
    def imshow(inp, title=None):
        """Imshow for Tensor."""
        inp = inp.cpu().numpy().transpose((1, 2, 0))
        # CIFAR-10 specific mean and std for unnormalization
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated

    fig = plt.figure(figsize=(10, 8))
    for j in range(min(4, len(images))): # Display first 4 images
        ax = fig.add_subplot(1, 4, j + 1, xticks=[], yticks=[])
        imshow(images.cpu()[j], title=f'{class_names[predicted[j]]} (True: {class_names[labels[j]]})')
    plt.show()

    # --- Saving and Loading the Model ---
    model_path = 'cifar10_image_classifier_model.pth' # Changed model filename
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    # To load the model later:
    # loaded_model = SimpleCNN(num_classes=num_classes)
    # loaded_model.load_state_dict(torch.load(model_path))
    # loaded_model.eval() # Set to evaluation mode after loading
    # print(f"Model loaded from {model_path}")

