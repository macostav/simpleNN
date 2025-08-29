import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from tqdm import tqdm

"""
Simple CNN to detect straight lines vs parabolas.
"""

class LineDataset(Dataset):
    def __init__(self, data_dir, transform = None):
        self.data = ImageFolder(root = data_dir, transform = transform) # Automatically labels images based on directory structure

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

class LineClassifier(nn.Module):
    def __init__(self, num_classes = 2):
        super(LineClassifier, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True) # The weights are already set; no need to train them
        self.features = nn.Sequential(*list(self.base_model.children())[:-1]) # Get rid of final layer of model

        enet_output_size = 1280 # The output size of the model
        
        # Make a classifier
        self.classifier = nn.Sequential(
            nn.Flatten(), # Flatten the tensors into a 1D vector
            nn.Linear(enet_output_size, num_classes) # Maps the 1280 output to our two classes; line or parabola
        )

    def forward(self, x):
        # Connect these parts and return the output
        x = self.features(x) # Pattern recognition part
        output = self.classifier(x) # Last layer for classification of images
        return output
    

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # halves spatial dimensions

        # Compute flatten size manually:
        # After conv1 + pool: 77 -> 38
        # After conv2 + pool: 38 -> 19
        # Channels after conv2: 64
        flatten_dim = 64 * 19 * 19  # 23104

        # Fully connected layers
        self.fc1 = nn.Linear(flatten_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 → ReLU → MaxPool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 → ReLU → MaxPool
        x = x.view(x.size(0), -1)             # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)                        # Output logits for 2 classes
        return x

if __name__ == "__main__":
    # Setting up the data
    transform = transforms.ToTensor() # Convert images to tensors

    train_dataset = LineDataset(data_dir = "data/training", transform = transform) # label: 0 is parabola; 1 is straight line
    test_dataset = LineDataset(data_dir = "data/testing", transform = transform)

    train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size =32, shuffle = False) # Don't need shuffling when checking model accuracy

    # Training loop
    num_epochs = 50
    train_losses, test_losses = [], []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Use GPU if possible for faster training

    model = LineClassifier(num_classes = 2)
    #model = SimpleCNN(num_classes=2)
    model.to(device)

    criterion = nn.CrossEntropyLoss() # Loss function; standard for multi-class classification; penalizes high confidence wrong predictions
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Algorithm to move to minimum of loss function; Adam is considered to be one of the best algorithms

    best_accuracy = 0.0
    accuracies = []
    for epoch in range(num_epochs):
        # Training loop
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc='Training loop'):
            # Move inputs and labels to the device
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation loop
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad(): 
            for images, labels in tqdm(test_loader, desc='Validation loop'):
                # Move inputs and labels to the device
                images, labels = images.to(device), labels.to(device)
            
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)

                # Get prediction
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_loss = running_loss / len(test_loader.dataset)
        accuracy = correct / total * 100
        test_losses.append(test_loss)
        accuracies.append(accuracy)

        print(f"Epoch {epoch+1}/{num_epochs} - "
        f"Train loss: {train_loss:.4f}, "
        f"Validation loss: {test_loss:.4f}, "
        f"Accuracy: {accuracy:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "pretrained.pth")

    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Test loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss over Epochs - Pretrained")
    plt.savefig("pretrained_loss_evolution.png")

    plt.figure()
    plt.plot(accuracies)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy over Epochs - Pretrained")
    plt.savefig("pretrained_accuracy_evolution.png")

