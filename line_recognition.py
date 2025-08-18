import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

import matplotlib.pyplot as plt # For data viz
import pandas as pd
import numpy as np
import sys
from tqdm.notebook import tqdm

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
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        enet_output_size = 1280
        
        # Make a classifier
        self.classifier = nn.Sequential(
            nn.Flatten(), # Flatten the tensors into a 1D vector
            nn.Linear(enet_output_size, num_classes) # Maps the 1280 output to our two classes
        )

    def forward(self, x):
        # Connect these parts and return the output
        x = self.features(x) # Pattern recognition part
        output = self.classifier(x) # Last layer for classification of images
        return output

if __name__ == "__main__":
    # Setting up the data
    transform = transforms.ToTensor() # Convert images to tensors
    dataset = LineDataset(data_dir = "data/training", transform = transform) # label: 0 is parabola; 1 is straight line
    dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)

    # Setting up the model
    model = LineClassifier(num_classes = 2)

    # Training the model
    criterion = nn.CrossEntropyLoss() # Loss function; standard for multi-class classification; penalizes high confidence wrong predictions
    optimizer = optim.Adam(model.parameters(), lr=0.001) # !!! LOOK AT THIS OPTIMIZER ALGORITHM

