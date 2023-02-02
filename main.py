from __future__ import print_function, division
import math
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils

from celestiaDataset import *
from starTModel import *

# Get device for training
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print(f"Using {device} device")

#  Initialise Dataset
grayscale = True

if grayscale:
    trsf = transforms.Compose([transforms.ToTensor(),transforms.Grayscale()])
else:
    trsf = transforms.ToTensor()

dataset = celestiaDataset(root_dir='samples/Set2', transform=trsf)

# Split Dataset into Subsets for training validation and test
trainData, validateData, testData = random_split(dataset,[.7, .15, .15])

# Create dataloaders for each subset
batchSize = 20
trainLoader = DataLoader(trainData, batch_size=batchSize, shuffle=True, num_workers=0)
validateLoader = DataLoader(validateData, batch_size=batchSize, num_workers=0)
testLoader = DataLoader(testData, batch_size=batchSize, num_workers=0)

model = StarTModel(80*120,3)
model.to(device)

# Training loop
nEpochs = 20
nSamples = len(trainData)
nIterations = math.ceil(nSamples/batchSize)
learningRate = 0.001

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),learningRate)

testImage, testCoords = testData[0]
testPred = model(testImage)
print(f'Prediction before training:',
      f'True Latitude: {testCoords[0][0].tolist():05.4f}, predicted: {testPred[0][0].tolist():05.4f}',
      f'True Altitude: {testCoords[0][1].tolist():05.4f}, predicted: {testPred[0][1].tolist():05.4f}',
      f'True Roll:     {testCoords[0][2].tolist():05.4f}, predicted: {testPred[0][2].tolist():05.4f}',
      sep = "\n")

    #   f'True Latitude: {testCoords[0].tolist():3.1f}',
for epoch in range(nEpochs):
    for i, (image,coords) in enumerate(trainLoader):
        image, coords = image.to(device), coords.to(device)
        Pred = model(image)
        
        l = loss(coords, Pred)

        l.backward()

        optimizer.step()

        optimizer.zero_grad()

    print(f'Epoch {epoch+1}/{nEpochs}, loss = {l.item():.3f}')

testPred = model(testImage)
print(f'Prediction after training:',
      f'True Latitude: {testCoords[0][0].tolist():05.4f}, predicted: {testPred[0][0].tolist():05.4f}',
      f'True Altitude: {testCoords[0][1].tolist():05.4f}, predicted: {testPred[0][1].tolist():05.4f}',
      f'True Roll:     {testCoords[0][2].tolist():05.4f}, predicted: {testPred[0][2].tolist():05.4f}',
      sep = "\n")