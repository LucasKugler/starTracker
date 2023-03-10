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
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

#  Initialise Dataset
grayscale = True

if grayscale:
    trsf = transforms.Compose([transforms.ToTensor(),transforms.Grayscale()])
else:
    trsf = transforms.ToTensor()

# dataset = celestiaDataset(root_dir='samples/Set3', transform=trsf, target_transform=scaleLabels())
dataset = celestiaDataset(root_dir='samples/Set4', quarters=True, transform=trsf)

# Split Dataset into Subsets for training validation and test
trainData, testData = random_split(dataset,[.8, .2])

# Create dataloaders for each subset
batchSize = 32
trainLoader = DataLoader(trainData, batch_size=batchSize, shuffle=True, num_workers=0)
testLoader = DataLoader(testData, batch_size=batchSize, num_workers=0)

# Create Model
# model = StarTModel(80*120,80*120,3)
model = StarTConv(quarters=True)
model.to(device)

# Training loop
nEpochs = 20
nSamples = len(trainData)
nIterations = math.ceil(nSamples/batchSize)
learningRate = 0.01

lossFunction = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(),learningRate)
optimizer = torch.optim.Adam(model.parameters())

# testImage, testCoords = testData[0]
# testImage = testImage.reshape(-1,80*120).to(device)
# testCoords = testCoords.to(device)
# testPred = model(testImage)
# print(f'Prediction before training:',
#       f'True Latitude: {testCoords[0][0].tolist():05.4f}, predicted: {testPred[0][0].tolist():05.4f}',
#       f'True Altitude: {testCoords[0][1].tolist():05.4f}, predicted: {testPred[0][1].tolist():05.4f}',
#       f'True Roll:     {testCoords[0][2].tolist():05.4f}, predicted: {testPred[0][2].tolist():05.4f}',
#       sep = "\n")

    #   f'True Latitude: {testCoords[0].tolist():3.1f}',
for epoch in range(nEpochs):
    for i, (images,coords) in enumerate(trainLoader):
        images = images.to(device)
        coords = coords.to(device)
        # Forward
        outputs = model(images)
        loss = lossFunction(coords, outputs)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 10 ==0:
            print(f'Epoch {epoch+1}/{nEpochs}, Iter {i+1}/{nIterations}, Loss = {loss.item():.3f}')

with torch.no_grad():
    errors = []
    for i, (images, coords) in enumerate(testLoader):
        testImages = images.to(device)
        testCoords = coords.to(device)
        testOutputs = model(testImages)
        errors[i] = lossFunction(testCoords,testOutputs)
    print(f'Accuracy = {np.array(errors).mean()}')




# testPred = model(testImage)
# print(f'Prediction after training:',
#       f'True Latitude: {testCoords[0][0].tolist():05.4f}, predicted: {testPred[0][0].tolist():05.4f}',
#       f'True Altitude: {testCoords[0][1].tolist():05.4f}, predicted: {testPred[0][1].tolist():05.4f}',
#       f'True Roll:     {testCoords[0][2].tolist():05.4f}, predicted: {testPred[0][2].tolist():05.4f}',
#       sep = "\n")