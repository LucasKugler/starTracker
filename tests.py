from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform, color
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from celestiaDataset import *


celestia_dataset = celestiaDataset(root_dir='samples/Set1')

fig = plt.figure()

for i in range(len(celestia_dataset)):
    sample = celestia_dataset[i]

    print(i, sample['image'].shape, sample['coords'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    plt.imshow(sample["image"])

    if i == 3:
        plt.show()
        break



transformed_dataset = celestiaDataset(root_dir='samples/Set1',transform=ToTensor())


for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]

    print(i, sample['image'].size(), sample['coords'].size())

    if i == 3:
        break