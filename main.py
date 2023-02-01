from __future__ import print_function, division
import os
import torch
from skimage import io, transform, color
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from celestiaDataset import *


celestia_dataset = celestiaDataset(root_dir='samples/Set1',transform=transforms.ToTensor())

dataloader = DataLoader(celestia_dataset, batch_size=100,
                        shuffle=True, num_workers=0)