from __future__ import print_function, division
import os
import torch
from skimage import io, transform, color
import numpy as np
from torch.utils.data import Dataset, DataLoader

class celestiaDataset(Dataset):
    """Celestia dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.sampleList = sorted(os.listdir(root_dir))

    def __len__(self):
        return len(self.sampleList)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        imgFullName = os.path.join(self.root_dir,
                                self.sampleList[idx])
        image = io.imread(imgFullName)
        coords = self.sampleList[idx].split("_")[1:4]
        coords = np.array([coords])
        sample = {'image': image, 'coords': coords}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToGrayscale(object):
    """Convert image to grayscale"""

    def __call__(self, sample):
        image, coords = sample["image"],  sample["coords"]
        return {'image': color.rgb2gray(image),
                'coords': coords}
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, coords = sample["image"],  sample["coords"]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose(2, 0, 1)
        return {'image': torch.from_numpy(image),
                'coords': torch.from_numpy(coords)}