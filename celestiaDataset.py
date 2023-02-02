from __future__ import print_function, division
import os
import torch
from skimage import io
import numpy as np
from torch.utils.data import Dataset

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
        # Create a sorted list of all files in directory
        self.sampleList = sorted(os.listdir(root_dir))

    def __len__(self):
        return len(self.sampleList)

    def __getitem__(self, idx):
        # Convert tensor to list if necessary
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get path of image file and read it
        imgFullName = os.path.join(self.root_dir,
                                self.sampleList[idx])
        image = io.imread(imgFullName)

        # Slice the file name to get the coordinates of the sample
        coords = self.sampleList[idx][:-4].split("_")[1:4]
        coords = np.array([coords], dtype = int)

        # Apply transformation if specified
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, coords


# class ToGrayscale(object):
#     """Convert image to grayscale"""

#     def __call__(self, sample):
#         image, coords = sample["image"],  sample["coords"]
#         return {'image': color.rgb2gray(image),
#                 'coords': coords}
    
# class ToTensor(object):
#     """Convert ndarrays in sample to Tensors."""

#     def __call__(self, sample):
#         image, coords = sample["image"],  sample["coords"]

#         # swap color axis because
#         # numpy image: H x W x C
#         # torch image: C x H x W
#         image = image.transpose((2, 0, 1))
#         return {'image': torch.from_numpy(image),
#                 'coords': torch.from_numpy(coords)}