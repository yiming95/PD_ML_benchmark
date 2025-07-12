import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
from PIL import Image
import pandas as pd
import os

import matplotlib.pyplot as plt

# Custom Dataset for loading .nii files
class MRIDataset(Dataset):
    def __init__(self, datapath, transform=None):
        self.nii_files = []
        self.labels = []
        self.tsvfile = None
        for folders in os.listdir(datapath):
            if "tar" not in folders:
                for subjects in os.listdir(datapath + folders + "\\"):
                    if "tsv" in subjects:
                        self.tsvfile = pd.read_csv(datapath + folders + "\\" + subjects, sep='\t', header=0)
                        continue
                    for data in os.listdir(datapath + folders + "\\" + subjects + "\\anat\\"):
                        if "nii" in data:
                            if "control" in subjects:
                                self.labels.append(0)
                            else:
                                self.labels.append(1)
                            self.nii_files.append(datapath + folders + "\\" + subjects + "\\anat\\" + data)

        self.transform = transform
        self.axial_slice = None
        self.coronal_slice = None
        self.sagittal_slice = None

    def crop_black_borders(self,slice_):
        """
        Crop the black borders around an MRI slice.
        Args:
            slice_ (np.ndarray): 2D MRI slice
        Returns:
            np.ndarray: Cropped MRI slice
        """
        # Find non-zero rows and columns
        rows = np.any(slice_, axis=1)
        cols = np.any(slice_, axis=0)

        # Get the indices of non-zero rows and columns
        row_min, row_max = np.where(rows)[0][[0, -1]]
        col_min, col_max = np.where(cols)[0][[0, -1]]

        # Crop the slice
        return slice_[row_min:row_max+1, col_min:col_max+1]


    def __len__(self):
        return len(self.nii_files)

    def __getitem__(self, idx):
        nii_file = self.nii_files[idx]
        label = torch.tensor(self.labels[idx])

        # Load the .nii file
        img = nib.load(nii_file).get_fdata()

        # Extract axial, coronal, and sagittal slices
        self.axial_slice = img[:, :, img.shape[2] // 2]
        self.coronal_slice = img[:, img.shape[1] // 2, :]
        self.sagittal_slice = img[img.shape[0] // 2, :, :]

        self.axial_slice = self.crop_black_borders(self.axial_slice)
        self.coronal_slice = self.crop_black_borders(self.coronal_slice)
        self.sagittal_slice = self.crop_black_borders(self.sagittal_slice)

        # Resize slices to 224x224
        self.axial_slice = np.array(Image.fromarray(self.axial_slice).resize((224, 224)))
        self.coronal_slice = np.array(Image.fromarray(self.coronal_slice).resize((224, 224)))
        self.sagittal_slice = np.array(Image.fromarray(self.sagittal_slice).resize((224, 224)))

        # Normalize each slice independently
        self.axial_slice = (self.axial_slice - np.min(self.axial_slice)) / (np.max(self.axial_slice) - np.min(self.axial_slice))
        self.coronal_slice = (self.coronal_slice - np.min(self.coronal_slice)) / (np.max(self.coronal_slice) - np.min(self.coronal_slice))
        self.sagittal_slice = (self.sagittal_slice - np.min(self.sagittal_slice)) / (np.max(self.sagittal_slice) - np.min(self.sagittal_slice))



        # Stack slices to form a 3-channel image
        img = np.stack((self.axial_slice, self.coronal_slice, self.sagittal_slice), axis=-1)
        # Convert to PIL Image
        #img = Image.fromarray((img * 255).astype(np.uint8))

        if self.transform:
            img = self.transform(img)

        return img, label

"""
transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Path to the data
datapath = 'E:\\pakinsonreviewcasestudy\\MRI\\data\\'
dataset = MRIDataset(datapath, transform=transform)

# Get the first sample
img, label = dataset[3]

# Plot the slices
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title('Axial Slice')
plt.imshow(dataset.axial_slice, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Coronal Slice')
plt.imshow(dataset.coronal_slice, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Sagittal Slice')
plt.imshow(dataset.sagittal_slice, cmap='gray')
plt.axis('off')

plt.show()
"""