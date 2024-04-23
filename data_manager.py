import numpy as np
import numpy.random as rd
import nibabel as nib
import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from monai.data import Dataset, DataLoader, CacheDataset
from monai.transforms import (
    Compose,
    RandFlip,
    Resize,
    RandRotate90,
    RandRotate,
    RandShiftIntensity,
    ToTensor,
    RandSpatialCrop,
    LoadImage,
    SqueezeDim,
    RandRotate,
    RandSimulateLowResolution,
    ScaleIntensity,
    SpatialPad,
)
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a custom dataset class
class Auto_encoder_Dataset(Dataset):
    def __init__(self, img_paths, seg_paths, transform=None):
        self.img_paths = img_paths
        self.seg_paths = seg_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        seg_path = self.seg_paths[idx]

        img = nib.load(img_path).get_fdata()
        seg = nib.load(seg_path).get_fdata()

        # only keep the slices with non nul segmentations
        seg_sum = np.sum(seg, axis=(1,2))
        indices = np.where(seg_sum != 0)[0]
        # delete all doubles in the list
        indices = list(dict.fromkeys(indices))
        med = int(np.median(indices))
        selected_slice = rd.randint(med-16, high=med+16)
        #selected_slice = med
        # if the list is not continuous, print a warning
        img = np.take(img, selected_slice, axis=0)
        seg = np.take(seg, selected_slice, axis=0)
       
        # add a channel dimension
        img = np.expand_dims(img, axis=0)
        seg = np.expand_dims(seg, axis=0)
        if self.transform:
            img = self.transform(img)
            seg = self.transform(seg)
        
        return img


def fetch_all_T2w_paths(source_folder):
    img_paths = []
    seg_paths = []

    for p in Path(source_folder).rglob('*.nii.gz'):

        if 'T2w' in str(p) and 'derivatives' not in str(p) and "code" not in str(p) and ".git" not in str(p):
            
            seg_path = str(p).replace('.nii.gz', '_softseg.nii.gz').replace('data-multi-subject','data-multi-subject/derivatives/labels_softseg')
            #check id seg_path, relative path exists
            if not os.path.exists(seg_path):
                print(f"Warning: {seg_path} does not exist. Skipping...")
                continue
            seg_paths.append(seg_path)
            img_paths.append(str(p))
    return img_paths, seg_paths




def dataset_splitter(img_paths, seg_paths, train_ratio=0.8, random_seed=42):
    """ Split the dataset into training and validation sets based on the specified ratio and random seed."""
    pd_data = pd.DataFrame({"image_path": img_paths, "segs": seg_paths})
    pd_train_data, pd_val_data = train_test_split(pd_data, train_size=train_ratio, random_state=random_seed)
    pd_train_data.reset_index(drop=True, inplace=True)
    pd_val_data.reset_index(drop=True, inplace=True)

    return pd_train_data, pd_val_data


def paths_to_Dataset(pd_data, val = False):
    """ Convert the file paths to a custom dataset object."""
    if val:
        transform = Compose([
        # select a random slice from the volume shaped : (32, 256, 256)
        RandSpatialCrop(roi_size=(256, 256), random_center=True, random_size=False),
        SpatialPad( spatial_size=(256, 256)),
        ScaleIntensity(minv=0.0, maxv=1.0),
        ToTensor(),
    ])
        
    
    else:
        
        transform = Compose([
        # select a random slice from the volume shaped : (32, 256, 256)
        RandSpatialCrop(roi_size=(256, 256), random_center=True, random_size=False),
        SpatialPad( spatial_size=(256, 256)),
        ScaleIntensity(minv=0.0, maxv=1.0),
        RandRotate90(prob=0.5),
        RandFlip(prob=0.5),
        # normalise between 0 and 1
        RandRotate(range_x=0.3, range_y=0.3, prob=0.2),
        ToTensor(),
        ])
        
    dataset = Auto_encoder_Dataset(pd_data["image_path"], pd_data["segs"], transform=transform)
    return dataset