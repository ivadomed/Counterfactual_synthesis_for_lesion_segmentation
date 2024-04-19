""" Taken and adapted from https://github.com/spine-generic/data-multi-subject """

import csv
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
from pathlib import Path
from skimage.transform import resize
from nilearn import surface
import nibabel as nib
from skimage import exposure
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


class SPINEGENERICDataset(Dataset):
    def __init__(self, root_dir, train=True, is_flip=False, augmentation=True):
        self.augmentation = augmentation
        self.train = train
        self.root = os.path.join(
                root_dir, '/data-multi-subject/')
        self.imgtype = imgtype
        self.is_flip = is_flip
        self.image_dataset, seg_dataset = self.get_dataset()

    def retrieve_paths(root):
        image_paths = []
        seg_paths = []
        for p in Path(root).rglob('*.nii.gz'):
            if 'derivatives' not in str(p) and "code" not in str(p) and ".git" not in str(p):
                seg_path = str(p).replace('.nii.gz', 'softseg.nii.gz').replace('data-multi-subject','data-multi-subject\derivatives\labels_softseg')
                #check id seg_path, relative path exists
                if not os.path.exists(seg_path):
                    print(f"Warning: {seg_path} does not exist. Skipping...")
                    continue
                image_paths.append(str(p))
                seg_paths.append(seg_path)
        paths_df = pd.DataFrame({'image': image_paths, 'seg': seg_paths})
        return paths_df

    def get_dataset(self):
        paths_df = retrieve_paths(self.root)
        train_paths, val_paths = train_test_split(paths_df, test_size=0.2, random_state=42)
        if self.train:
            return train_paths
        else:
            return val_paths

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image_path = self.image_dataset.iloc[index]['image']
        seg_path = self.image_dataset.iloc[index]['seg']


        img = nib.load(img_path).get_fdata()
        seg = nib.load(seg_path).get_fdata()

        # only keep the slices with non nul segmentations
        seg_sum = np.sum(seg, axis=(1,2))
        indices = np.where(seg_sum != 0)[0]
        # delete all doubles in the list
        indices = list(dict.fromkeys(indices))
        med = int(np.median(indices))
        selected_frame = list(range(med-16,med+16))
        # if the list is not continuous, print a warning
        img = np.take(img, selected_frame, axis=0)
        if self.augmentation:
            if sel.train:
                img = self.train_transform(img)
            else:
                img = self.val_transform(img)
        imageout = torch.from_numpy(img).float()
        imageout.unsqueeze_(0)

        return {'data': imageout}
