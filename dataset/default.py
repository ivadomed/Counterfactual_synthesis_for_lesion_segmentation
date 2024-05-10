from torch.utils.data import Dataset
import torchio as tio
import os
from typing import Optional
import argparse
from pathlib import Path


PREPROCESSING_TRANSORMS = tio.Compose([
    tio.RescaleIntensity(out_min_max=(-1, 1)),

    # change here to the desired shape (/!\ must be powers of 2, GPU memory consuption is proportional to the size of the image)
    tio.CropOrPad(target_shape=(16, 256, 256))
])

TRAIN_TRANSFORMS = tio.Compose([
    # add any data desired data aumgentation transforms here and 
    tio.RandomFlip(axes=(1), flip_probability=0.0),
])


class DEFAULTDataset(Dataset):
    def __init__(self, root_dir: str):
        super().__init__()
        self.root_dir = root_dir
        self.preprocessing = PREPROCESSING_TRANSORMS
        self.transforms = TRAIN_TRANSFORMS
        self.file_paths = self.get_data_files()

    def get_data_files(self):
        nifti_file_names = os.listdir(self.root_dir)
        folder_names = []
        for nifti_file_name in nifti_file_names:
            if nifti_file_name.endswith('.nii'):
                folder_names.append(os.path.join(self.root_dir, nifti_file_name))
        print(f'Found {len(folder_names)} files in {self.root_dir}')
        return folder_names

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int): 
        img = tio.ScalarImage(self.file_paths[idx])
        img = self.preprocessing(img)
        img = self.transforms(img)
        return {'data': img.data}
