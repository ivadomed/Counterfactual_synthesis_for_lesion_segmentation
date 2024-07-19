from torch.utils.data import Dataset
import torchio as tio
from monai.transforms import (
    Compose,
    RandSpatialCrop,
    RandShiftIntensity,
    RandRotate,
)
import os
from typing import Optional
import argparse
from pathlib import Path


PREPROCESSING_TRANSORMS = tio.Compose([
    tio.RescaleIntensity(out_min_max=(-1, 1)),

    # change here to the desired shape (/!\ must be powers of 2, GPU memory consuption is proportional to the size of the image)
    tio.CropOrPad(target_shape=(32, 256, 256))
])

# Since the training of the VQGAN is more memory intensive, one shall want to use smaller images
TRAIN_VQGAN_TRANSFORMS = Compose([
    RandSpatialCrop((16, 256, 256), random_size=False),
    RandShiftIntensity(offsets=0.1, prob=0.5),
    RandRotate(range_x=0.3, range_y=0.3, range_z=0.3, prob=0.5),
])

# For the DDPM training, we can use larger images since the decoding can be devided in several patches
TRAIN_DDPM_TRANSFORMS = Compose([
    RandShiftIntensity(offsets=0.1, prob=0.5),
    RandRotate(range_x=0.3, range_y=0.0, range_z=0.0, prob=0.5),
])


class DEFAULTDataset(Dataset):
    def __init__(self, root_dir: str, is_VQGAN: bool = False):
        super().__init__()
        self.root_dir = root_dir
        self.preprocessing = PREPROCESSING_TRANSORMS
        if is_VQGAN:
            self.transforms = TRAIN_VQGAN_TRANSFORMS
        else:
            self.transforms = TRAIN_DDPM_TRANSFORMS
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
        img.data = self.transforms(img.data)
        return {'data': img.data}
