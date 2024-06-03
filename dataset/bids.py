"""
This dataset expect the BIDS format for the data. 
It will look for the .nii.gz files in the root_dir which are in the sub-## folders
it can take a list of contrasts to filter the files as input
"""



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
    RandRotate(range_x=0.3, range_y=0.3, range_z=0.3, prob=0.5),
])


class BIDSDataset(Dataset):
    def __init__(self, root_dir: str, is_VQGAN: bool = False, contrasts = None):
        super().__init__()
        self.root_dir = root_dir
        self.preprocessing = PREPROCESSING_TRANSORMS
        if is_VQGAN:
            self.transforms = TRAIN_VQGAN_TRANSFORMS
        else:
            self.transforms = TRAIN_DDPM_TRANSFORMS
        self.contrasts = contrasts
        self.file_paths = self.get_data_files()

    def get_data_files(self):
        nifti_file_names = os.listdir(self.root_dir)
        files_names = []
        for root, dirs, files in os.walk(self.root_dir):
            if self.root_dir +'/sub-' in root or self.root_dir +'\\sub-' in root:
                for file in files:
                    if file.endswith('.nii.gz'):
                        if self.contrasts is not None:
                            for contrast in self.contrasts:
                                if contrast in file:
                                    files_names.append(os.path.join(root, file))
        print(f'Found {len(files_names)} files in {self.root_dir}')

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int): 
        img = tio.ScalarImage(self.file_paths[idx])
        img = self.preprocessing(img)
        img.data = self.transforms(img.data)
        return {'data': img.data}
