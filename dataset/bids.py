"""
This dataset expect the BIDS format for the data. 
It will look for the .nii.gz files in the root_dir which are in the sub-## folders
It can take a list of contrasts to filter the files as input with the contrasts argument
If derivatives is set to True, it will look for the derivatives in the derivatives folder
"""


from torch.utils.data import Dataset
import torchio as tio
from monai.transforms import (
    LoadImaged,
    LoadImage,
    Compose,
    RandSpatialCropd,
    RandShiftIntensityd,
    RandRotated,
    ScaleIntensityRanged,
    ResizeWithPadOrCropd,
    EnsureChannelFirstd,
    ToTensord,
)
import os
from typing import Optional
import argparse
from pathlib import Path
import time



class BIDSDataset(Dataset):
    def __init__(self, root_dir: str, is_VQGAN: bool = False, contrasts = None, derivatives = False):
        super().__init__()
        self.root_dir = root_dir
        self.is_VQGAN = is_VQGAN
        self.contrasts = contrasts
        self.derivatives = derivatives
        self.derivatives_dicts = []
        self.file_paths = self.get_data_files()
        
    
    def find_derivatives(self, file, derivatives_files_paths):
        self.derivatives_dicts.append({})
        if file in derivatives_files_paths:
            for derivative_dict in derivatives_files_paths[file]:
                derivative_key = derivative_dict['derivative_key']
                derivative_path = derivative_dict['derivate_path']
                self.derivatives_dicts[-1][derivative_key] = derivative_path

    def get_data_files(self):
        files_paths = []
        if self.derivatives:
            derivatives_files_paths = {}
            for root_deriv, dirs_deriv, files_deriv in os.walk(self.root_dir+'/derivatives'):
                for file_deriv in files_deriv:
                    if file_deriv.endswith('.nii.gz'):
                        derivative_key = file_deriv.split('_')[-1].split('.')[0]
                        file_path = file_deriv.split('_'+derivative_key)[0]+'.nii.gz'
                        if file_path in derivatives_files_paths:
                            derivatives_files_paths[file_path].append({'derivative_key':derivative_key, 'derivate_path':os.path.join(root_deriv, file_deriv)})
                        else:
                            derivatives_files_paths[file_path] = [{'derivative_key':derivative_key, 'derivate_path':os.path.join(root_deriv, file_deriv)}]
        for root, dirs, files in os.walk(self.root_dir):
            if self.root_dir +'/sub-' in root or self.root_dir +'\\sub-' in root:
                for file in files:
                    if file.endswith('.nii.gz'):
                        if len(self.contrasts) > 0:
                            for contrast in self.contrasts:
                                if contrast in file:
                                    files_paths.append(os.path.join(root, file))
                                    if self.derivatives:
                                        self.find_derivatives(file, derivatives_files_paths)
                        else:
                            files_paths.append(os.path.join(root, file))
                            if self.derivatives:
                                self.find_derivatives(file, derivatives_files_paths)

        print(f'Found {len(files_paths)} files in {self.root_dir}')
        return files_paths
    
    def get_sample_dict(self, idx: int):
        sample_dict = {}
        sample_dict['data'] = self.file_paths[idx]
        if self.derivatives:
            sample_dict.update(self.derivatives_dicts[idx])
        return sample_dict
    
    
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        sample_dict = self.get_sample_dict(idx)
        keys = list(sample_dict.keys())

        img = LoadImage()(sample_dict['data'])
        img_np = img.numpy()
        a_min = img_np.min().astype(float)
        a_max = img_np.max().astype(float)


        TRAIN_VQGAN_TRANSORMS = Compose([
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            ScaleIntensityRanged(keys=['data'], a_min=a_min, a_max=a_max, b_min=-1, b_max=1),
            # change here to the desired shape (/!\ must be powers of 2, GPU memory consuption is proportional to the size of the image)
            ResizeWithPadOrCropd(keys=keys, spatial_size=[32, 256, 256], mode="replicate"),
            RandSpatialCropd(keys=keys,  roi_size=[16, 256, 256], random_size=False),
            RandShiftIntensityd(keys=keys, offsets=0.1, prob=0.5),
            RandRotated(keys=keys, range_x=0.3, range_y=0.3, range_z=0.3, prob=0.5),
            ToTensord(keys=keys),
        ])


        # For the DDPM training, we can use larger images since the decoding can be devided in several patches
        TRAIN_DDPM_TRANSFORMS = Compose([
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            ScaleIntensityRanged(keys=['data'], a_min=a_min, a_max=a_max, b_min=-1, b_max=1),
            # change here to the desired shape (/!\ must be powers of 2, GPU memory consuption is proportional to the size of the image)
            ResizeWithPadOrCropd(keys=keys, spatial_size=[32, 256, 256], mode="replicate"),
            RandShiftIntensityd(keys=keys, offsets=0.1, prob=0.5),
            RandRotated(keys=keys, range_x=0.3, range_y=0.3, range_z=0.3, prob=0.5),
            ToTensord(keys=keys),
        ])


        if self.is_VQGAN:
            sample_dict = TRAIN_VQGAN_TRANSORMS(sample_dict)
        else:
            sample_dict = TRAIN_DDPM_TRANSFORMS(sample_dict)
        
        return sample_dict
