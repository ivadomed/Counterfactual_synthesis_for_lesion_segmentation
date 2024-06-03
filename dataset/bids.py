"""
This dataset expect the BIDS format for the data. 
It will look for the .nii.gz files in the root_dir which are in the sub-## folders
It can take a list of contrasts to filter the files as input with the contrasts argument
If derivatives is set to True, it will look for the derivatives in the derivatives folder
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
    def __init__(self, root_dir: str, is_VQGAN: bool = False, contrasts = None, derivatives = False):
        super().__init__()
        self.root_dir = root_dir
        self.preprocessing = PREPROCESSING_TRANSORMS
        if is_VQGAN:
            self.transforms = TRAIN_VQGAN_TRANSFORMS
        else:
            self.transforms = TRAIN_DDPM_TRANSFORMS
        self.contrasts = contrasts
        self.file_paths = self.get_data_files()
        self.derivatives = derivatives
        self.derivatives_dicts = []
    
    def find_derivatives(self, file):
        self.derivatives_dicts.append({})
        for root_deriv, dirs_deriv, files_deriv in os.walk(self.root_dir+'/derivatives'):
            for file_deriv in files_deriv:
                if file_deriv.endswith('.nii.gz'):
                    derivative_key = file_deriv.split('_')[-1].split('.')[0]
                    self.derivatives_dicts[-1][derivative_key] = os.path.join(root_deriv, file_deriv)

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
                        else:
                            files_names.append(os.path.join(root, file))
                        if self.derivatives:
                                        self.find_derivatives(file)

        print(f'Found {len(files_names)} files in {self.root_dir}')
    

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        data_dict = {}
        img = tio.ScalarImage(self.file_paths[idx])
        img = self.preprocessing(img)

        # Apply transforms to the main image
        img_data = img.data
        img_data = img_data.unsqueeze(0)  # Add a batch dimension
        transformed_img_data = self.transforms(img_data)
        transformed_img_data = transformed_img_data.squeeze(0)  # Remove the batch dimension
        data_dict['data'] = transformed_img_data

        if self.derivatives:
            for derivative_key, derivative_path in self.derivatives_dicts[idx].items():
                derivative_img = tio.ScalarImage(derivative_path)
                derivative_img = self.preprocessing(derivative_img)

                # Apply the same transforms (with the same parameters) to the derivative
                derivative_data = derivative_img.data
                derivative_data = derivative_data.unsqueeze(0)  # Add a batch dimension
                self.transforms.set_randomness(state=None)  # Reset the randomness state of the transforms
                transformed_derivative_data = self.transforms(derivative_data, inputs=transformed_img_data.unsqueeze(1))
                transformed_derivative_data = transformed_derivative_data.squeeze(0).squeeze(0)  # Remove the batch and the additional dimension
                data_dict[derivative_key] = transformed_derivative_data

        return data_dict
