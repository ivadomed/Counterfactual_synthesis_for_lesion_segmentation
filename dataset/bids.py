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
    Transposed,
    Flipd,
)
import os
from typing import Optional
import argparse
from pathlib import Path
import time



class BIDSDataset(Dataset):
    def __init__(self, root_dir: str, is_VQGAN: bool = False, contrasts = [], derivatives: bool = False, mandatory_derivates=[]):
        super().__init__()
        self.root_dir = root_dir

        # If is_VQGAN is set to True, the dataset will use the VQGAN training pipeline
        self.is_VQGAN = is_VQGAN

        # The list of contrasts to filter the files, if empty, it will take all contrasts
        self.contrasts = contrasts

        # If derivatives is set to True, the dataset will look for the derivatives in the derivatives folder
        self.derivatives = derivatives
        

        # Encapssulate a list of dictionnaries with the derivative name as key and the path as value
        self.derivatives_paths = []
        self.file_paths = self.get_data_files()
        
        if self.derivatives:
            #list of derivatives that must be present in the derivatives folder in order to be considered
            self.mandatory_derivates = mandatory_derivates
            self.check_mandatory_derivatives()
        
        print(f'Found {len(self.file_paths)} files in {self.root_dir}')
    
    def check_mandatory_derivatives(self):
        """
        This function remove all the file path which don't have the mandatory derivatives
        """
        idx_to_pop = []
        for idx, derivative_paths in enumerate(self.derivatives_paths):
            for mandatory_derivative in self.mandatory_derivates:
                if mandatory_derivative not in derivative_paths:
                    idx_to_pop.append(idx)
                    break
        # sort the list in reverse order to avoid index errors
        idx_to_pop.sort(reverse=True)
        for idx in idx_to_pop:
            self.file_paths.pop(idx)
            self.derivatives_paths.pop(idx)

    
    def find_derivatives(self):
        """
        This function will look for the derivatives in the derivatives folder.
        It will return a dictionnary with the original file name as key and a list of dictionnaries with the derivative name as key and the path as value
        (finding derivatives first ensure to browse the derivatives folder only once and speeds up the derivative linking)
        """
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
        return derivatives_files_paths
        
    
    def link_derivatives(self, file, derivatives_files_paths):
        """
        This function uses the dictionnary of derivatives done by "find_derivatives"
        in order to add the derivatives information to the self.derivatives_paths list
        which allow using an index instead of a file name to get the derivatives
        """
        self.derivatives_paths.append({})
        if file in derivatives_files_paths:
            for derivative_dict in derivatives_files_paths[file]:
                derivative_key = derivative_dict['derivative_key']
                derivative_path = derivative_dict['derivate_path']
                self.derivatives_paths[-1][derivative_key] = derivative_path
    

    def get_data_files(self):
        """
        Browse the root_dir to find the .nii.gz files in the sub-## folders with the correct contrasts
        link the derivatives if the derivatives argument is set to True
        """
        files_paths = []
        if self.derivatives:
            derivatives_files_paths = self.find_derivatives()

        for root, dirs, files in os.walk(self.root_dir):
            if self.root_dir +'/sub-' in root or self.root_dir +'\\sub-' in root:
                for file in files:
                    if file.endswith('.nii.gz'):
                        if len(self.contrasts) > 0:
                            for contrast in self.contrasts:
                                if contrast in file:
                                    files_paths.append(os.path.join(root, file))
                                    if self.derivatives:
                                        self.link_derivatives(file, derivatives_files_paths)
                        else:
                            files_paths.append(os.path.join(root, file))
                            if self.derivatives:
                                self.link_derivatives(file, derivatives_files_paths)

        
        return files_paths
    
    def get_sample_dict(self, idx: int):
        """
        This function returns a dictionnary with the paths of the main image on the "data" key
        and the derivatives path on the derivatives names keys
        """
        sample_dict = {}
        sample_dict['data'] = self.file_paths[idx]
        if self.derivatives:
            sample_dict.update(self.derivatives_paths[idx])
        return sample_dict
    
    
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        sample_paths = self.get_sample_dict(idx)
        keys = list(sample_paths.keys())

        # Load the image to get the min and max values so it can be and normalized between -1 and 1 then 
        # => (probably not the best way, feel free to suggest a better one)
        img = LoadImage()(sample_paths['data'])
        img_np = img.numpy()
        a_min = img_np.min().astype(float)
        a_max = img_np.max().astype(float)


        TRAIN_VQGAN_TRANSORMS = Compose([
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            Transposed(keys=keys, indices=(0, 3, 1, 2)),
            Flipd(keys=keys, spatial_axis=1),
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
            #Transposed(keys=keys, indices=(0, 3, 1, 2)),
            #Flipd(keys=keys, spatial_axis=1),
            ScaleIntensityRanged(keys=['data'], a_min=a_min, a_max=a_max, b_min=-1, b_max=1),
            # change here to the desired shape (/!\ must be powers of 2, GPU memory consuption is proportional to the size of the image)
            ResizeWithPadOrCropd(keys=keys, spatial_size=[32, 256, 256], mode="replicate"),
            RandShiftIntensityd(keys=keys, offsets=0.1, prob=0.5),
            RandRotated(keys=keys, range_x=0.3, range_y=0.0, range_z=0.0, prob=0.5),
            ToTensord(keys=keys),
        ])


        if self.is_VQGAN:
            sample_tensors = TRAIN_VQGAN_TRANSORMS(sample_paths)
        else:
            sample_tensors = TRAIN_DDPM_TRANSFORMS(sample_paths)
        
        return sample_tensors
