import nibabel as nib
import monai
from monai.data import Dataset, DataLoader, CacheDataset
from monai.transforms import (
    Compose,
    RandFlip,
    RandRotate90,
    RandRotate,
    RandShiftIntensity,
    ToTensor,
    RandSpatialCrop,
    LoadImage,
    SqueezeDim,
    RandRotate,
    RandSimulateLowResolution,
    CenterSpatialCrop,
)
import os



# Define a custom dataset class
class Dataset_2D(Dataset):
    def __init__(self, paths, transform=None):
        self.data = {"paths" : paths}
        self.transform = transform
        self.length = len(self.data["paths"])

    def __len__(self):
        return len(self.data["paths"])

    def __getitem__(self, index):
        path = self.data["paths"][index]

        if self.transform:
            image = self.transform(path)

        return image
    
# use monai to define the transforms for data augmentation
# perform the following transformations : rotation (random between +3° and -3°), flipping (random between 0°,  90 °, 180° and 270°), cropping (Random size, random place) and shifting (random shift)

train_transforms = Compose(
    [
        LoadImage(image_only = True, ensure_channel_first = True),
        RandShiftIntensity(offsets = 0.1, prob = 0.5),
        RandRotate(range_x = 3, range_y = 3, range_z = 3, prob = 0.2),
        CenterSpatialCrop(roi_size = (1, -1, -1)),
        RandSpatialCrop([1, 256, 256],  max_roi_size = [1, 256, 256], random_center = False),
        SqueezeDim(dim = 1),
        ToTensor(),
        
    ]
)

val_transforms = Compose(
    [
        LoadImage(image_only=True, ensure_channel_first=True),
        ToTensor(),
    ]
)

def get_paths(base_dir):

    desired_extension = ".json"

    # Initialize lists to store the relative paths for T2w files
    t2w_file_paths = []

    print("Searching for T1w, T2w, and DWI files in", base_dir, "...")

    # Traverse the directory structure
    for root, dirs, files in os.walk(base_dir):
        # Exclude the "derivatives" subfolder
        if "derivatives" in dirs:
            dirs.remove("derivatives")
        for file in files:
            # Check if the file name contains the desired names
            if "T2w" in file and file.endswith(desired_extension):
                # Get the relative path of the T1w file
                relative_path = os.path.relpath(os.path.join(root, file), base_dir)
                # Remove the file extension
                relative_path = os.path.splitext(base_dir + relative_path)[0] + ".nii.gz"
                # Append the relative path to the T1w file paths list
                t2w_file_paths.append(relative_path)


    t2w_file_paths = t2w_file_paths[:5]

    print("Found", len(t2w_file_paths), "T1w files")
    return t2w_file_paths

def get_dataloader(paths, batch_size = 4, num_workers = 1):
    transform = Compose(
    [
        LoadImage(image_only = True, ensure_channel_first = True),
        RandShiftIntensity(offsets = 0.1, prob = 0.5),
        RandRotate(range_x = 3, range_y = 3, range_z = 3, prob = 0.2),
        CenterSpatialCrop(roi_size = (1, -1, -1)),
        RandSpatialCrop([1, 256, 256],  max_roi_size = [1, 256, 256], random_center = False),
        SqueezeDim(dim = 1),
        ToTensor(),
        
    ]
    )

    dataset = Dataset_2D(paths, transform)
    dataloader = DataLoader(dataset, batch_size = batch_size, num_workers = num_workers, pin_memory = True)
    return dataloader