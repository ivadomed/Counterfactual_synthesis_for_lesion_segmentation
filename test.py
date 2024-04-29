from torch.utils.data import DataLoader
from data_manager import Auto_encoder_Dataset, dataset_splitter, paths_to_Dataset, fetch_all_T2w_paths
from Auto_encoder_network import AutoEncoder_2D, AutoEncoder_2D_low, AutoEncoder_2D_MLP
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd



### Retrieve the arguments
# Create the parser
parser = argparse.ArgumentParser(description='Train and evaluate the model.')

# Add the arguments
parser.add_argument('--model_type', type=str, default='high',
                    help='the model structure among high, low and MLP. high by Default')
parser.add_argument('--model_path', type=str, default='none',
                    help='the path to the model, Random weights by Default')
parser.add_argument('--sample_directory_input', type=str, default='data/samples',
                    help='the path to the sample directory, data/samples  by Default')
parser.add_argument('--sample_directory_output', type=str, default='outputs',
                    help='the path to the output model, outputs by Default')


def fetch_all_nii_paths(source_folder):
    img_paths = []
    for p in Path(source_folder).rglob('*.nii.gz'):
        img_paths.append(str(p))
    # convert to data frame
    img_paths = np.array(img_paths)
    img_paths_df = pd.DataFrame(img_paths, columns=['image_path'])
    return img_paths_df

# Parse the arguments
args = parser.parse_args()

# Access the arguments
model_type = args.model_type
model_path = args.model_path
sample_directory_input = args.sample_directory_input
model_output = args.sample_directory_output



#### Load the data

# Find the relative paths of the T1w and T2w files in the specified directory
nii_file_paths = fetch_all_nii_paths(sample_directory_input)


# Create the test datasets
test_dataset = paths_to_Dataset(nii_file_paths, test=True)

# Create the test dataloaders
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

#### Define the model

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if model_type == 'high':
    model = AutoEncoder_2D()
elif model_type == 'low':
    model = AutoEncoder_2D_low()
elif model_type == 'MLP':
    model = AutoEncoder_2D_MLP()
else:
    raise ValueError("model_type should be among high, low and MLP")
model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
model.to(device)

model_name = model_path.split('/')[-1].replace('.pth', '')


for i, data in enumerate(test_loader):
    inputs = data
    img_name = test_dataset.img_paths[i].split('\\')[-1].replace('.nii.gz', '')
    #to device and to float 64
    inputs = inputs.float().to(device)
    fig, ax = plt.subplots(1,3)
    outputs = model(inputs)
    # size
    fig.set_size_inches(30, 10)
    ax[0].imshow(inputs.cpu().detach().numpy().squeeze())
    ax[0].set_title("Input")
    ax[1].imshow(outputs.cpu().detach().numpy().squeeze())
    ax[1].set_title(f"Output_{model_name}")

    # show pixel wise square difference
    diff = (inputs.cpu().detach().numpy().squeeze() - outputs.cpu().detach().numpy().squeeze())**1
    ax[2].imshow(diff, cmap='gray')
    ax[2].set_title("Difference")

    # Save the plot
    plt.savefig(f"outputs//triptique_{img_name}_{model_name}.png")
    plt.close()


