from torch.utils.data import DataLoader
from data_manager import Auto_encoder_Dataset, dataset_splitter, paths_to_Dataset, fetch_all_T2w_paths
from Auto_encoder_network import AutoEncoder_2D
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score
import argparse


### Retrieve the arguments
# Create the parser
parser = argparse.ArgumentParser(description='Train and evaluate the model.')

# Add the arguments
parser.add_argument('--evaluate', type=bool, default=True,
                    help='a boolean for the evaluation mode, True by Default')
parser.add_argument('--model_path', type=str, default='none',
                    help='the path to the model, Random weights by Default')
parser.add_argument('--model_output', type=str, default='checkpoints//model.pth',
                    help='the path to the output model, checkpoints//model.pth by Default')
parser.add_argument('--data_path', type=str, default='data//data-multi-subject//',
                    help='the path to the data, data//data-multi-subject// by Default')
parser.add_argument('--num_epochs', type=int, default=200,
                    help='the number of epochs, 200 by Default')
# Parse the arguments
args = parser.parse_args()

# Access the arguments
evaluate = args.evaluate
model_path = args.model_path
model_output = args.model_output
data_path = args.data_path
num_epochs = args.num_epochs

# Define the training loop
def train_one_epoch(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs = data
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return model, running_loss / len(train_loader)

#### Load the data

# Find the relative paths of the T1w and T2w files in the specified directory
t1w_file_paths, t2w_file_paths = fetch_all_T2w_paths(data_path)

# Split the data into training and validation sets
pd_train_data, pd_val_data = dataset_splitter(t1w_file_paths, t2w_file_paths)

# Create the training and validation datasets
train_dataset = paths_to_Dataset(pd_train_data)
val_dataset = paths_to_Dataset(pd_val_data, val=True)

#### Train the model

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoEncoder_2D()
if model_path != 'none':
    model.load_state_dict(torch.load(model_path), map_location=device)
    print("Model loaded")
# move the model to the device
model.to(device)
# define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# define the binary cross entropy loss function
criterion = nn.BCELoss()
# define the data loader
dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1} / {num_epochs}")
    model, train_loss = train_one_epoch(model, dataloader, criterion, optimizer)
    print(f"Epoch {epoch + 1} training loss: {train_loss}")
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"checkpoints//model_save_epoch_{epoch + 1}.pth")
#save model
torch.save(model.state_dict(), model_output)

#### Evaluate the model

if evaluate:
    #model.eval()
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    val_loss = 0.0
    for i, data in enumerate(val_loader):
        inputs = data
        #to device and to float 64
        inputs = inputs.float().to(device)
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        val_loss += loss.item()
    print(f"Validation loss: {val_loss / len(val_loader)}")
    print("Evaluation done")
