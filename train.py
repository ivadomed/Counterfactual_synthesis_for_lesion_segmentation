from Data_manager import get_paths, get_dataloader, Dataset_2D
from Simple_2D_Diffusion_model import *
from monai.data import Dataset, DataLoader, CacheDataset
import torch

# define model and diffusion
timesteps = 500
device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNetModel(
    in_channels=1,
    model_channels=96,
    out_channels=1,
    channel_mult=(1, 2, 2),
    attention_resolutions=[]
)
model.to(device)

gaussian_diffusion = GaussianDiffusion(timesteps=timesteps)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

## Load Data
train_paths = get_paths("data//data-multi-subject")
train_loader = get_dataloader(train_paths, batch_size=4)

## Train the model
epochs = 1
# Fill here

for epoch in range(epochs):
    model.train()
    print(f"Epoch {epoch+1}/{epochs}")
    for x in train_loader:
        print(x.shape)
        x = x.to(device)
        t = torch.randint(0, timesteps, (x.shape[0],), device=device)
        optimizer.zero_grad()
        loss = gaussian_diffusion.train_losses(model, x, t)
        loss.backward()
        optimizer.step()
    if (epoch+1) % 100 == 0:
        torch.save(model.state_dict(), f"model_{epoch+1}.pth")
        

#save model

torch.save(model.state_dict(), "model.pth")