# train.py
import os
import torch
import argparse
from tqdm import tqdm
import numpy as np
from monai.data import DataLoader, CacheDataset
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Orientationd, Compose,
    Rand3DElasticd, RandFlipd, RandRotate90d
)
from models.unet_dcr_3d import create_model as dcr
from torch.optim import Adam
import torch.nn as nn

# Set up argument parsing for epochs
parser = argparse.ArgumentParser(description="Train 3D Model")
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
args = parser.parse_args()

# Set device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Print whether CUDA is being used
if torch.cuda.is_available():
    print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Training on CPU")

# Load images and masks for training
def generate_list_images(directory_path):
    file_path = []
    for root, directories, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.nii.gz'):
                file_path.append(os.path.join(root, file))
    return sorted(file_path)

input_dir = 'data/images'
output_dir = 'data/masks'
list_inputs = generate_list_images(input_dir)
list_outputs = generate_list_images(output_dir)

train_files = [{"image": img, "mask": mask} for img, mask in zip(list_inputs, list_outputs)]

# Define data transformations
keys = ["image", "mask"]
train_transforms = Compose([
    LoadImaged(keys=keys),
    EnsureChannelFirstd(keys=keys),
    Orientationd(keys=keys, axcodes="RAS"),
    Rand3DElasticd(keys=keys, mode=['bilinear', 'bilinear'], sigma_range=(1, 11), magnitude_range=(50, 150), prob=1),
    RandFlipd(keys=keys, spatial_axis=[0], prob=0.10),
    RandFlipd(keys=keys, spatial_axis=[1], prob=0.10),
    RandFlipd(keys=keys, spatial_axis=[2], prob=0.10),
    RandRotate90d(keys=keys, prob=0.10, max_k=3),
])

# DataLoader for training
batch_size = 1
train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=0)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)

# Model setup
model = dcr(in_channels=1, base_channels=32).to(device)
optimizer = Adam(model.parameters(), lr=1e-5)
loss_fn = nn.L1Loss()

# Training loop
def train_model(epochs):
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}")
        
        for step, batch in enumerate(progress_bar):
            x, y = batch["image"].to(device), batch["mask"].to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Update the progress bar description
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Print the average loss for the epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Average Loss for Epoch {epoch+1}: {avg_epoch_loss:.4f}")

    # Save final model
    torch.save(model.state_dict(), r"essentials\model_weights\best_metric_model.pth")

if __name__ == "__main__":
    train_model(args.epochs)
