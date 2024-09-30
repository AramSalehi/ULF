#%%
import os
import torch
import numpy as np
import nibabel as nib
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Orientationd, Compose
)
import argparse
from models.unet_dcr_3d import create_model as dcr
from monai.inferers import sliding_window_inference
import matplotlib.pyplot as plt
# Set device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load images from .npy, .nii, .nii.gz
def load_image(file_path):
    ext = os.path.splitext(file_path)[1]
    if ext == '.npy':
        return np.load(file_path)
    elif ext in ['.nii', '.gz']:
        return nib.load(file_path).get_fdata()
    else:
        raise ValueError("Unsupported file format: {}".format(ext))

# Function to save images in the same format
def save_image(data, file_path, original_file_path):
    ext = os.path.splitext(original_file_path)[1]
    
    # Cast data to an appropriate type
    if ext in ['.nii', '.gz']:
        # Convert data to int16 or float32 for compatibility
        data = data.astype(np.float32)  # Or np.float32, depending on your needs
        nib.save(nib.Nifti1Image(data, affine=np.eye(4)), file_path)
    elif ext == '.npy':
        np.save(file_path, data)
    else:
        raise ValueError("Unsupported file format: {}".format(ext))

# Function to handle prediction
def predict_image(image_path, result_dir, model):
    print(f"Processing: {image_path}")
    
    # Load image
    image_data = load_image(image_path)
    image_data = torch.tensor(image_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # Shape: [1, 1, H, W, D]
   
    
    # Perform inference
    with torch.no_grad():
        output = sliding_window_inference(image_data, (64, 64, 64), 4, model, overlap=0.8,mode="gaussian")

    # Convert output to numpy
    output_np = torch.squeeze(output).cpu().numpy()
    plt.imshow(np.rot90(output_np[:,64,:],1),cmap="gray")
    plt.title("Output")
    plt.show()
    plt.imshow(np.rot90(torch.squeeze(image_data).cpu().numpy()[:,64,:],1),cmap="gray")
    plt.title("Input")
    plt.show()
    # Create results directory if it doesn't exist
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Save output in the same format
    output_name = os.path.basename(image_path)  # Get the file name
    output_path = os.path.join(result_dir, output_name)
    save_image(output_np, output_path, image_path)

    print(f"Saved result: {output_path}")

# Main function to iterate over all files in a directory
def predict_on_directory(input_dir, result_dir, model):
    valid_extensions = ['.npy', '.nii', '.gz']
    for root, _, files in os.walk(input_dir):
        for file in files:
            if any(file.endswith(ext) for ext in valid_extensions):
                image_path = os.path.join(root, file)
                predict_image(image_path, result_dir, model)

# Argument parser for specifying the input path, result folder, and model weights
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Model Inference")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing input images")
    parser.add_argument('--result_dir', type=str, default='results', help="Directory to save predicted results")
    parser.add_argument('--model_weights', type=str, required=True, help="Path to the model weights file (.pth)")

    args = parser.parse_args()

    # Load the model and weights
    model = dcr(in_channels=1, base_channels=32).to(device)
    model.load_state_dict(torch.load(args.model_weights, map_location=device))
    model.eval()

    # Run prediction on the directory
    predict_on_directory(args.input_dir, args.result_dir, model)

# %%
