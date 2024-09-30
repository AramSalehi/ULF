import os
import numpy as np
import nibabel as nib
import argparse
import matplotlib.pyplot as plt

# Function to load the image from .npy, .nii, or .nii.gz
def load_image(file_path):
    ext = os.path.splitext(file_path)[1]
    if ext == '.npy':
        return np.load(file_path)
    elif ext in ['.nii', '.gz']:
        return nib.load(file_path).get_fdata()
    else:
        raise ValueError("Unsupported file format: {}".format(ext))

# Function to visualize a specific slice from the input and result images
def visualize_slice(input_path, result_path, slice_index):
    # Load the input image and result
    input_img = load_image(input_path)
    result_img = load_image(result_path)

    # Ensure the slice index is within bounds
    if slice_index < 0 or slice_index >= input_img.shape[2]:
        raise ValueError(f"Slice index {slice_index} is out of bounds for image depth {input_img.shape[2]}")

    # Plot the input and result side by side for the specified slice
    plt.figure(figsize=(12, 6))
    
    # Input image slice
    plt.subplot(1, 2, 1)
    plt.title(f"Input Image (Slice {slice_index})")
    plt.imshow(input_img[:, :, slice_index], cmap="gray")
    plt.axis('off')
    
    # Result image slice
    plt.subplot(1, 2, 2)
    plt.title(f"Result Image (Slice {slice_index})")
    plt.imshow(result_img[:, :, slice_index], cmap="gray")
    plt.axis('off')
    
    # Show the plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Visualize input and result images for a specific slice.")
    parser.add_argument('--input', type=str, required=True, help="Path to the input image (can be .npy, .nii, .nii.gz)")
    parser.add_argument('--result', type=str, required=True, help="Path to the result image (can be .npy, .nii, .nii.gz)")
    parser.add_argument('--slice_index', type=int, required=True, help="Slice index to visualize")
    
    args = parser.parse_args()

    # Visualize the specified slice
    visualize_slice(args.input, args.result, args.slice_index)
