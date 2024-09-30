import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt
from patchify import patchify, unpatchify
import argparse
from function import *  # Assuming this module is available and defines create_3D_noisy_and_clean_data
import random
from tqdm import tqdm

def main(save_path, num_samples, noise_factor, res, fov, seq, tr_range, te_range, image_size):
    # Load data
    T1_3D = np.load(r'essentials/T1_3D_gradientdistortion.npy')
    T2_3D = np.load(r'essentials/T2_3D_gradientdistortion.npy')
    M0_3D = np.load(r'essentials/M0_3D_gradientdistortion.npy')
    B1_3D = np.load(r'essentials/B1_3D_gradientdistortion.npy')
    flipangle_3D = np.load(r'essentials/flipAngleMaprescale_3D_gradientdistortion.npy')
    t2_star_3D = np.load(r'essentials/t2_star_tensor_3D_gradientdistortion.npy')
    ADC_3D = np.load(r'essentials/ADC_3D_gradientdistortion.npy')

    # Field of view and resolution settings
    fov1, fov2, fov3 = fov
    res1, res2, res3 = res
    Resolution = [res1, res2, res3]
    Data_mat = [int(fov1/res1), int(fov2/res2), int(fov3/res3)]
    Bandwidth = 40000

    # Parameters
    TI = 0
    TI2 = 0
    alpha = 30

    # Sequence generation (random TR and TE pairs)
    random.seed(42)
    max_TR, max_TE = tr_range[1], te_range[1]
    pairs = set()
    
    while len(pairs) < num_samples:
        TR = random.randint(te_range[1] + 1, max_TR)
        TE = random.randint(0, TR - 1)
        pairs.add((TR, TE))

    pairs = list(pairs)

    # Create save directories
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Created directory: {save_path}")
    
    # Loop through TR-TE pairs and create data
    for TR, TE in tqdm(pairs[:num_samples], total=num_samples):
        mask, image = create_3D_noisy_and_clean_data([fov1, fov2, fov3], Resolution, Bandwidth, seq, TR, TE, TI, TI2, alpha, noise_factor, T1_3D, T2_3D, M0_3D, B1_3D, flipangle_3D, t2_star_3D, ADC_3D)

        # Determine if patchifying is necessary or saving the entire image
        if image.shape[0] == image_size and image.shape[1] == image_size and image.shape[2] == image_size:
            # Image size matches, save the entire image directly
            input_img = np.expand_dims(image, axis=0)
            input_mask = np.expand_dims(mask, axis=0)
        else:
            # Use patchifying for smaller sizes (e.g., 64x64x64)
            img_patches = patchify(image, (64, 64, 64), step=32)
            mask_patches = patchify(mask, (64, 64, 64), step=32)

            input_img = np.reshape(img_patches, (-1, img_patches.shape[3], img_patches.shape[4], img_patches.shape[5]))
            input_mask = np.reshape(mask_patches, (-1, mask_patches.shape[3], mask_patches.shape[4], mask_patches.shape[5]))

        image_path = os.path.join(save_path, "images", f"{seq}/sub-{TR}-{TE}")
        mask_path = os.path.join(save_path, "masks", f"{seq}/sub-{TR}-{TE}")

        for path in [image_path, mask_path]:
            if not os.path.exists(path):
                os.makedirs(path)

        # Save the output in the given format and size
        for i in range(input_img.shape[0]):
            nii_img = nib.Nifti1Image(input_img[i, :, :, :], np.eye(4))
            nii_mask = nib.Nifti1Image(input_mask[i, :, :, :], np.eye(4))
            filename = f"sub-{TR}-{TE}_{i}_{noise_factor}.nii.gz"

            nib.save(nii_img, os.path.join(image_path, filename))
            nib.save(nii_mask, os.path.join(mask_path, filename))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate noisy and clean 3D medical images.")
    
    # Add arguments for configurable paths and parameters
    parser.add_argument('--save_path', type=str, default='data/', help='Path to save the images')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of TR-TE pairs to generate')
    parser.add_argument('--noise_factor', type=float, default=1, help='Noise factor to apply to the data')
    parser.add_argument('--res', nargs=3, type=float, default=[1.95, 2.34, 2.14], help='Resolution in mm (x, y, z)')
    parser.add_argument('--fov', nargs=3, type=float, default=[250, 300, 275], help='Field of view in mm (x, y, z)')
    parser.add_argument('--seq', type=str, default='SE', help='Sequence type (e.g., SE, GE, IN, etc.)')
    parser.add_argument('--tr_range', nargs=2, type=int, default=[100, 800], help='Range of TR values (min, max)')
    parser.add_argument('--te_range', nargs=2, type=int, default=[10, 50], help='Range of TE values (min, max)')
    parser.add_argument('--image_size', type=int, default=64, help='Image size for saving (64 for patch, other for full size)')

    args = parser.parse_args()
    
    # Pass the arguments to the main function
    main(args.save_path, args.num_samples, args.noise_factor, args.res, args.fov, args.seq, args.tr_range, args.te_range, args.image_size)
