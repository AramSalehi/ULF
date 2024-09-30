
# 3D Medical Image Generation and Model Training

This repository contains scripts for generating noisy and clean 3D medical images, training a 3D UNet-based model, and testing the trained model for image reconstruction tasks.

## Prerequisites

- **Python 3.x**: Ensure Python is installed on your system.
- **Required Libraries**: Install the required Python libraries listed in `requirements.txt` by running the following command:

  ```bash
  pip install -r requirements.txt
  ```

## Setup

### Step 1: Download the Required Files

Before running the scripts, you need to download essential data files from Google Drive. These files are required for generating the 3D images and training the model.

1. Download the files from the following [Google Drive link](https://drive.google.com/file/d/1NKwA4Caf_0d4MNOrhIyks-1C8rlYbZtv/view?usp=sharing).
2. Save the downloaded files into the `essentials/` folder inside the project directory:

   ```
   your-repository/essentials/
   ```

   The following files should be present in the `essentials/` folder:
   - `T1_3D_gradientdistortion.npy`
   - `T2_3D_gradientdistortion.npy`
   - `M0_3D_gradientdistortion.npy`
   - `B1_3D_gradientdistortion.npy`
   - `flipAngleMaprescale_3D_gradientdistortion.npy`
   - `t2_star_tensor_3D_gradientdistortion.npy`
   - `ADC_3D_gradientdistortion.npy`

### Step 2: Generate Data

Once the required files are downloaded, you can run the `main.py` script to generate noisy and clean 3D images.

#### Example Command:

```bash
python main.py --save_path 'output/' --num_samples 5 --noise_factor 1 --res 1.95 2.34 2.14 --fov 250 300 275 --seq 'SE' --tr_range 100 800 --te_range 10 50 --image_size 64
```

This will generate 5 samples with the specified TR-TE pairs and save them in the `output/` directory.

## Training the Model

Once the data is prepared, you can train the 3D model using the `train.py` script.

### Step 1: Prepare Data

Ensure that the generated images and masks are located in the `data/images/` and `data/masks/` directories, respectively.

### Step 2: Run the Training Script

You can run the `train.py` script to train the model on the generated data. The script accepts the number of epochs as an argument.

#### Example Command:

```bash
python train.py --epochs 10
```

This will train the model for 10 epochs and save the best model weights in the `essentials/model_weights/` directory.

## Testing the Model

Once the model has been trained, you can use the `test.py` script to run inference on new data.

### Step 1: Prepare Input Data

Make sure you have input images stored in a directory. The script supports `.npy`, `.nii`, and `.gz` file formats.

### Step 2: Run the Testing Script

To run inference and generate predictions, use the `test.py` script. Specify the directory containing input images, the directory to save the results, and the path to the trained model weights.

#### Example Command:

```bash
python test.py --input_dir 'test_images/' --result_dir 'results/' --model_weights 'essentials/model_weights/best_metric_model.pth'
```

This will run inference on all images in the `test_images/` directory and save the predicted results in the `results/` directory.

### Testing Parameters

- `--input_dir`: Directory containing input images for testing.
- `--result_dir`: Directory to save the predicted results (default: `results/`).
- `--model_weights`: Path to the trained model weights file (e.g., `best_metric_model.pth`).

## Output Files

- The generated images will be saved in the `images/` directory under the sequence type.
- The masks will be saved in the `masks/` directory.
- The trained model weights will be saved in the `essentials/model_weights/` directory.
- The predicted test results will be saved in the `results/` directory.

## Further Steps

Additional instructions and features will be provided as the project evolves.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

