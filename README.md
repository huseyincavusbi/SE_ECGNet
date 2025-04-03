# SE-ECGNet: ECG Classification with Squeeze-and-Excitation Networks

This project demonstrates the effectiveness of Squeeze-and-Excitation (SE) blocks for distinguishing between normal and abnormal ECG signals using the PTB-XL dataset.

## Key Features

*   Utilizes the large, publicly available **PTB-XL ECG dataset**.
*   Employs a **SEResNeXt50** model (from `timm`) highlighting the capability of SE blocks.
*   Addresses class imbalance using **WeightedRandomSampler** and **Focal Loss**.
*   Includes data preprocessing, augmentation, and visualization steps.
*   Uses modern training techniques like **AdamW**, **ReduceLROnPlateau**, **Mixed Precision Training**, and **Early Stopping**.

## Dataset

*   **PTB-XL:** A large publicly available electrocardiography dataset containing ~22,000 recordings from ~19,000 patients.
*   Downloaded via KaggleHub (`khyeh0719/ptb-xl-dataset`).

## Methodology (Step-by-Step)

1.  **Dataset Acquisition:** Download the PTB-XL dataset using KaggleHub.
2.  **Data Preprocessing:**
    *   Load metadata (`ptbxl_database.csv`).
    *   Handle missing values 
    *   Categorize ECGs into 'Normal' and 'Abnormal' based on text patterns in the 'report' column.
    *   Save categorized metadata subsets.
3.  **Signal Loading & Preparation:**
    *   Use `wfdb` library to load raw ECG signal data (`.dat` files).
    *   Pad or truncate signals to a fixed length (4096 samples).
4.  **Data Augmentation (Training Set):**
    *   Apply `RandomTimeWarp`, `RandomShift`, `RandomScale`, `AddGaussianNoise`, `RandomAmplitudeScale`.
5.  **Dataset Splitting & Sampling:**
    *   Stratified train/validation/test split (80/20).
    *   Use `WeightedRandomSampler` during training to address class imbalance between Normal/Abnormal ECGs.
6.  **Model Training:**
    *   Train the SEResNeXt50 model for binary classification (Normal vs. Abnormal).
    *   Utilize Mixed Precision Training for efficiency.
    *   Optimize using AdamW and Focal Loss.
    *   Adjust learning rate with ReduceLROnPlateau scheduler.
    *   Employ Early Stopping based on validation loss.
7.  **Evaluation:**
    *   Monitor training and validation loss/accuracy per epoch.
    *   Generate classification reports on the validation set.

## Technical Details

*   **Model Architecture:** SEResNeXt50 (`timm` library) with SE blocks.
*   **Loss Function:** Focal Loss (alpha=0.25, gamma=2).
*   **Optimizer:** AdamW (lr=0.0001, weight_decay=1e-4).
*   **Scheduler:** ReduceLROnPlateau (monitoring `val_loss`).
*   **Training Techniques:** Mixed Precision (AMP), Early Stopping (patience=7).
*   **Preprocessing:** Missing value imputation (median/mode), column dropping, signal padding/truncation (length 4096).
*   **Augmentation:** Time Warp, Shift, Scale, Gaussian Noise, Amplitude Scale.
*   **Sampling:** WeightedRandomSampler.

*   ## How to Run

1.  **Recommended:** Use **Google Colab** for easy setup and GPU acceleration.
    *   Upload the `SE-ECGNet.ipynb` notebook to your Google Drive and open it with Colab.
    *   Ensure a GPU runtime is selected (`Runtime` -> `Change runtime type` -> `Hardware accelerator` -> `GPU`).
    *   Colab often has many dependencies pre-installed. The notebook includes `!pip install` commands for missing ones (`wfdb`, `timm`).
2.  Run the notebook cells sequentially.
    *   The initial cells handle dataset download via KaggleHub (you might need to authenticate Kaggle).
