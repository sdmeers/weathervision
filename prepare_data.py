import os
import glob
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- Configuration ---
# Path to the raw dataset downloaded from Kaggle
RAW_DATA_PATH = './dataset/Multi-class Weather Dataset'

# Path where we will store the organized, split data
PROCESSED_DATA_PATH = './processed_data'

# Define the ratio for splitting data into training and validation sets
# 0.2 means 20% of the data will be used for validation
VALIDATION_SPLIT = 0.2

# A random seed ensures that our data split is the same every time we run the script.
# This is crucial for reproducibility.
RANDOM_SEED = 42

# --- 1. Find all image paths and extract labels ---
print(f"Scanning for images in: {RAW_DATA_PATH}")

# Use glob to find all files ending in .jpg, .jpeg, or .png, recursively
image_paths = glob.glob(os.path.join(RAW_DATA_PATH, '**', '*.jpg'), recursive=True)
image_paths.extend(glob.glob(os.path.join(RAW_DATA_PATH, '**', '*.jpeg'), recursive=True))
image_paths.extend(glob.glob(os.path.join(RAW_DATA_PATH, '**', '*.png'), recursive=True))

# The label (e.g., 'Cloudy', 'Rain') is the name of the parent directory of the image
labels = [os.path.basename(os.path.dirname(p)) for p in image_paths]

# Create a pandas DataFrame to hold the paths and labels
data_df = pd.DataFrame({'path': image_paths, 'label': labels})

print(f"Found {len(data_df)} images belonging to {len(data_df['label'].unique())} classes.")
print("Class distribution:")
print(data_df['label'].value_counts())

# --- 2. Split the data into training and validation sets ---
print("\nSplitting data into training and validation sets...")

train_df, val_df = train_test_split(
    data_df, 
    test_size=VALIDATION_SPLIT, 
    random_state=RANDOM_SEED, 
    stratify=data_df['label']  # Ensures class distribution is similar in train/val sets
)

print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")

# --- 3. Create the processed data directory structure and copy files ---
print(f"\nCreating processed data directory at: {PROCESSED_DATA_PATH}")

# Remove the directory if it exists to ensure a clean slate
if os.path.exists(PROCESSED_DATA_PATH):
    shutil.rmtree(PROCESSED_DATA_PATH)

def copy_files(df, split_name):
    """Copies files from the source path to the destination directory structure."""
    print(f"Copying {split_name} files...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        dest_dir = os.path.join(PROCESSED_DATA_PATH, split_name, row['label'])
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy(row['path'], dest_dir)

copy_files(train_df, 'train')
copy_files(val_df, 'validation')

print("\nData preparation complete!")
