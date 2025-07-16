import kagglehub
import os
import shutil

# Define the dataset handle from Kaggle
dataset_handle = 'pratik2901/multiclass-weather-dataset'

# Define the desired local path for the dataset
# We'll place it in a 'dataset' folder in the current project directory
local_dataset_path = './dataset'

print(f"Downloading dataset: {dataset_handle}...")

# kagglehub downloads to a local cache and returns the path
# We don't have direct control over the initial download location
cached_path = kagglehub.dataset_download(dataset_handle)

print(f"Dataset downloaded to cache at: {cached_path}")

# Now, we'll copy the data from the cache to our desired local directory
# to have it inside our project folder.

# If the target directory already exists, remove it for a clean copy
if os.path.exists(local_dataset_path):
    print(f"Removing existing directory at: {local_dataset_path}")
    shutil.rmtree(local_dataset_path)

# Copy the entire directory from the cache to our local path
print(f"Copying dataset from {cached_path} to {local_dataset_path}...")
shutil.copytree(cached_path, local_dataset_path)

print("Dataset successfully copied to the project's ./dataset folder!")
