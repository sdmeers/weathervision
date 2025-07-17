# WeatherVision: Multi-class Weather Classification

This repository contains a deep learning project for multi-class weather classification. The model is trained to classify weather conditions from images into categories such as sunshine, cloudy, rain, sunrise, and foggy.

## Table of Contents

- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Dependencies](#dependencies)

## Features

-   **Data Preparation:** Scripts to download and prepare the dataset for training.
-   **Model Training:** Train a deep learning model for weather classification.
-   **Evaluation:** Evaluate the trained model's performance and generate visualizations like confusion matrices and ROC curves.
-   **Prediction:** Use the trained model to predict the weather category of new images.

## Dataset

The project uses the "Multi-class Weather Dataset for Image Classification" from Kaggle. The `download_data.py` script can be used to download the dataset using the Kaggle API.

The dataset should be placed in the `dataset` directory.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd weathervision
    ```

2.  **Install dependencies:**
    Make sure you have Python 3 installed. You can install the required packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The project is divided into several scripts that should be run in the following order:

1.  **Download the data:**
    This script downloads the weather dataset from Kaggle.
    ```bash
    python download_data.py
    ```

2.  **Prepare the data:**
    This script processes the raw data and organizes it into training and validation sets.
    ```bash
    python prepare_data.py
    ```

3.  **Train the model:**
    This script trains the classification model on the prepared data.
    ```bash
    python train.py
    ```

4.  **Evaluate the model:**
    This script evaluates the model's performance on the validation set and generates `confusion_matrix.png` and `roc_curves.png`.
    ```bash
    python evaluate.py
    ```

5.  **Make predictions:**
    Use this script to make predictions on new images.
    ```bash
    python predict.py --image <path/to/image>
    ```

## Results

The evaluation script (`evaluate.py`) saves the following files:

-   `confusion_matrix.png`: A confusion matrix visualizing the performance of the classification model.
-   `roc_curves.png`: ROC curves for each class, showing the model's diagnostic ability.

## Dependencies

The project relies on the following Python libraries:

-   `torch`
-   `torchvision`
-   `scikit-learn`
-   `kaggle`
-   `kagglehub`
-   `pandas`
-   `matplotlib`
-   `numpy`
-   `tqdm`

You can install all dependencies by running `pip install -r requirements.txt`.
