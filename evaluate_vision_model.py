import requests
import base64
import json
import os
import glob
import random
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import label_binarize

# --- Configuration ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:4b-it-qat"
DATA_DIR = "/home/sdmeers/Code/weathervision/processed_data_vision_model"
CLASSES = ["Cloudy", "Rain", "Shine"]
# Set SAMPLE_FRACTION to 1.0 to run on all images
SAMPLE_FRACTION = 0.2
PROMPT = f""" You are a weather classification expert. Your task is to analyze the image and classify the weather conditions. Respond with ONLY a valid JSON object containing the probability for each of the following categories: {json.dumps(CLASSES)}. The probabilities must sum to 1.0. Example response: {{\"Cloudy\": 0.15, \"Rain\": 0.8, \"Shine\": 0.05}}"""

def get_all_image_paths(directory):
    """Gets all image paths from the specified directory."""
    patterns = ["**/*.jpg", "**/*.jpeg", "**/*.png"]
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(os.path.join(directory, pattern), recursive=True))
    return paths

def get_prediction(image_path):
    """Sends an image to the Ollama API and returns the predicted probabilities."""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        data = {
            "model": MODEL_NAME,
            "prompt": PROMPT,
            "stream": False,
            "images": [encoded_string]
        }

        response = requests.post(OLLAMA_API_URL, json=data, timeout=60)
        response.raise_for_status()

        response_text = response.json()['response'].strip()
        # Clean the response to extract only the JSON part
        json_part = response_text[response_text.find('{'):response_text.rfind('}')+1]
        probabilities = json.loads(json_part)
        
        # Ensure all classes are present and in the correct order
        ordered_probs = [probabilities.get(c, 0.0) for c in CLASSES]
        return ordered_probs

    except (requests.RequestException, json.JSONDecodeError, KeyError, FileNotFoundError) as e:
        print(f"\nError processing {os.path.basename(image_path)}: {e}")
        return [0.0] * len(CLASSES) # Return a neutral prediction on error

def main():
    """Main function to run the evaluation."""
    print("Starting evaluation...")
    all_image_paths = get_all_image_paths(DATA_DIR)
    
    if not all_image_paths:
        print(f"Error: No images found in {DATA_DIR}. Please check the path and subdirectories.")
        return

    # --- Sampling Logic ---
    if SAMPLE_FRACTION < 1.0:
        num_samples = int(len(all_image_paths) * SAMPLE_FRACTION)
        image_paths = random.sample(all_image_paths, num_samples)
        print(f"Running evaluation on a random sample of {len(image_paths)} images ({SAMPLE_FRACTION:.0%}).")
    else:
        image_paths = all_image_paths
        print(f"Running evaluation on all {len(image_paths)} images.")


    y_true = []
    y_pred_probs = []

    for image_path in tqdm(image_paths, desc="Evaluating model"):
        true_label = os.path.basename(os.path.dirname(image_path))
        if true_label in CLASSES:
            y_true.append(true_label)
            probs = get_prediction(image_path)
            y_pred_probs.append(probs)

    if not y_true:
        print("Could not find any valid images to process. Make sure your image folders are named correctly.")
        return

    y_true_b = label_binarize(y_true, classes=CLASSES)
    y_pred_probs = np.array(y_pred_probs)
    y_pred_labels = [CLASSES[i] for i in np.argmax(y_pred_probs, axis=1)]

    # --- Classification Report ---
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred_labels, labels=CLASSES))

    # --- Confusion Matrix ---
    print("Generating confusion matrix...")
    cm = confusion_matrix(y_true, y_pred_labels, labels=CLASSES)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('ollama_confusion_matrix.png')
    plt.close()
    print("Saved confusion matrix to ollama_confusion_matrix.png")

    # --- ROC Curve ---
    print("Generating ROC curves...")
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(CLASSES)):
        fpr[i], tpr[i], _ = roc_curve(y_true_b[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(12, 10))
    for i in range(len(CLASSES)):
        plt.plot(fpr[i], tpr[i], label=f'ROC curve for {CLASSES[i]} (area = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig('ollama_roc_curves.png')
    plt.close()
    print("Saved ROC curves to ollama_roc_curves.png")
    print("\nEvaluation complete.")

if __name__ == "__main__":
    main()
