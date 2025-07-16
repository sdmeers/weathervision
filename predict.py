import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import argparse

# --- Configuration ---
MODEL_PATH = './model.pth'
# The class names must be in the same order as the training data folders
CLASS_NAMES = ['Cloudy', 'Rain', 'Shine', 'Sunrise']
NUM_CLASSES = len(CLASS_NAMES)

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_image(image_path):
    """Loads and preprocesses an image for model prediction."""
    # Define the transformations
    # IMPORTANT: We resize to 224x224 directly, which is what the model was trained on.
    # This handles the high-resolution input issue.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    
    # Add a batch dimension (models expect a batch of images)
    # The result is a tensor of shape [1, 3, 224, 224]
    return image_tensor.unsqueeze(0)

def predict(image_path):
    """Loads the model and makes a prediction on a single image."""
    # --- 1. Load Model ---
    # Initialize the model with the same architecture
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    
    # Load the saved weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval() # Set to evaluation mode

    # --- 2. Preprocess Image ---
    image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)

    # --- 3. Make Prediction ---
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class_index = torch.max(probabilities, 1)

    # --- 4. Get Result ---
    predicted_class_name = CLASS_NAMES[predicted_class_index.item()]
    confidence_score = confidence.item()
    
    return predicted_class_name, confidence_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict the weather in an image.')
    parser.add_argument('image_path', type=str, help='Path to the image file.')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found at {args.image_path}")
    else:
        predicted_class, confidence = predict(args.image_path)
        print(f"The image is predicted to be: {predicted_class} with confidence {confidence:.4f}")
