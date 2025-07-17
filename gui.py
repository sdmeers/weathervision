
import tkinter as tk
from tkinter import filedialog, ttk, font
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import models, transforms
import requests
import base64
import json
import io

# --- Configuration ---
RESNET_MODEL_PATH = '/home/sdmeers/Code/weathervision/model.pth'
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL_NAME = "gemma3:4b-it-qat"

# Class names must be consistent with the trained models
RESNET_CLASSES = ['Cloudy', 'Rain', 'Shine', 'Sunrise']
OLLAMA_CLASSES = ["Cloudy", "Rain", "Shine"]

# --- ResNet-18 Model Loading and Prediction ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_resnet_model():
    """Loads the pre-trained ResNet-18 model."""
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(RESNET_CLASSES))
    try:
        model.load_state_dict(torch.load(RESNET_MODEL_PATH, map_location=device))
        model = model.to(device)
        model.eval()
        return model
    except FileNotFoundError:
        return None

resnet_model = load_resnet_model()

def predict_resnet(image_path):
    """Predicts using the ResNet-18 model."""
    if not resnet_model:
        return "ResNet-18 model not found.", 0.0

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = resnet_model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    
    return RESNET_CLASSES[predicted_idx.item()], confidence.item()

# --- Gemma 3 Model Prediction ---
def predict_gemma(image_path):
    """Predicts using the Gemma 3 model via Ollama."""
    prompt = f'You are a weather classification expert. Analyze the image and classify it into one of the following categories: {json.dumps(OLLAMA_CLASSES)}. Respond with a single word.'
    
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        data = {
            "model": OLLAMA_MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "images": [encoded_string]
        }

        response = requests.post(OLLAMA_API_URL, json=data, timeout=60)
        response.raise_for_status()
        
        result = response.json()['response'].strip()
        # Simple confidence simulation - not true probability
        return result, 0.99 # Placeholder confidence

    except requests.RequestException as e:
        return f"Ollama API Error: {e}", 0.0

# --- GUI Application ---
class WeatherVisionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("WeatherVision Classifier")
        self.root.geometry("600x600")
        self.root.configure(bg='white') # Set background to white

        self.image_path = None

        # --- Styling ---
        style = ttk.Style()
        style.theme_use('clam') # Use 'clam' theme for a modern look

        # Configure fonts
        self.custom_font_large = font.Font(family="Helvetica Neue", size=14, weight="bold")
        self.custom_font_medium = font.Font(family="Helvetica Neue", size=12)

        # Configure button style
        style.configure('TButton',
                        font=self.custom_font_medium,
                        background='#2196F3', # Blue
                        foreground='white',
                        padding=10,
                        relief='flat')
        style.map('TButton',
                  background=[('active', '#1976D2')],
                  foreground=[('disabled', 'white')])

        # Configure Combobox style
        style.configure('TCombobox',
                        font=self.custom_font_medium,
                        fieldbackground='white',
                        background='white',
                        foreground='black')
        style.map('TCombobox',
                  fieldbackground=[('readonly', 'white')],
                  selectbackground=[('readonly', 'white')],
                  selectforeground=[('readonly', 'black')])

        # Configure Label style
        style.configure('TLabel',
                        background='white',
                        foreground='black',
                        font=self.custom_font_medium)

        # --- Widgets ---
        self.load_button = ttk.Button(root, text="Load Image", command=self.load_image)
        self.load_button.pack(pady=15)

        self.image_label = tk.Label(root, bg='white') # Ensure image label background is white
        self.image_label.pack(pady=15)

        self.model_var = tk.StringVar(value="Gemma 3")
        self.model_menu = ttk.Combobox(root, textvariable=self.model_var, values=["ResNet-18", "Gemma 3"], state="readonly")
        self.model_menu.pack(pady=15)

        self.classify_button = ttk.Button(root, text="Classify Image", command=self.classify_image, state=tk.DISABLED)
        self.classify_button.pack(pady=15)

        self.result_label = tk.Label(root, text="Result will be shown here", font=self.custom_font_large, bg='white')
        self.result_label.pack(pady=25)

    def load_image(self):
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            img = Image.open(self.image_path)
            img.thumbnail((400, 400)) # Resize for display
            photo = ImageTk.PhotoImage(img)

            self.image_label.config(image=photo)
            self.image_label.image = photo
            self.classify_button.config(state=tk.NORMAL)
            self.result_label.config(text="Image loaded. Ready to classify.")

    def classify_image(self):
        if not self.image_path:
            return

        model_choice = self.model_var.get()
        self.result_label.config(text=f"Classifying with {model_choice}...")
        self.root.update_idletasks()

        if model_choice == "ResNet-18":
            prediction, confidence = predict_resnet(self.image_path)
        elif model_choice == "Gemma 3":
            prediction, confidence = predict_gemma(self.image_path)
        else:
            prediction, confidence = "Invalid model", 0.0

        self.result_label.config(text=f"Prediction: {prediction} ({confidence:.2%})")

if __name__ == "__main__":
    root = tk.Tk()
    app = WeatherVisionApp(root)
    root.mainloop()
