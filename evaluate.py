import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from itertools import cycle

# --- Configuration ---
PROCESSED_DATA_PATH = './processed_data'
MODEL_PATH = './model.pth'
BATCH_SIZE = 32

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Data Loading ---
# Use the same transformations as the validation set during training
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the validation dataset
validation_dataset = datasets.ImageFolder(os.path.join(PROCESSED_DATA_PATH, 'validation'), data_transform)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

class_names = validation_dataset.classes
num_classes = len(class_names)
print(f"Evaluating on {len(validation_dataset)} images, with {num_classes} classes: {class_names}")

# --- Model Loading ---
print("\nLoading the trained model...")

# Initialize the model with the same architecture as during training
model = models.resnet18(weights=None) # We don't need pre-trained weights, we have our own
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# Load the saved model weights
model.load_state_dict(torch.load(MODEL_PATH))

# Move the model to the correct device and set it to evaluation mode
model = model.to(device)
model.eval()

print("Model loaded successfully.")

# --- Evaluation Loop ---
print("\nRunning evaluation...")

all_preds = []
all_labels = []
all_scores = []

with torch.no_grad(): # Deactivates autograd engine, reduces memory usage and speeds up computations
    for inputs, labels in tqdm(validation_loader, desc="Evaluating"):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        
        # Get probability scores for ROC curve
        scores = torch.nn.functional.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_scores.extend(scores.cpu().numpy())

print("Evaluation complete.")

# Convert lists to numpy arrays
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_scores = np.array(all_scores)

# --- Results ---
print("\n--- Classification Report ---")
print(classification_report(all_labels, all_preds, target_names=class_names))

print("\n--- Confusion Matrix ---")
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved to confusion_matrix.png")

# --- ROC and AUC ---
print("\n--- Generating ROC and AUC... ---")

# Binarize the labels for One-vs-Rest (OvR) ROC curve calculation
y_test_binarized = label_binarize(all_labels, classes=range(num_classes))

# Dictionaries to hold ROC curve data
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], all_scores[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure(figsize=(12, 10))

colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'deeppink', 'navy', 'green'])
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(class_names[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2) # Dashed diagonal
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc="lower right")
plt.savefig('roc_curves.png')
print("ROC curves saved to roc_curves.png")
