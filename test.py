from model import LungCancerModel
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Define the device for GPU or CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = LungCancerModel(num_classes=3).to(DEVICE)
model.load_state_dict(torch.load("lung_model_pt_final.pth"))

# Define the transforms (same as during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Check if the LIDC_Y_Net folder exists
test_folder = "data"

if not os.path.exists(test_folder):
    print(f"Dataset folder not found at {test_folder}")
else:
    test_dataset = datasets.ImageFolder(os.path.join(test_folder, "test"), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Confusion Matrix and Classification Report
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
    plt.title("Confusion Matrix")
    plt.show()

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))
