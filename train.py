import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets
from sklearn.utils.class_weight import compute_class_weight
from model import LungCancerModel  # Importing the model
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Settings ---
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 3
DATA_DIR = 'data/'  # 'data/train' and 'data/val' expected
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data Loading ---
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=val_transform)

# Compute class weights
labels = [sample[1] for sample in train_dataset]
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

# Weighted Sampler
class_counts = np.bincount(labels)
sample_weights = [1.0 / class_counts[label] for _, label in train_dataset]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- Model, Loss, Optimizer ---
model = LungCancerModel(NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# --- Training Loop ---
best_val_loss = float('inf')
train_acc_hist, val_acc_hist = [], []

for epoch in range(EPOCHS):
    model.train()
    correct, total = 0, 0
    running_loss = 0.0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training"):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        running_loss += loss.item()

    train_accuracy = correct / total
    train_acc_hist.append(train_accuracy)

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_accuracy = val_correct / val_total
    val_acc_hist.append(val_accuracy)

    print(f"Epoch {epoch+1}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

    # --- Save Best ---
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_lung_model_pt.pth")
        print("Saved best model")

# --- Save Final Model ---
torch.save(model.state_dict(), "lung_model_pt_final.pth")
print("Final model saved to lung_model_pt_final.pth")

# --- Plot Accuracy ---
plt.figure(figsize=(10, 6))
plt.plot(train_acc_hist, label='Train Accuracy', marker='o')
plt.plot(val_acc_hist, label='Validation Accuracy', marker='x')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
