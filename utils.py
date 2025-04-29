import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, WeightedRandomSampler

def compute_class_weights(train_dataset, device):
    """
    Computes class weights for handling class imbalance.
    Args:
    - train_dataset (torchvision.datasets.ImageFolder): The training dataset.
    - device (torch.device): The device (CPU/GPU) to transfer tensors to.
    
    Returns:
    - class_weights_tensor (torch.Tensor): A tensor containing class weights.
    """
    labels = [sample[1] for sample in train_dataset]
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    return class_weights_tensor

def get_weighted_sampler(train_dataset):
    """
    Creates a WeightedRandomSampler to balance class distribution in the dataset.
    Args:
    - train_dataset (torchvision.datasets.ImageFolder): The training dataset.
    
    Returns:
    - sampler (torch.utils.data.WeightedRandomSampler): The weighted random sampler.
    """
    labels = [sample[1] for sample in train_dataset]
    class_counts = np.bincount(labels)
    sample_weights = [1.0 / class_counts[label] for _, label in train_dataset]
    
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    return sampler

def plot_accuracy(train_acc_hist, val_acc_hist, save_path="accuracy_plot.png"):
    """
    Plots and saves the training vs validation accuracy over epochs.
    Args:
    - train_acc_hist (list): List of training accuracies over epochs.
    - val_acc_hist (list): List of validation accuracies over epochs.
    - save_path (str): Path to save the accuracy plot image.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_acc_hist, label='Train Accuracy', marker='o')
    plt.plot(val_acc_hist, label='Validation Accuracy', marker='x')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()

def save_model(model, save_path):
    """
    Saves the model weights to the specified path.
    Args:
    - model (torch.nn.Module): The model to save.
    - save_path (str): Path to save the model weights.
    """
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def load_model(model, load_path, device):
    """
    Loads model weights from the specified path.
    Args:
    - model (torch.nn.Module): The model to load the weights into.
    - load_path (str): Path to load the model weights from.
    - device (torch.device): The device (CPU/GPU) to load the model onto.
    
    Returns:
    - model (torch.nn.Module): The model with loaded weights.
    """
    model.load_state_dict(torch.load(load_path, map_location=device))
    print(f"Model loaded from {load_path}")
    return model

def evaluate_model(model, test_loader, device):
    """
    Evaluates the model on the test dataset and prints classification metrics.
    Args:
    - model (torch.nn.Module): The model to evaluate.
    - test_loader (torch.utils.data.DataLoader): The DataLoader for the test dataset.
    - device (torch.device): The device (CPU/GPU) to perform evaluation on.
    
    Returns:
    - cm (numpy.ndarray): The confusion matrix.
    - classification_rep (str): The classification report.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(all_labels, all_preds)
    classification_rep = classification_report(all_labels, all_preds, target_names=test_loader.dataset.classes)
    
    return cm, classification_rep
