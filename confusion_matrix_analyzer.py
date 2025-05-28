import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch_geometric.loader import DataLoader
import pandas as pd

def get_predictions(model, test_data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    
    return np.array(y_true), np.array(y_pred)

def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(20, 16))
    ax = plt.gca()
    
    # Calculate percentages and handle division by zero
    total_per_class = cm.sum(axis=1)
    cm_normalized = np.zeros_like(cm, dtype=float)
    
    for i in range(cm.shape[0]):
        if total_per_class[i] > 0:
            cm_normalized[i] = cm[i] / total_per_class[i] * 100
    
    # Create custom annotation text
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            percentage = cm_normalized[i, j]
            count = cm[i, j]
            annot[i, j] = f'{percentage:.1f}%\n{count}'
    
    # Create heatmap
    sns.heatmap(cm_normalized, annot=annot, fmt='', 
                xticklabels=class_names, yticklabels=class_names,
                cbar=False, square=True, cmap='Blues',
                annot_kws={'size': 10, 'weight': 'bold'})
    
    # Add title and labels
    plt.title('Application Classification Confusion Matrix', fontsize=16, pad=20)
    plt.xlabel('Predicted Application', fontsize=14, labelpad=10)
    plt.ylabel('True Application', fontsize=14, labelpad=10)
    
    # Rotate labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Remove gridlines
    ax.grid(False)
    
    # Add overall accuracy
    total_correct = np.sum(np.diag(cm))
    total_samples = np.sum(cm)
    accuracy = total_correct / total_samples * 100
    plt.figtext(0.02, 0.97, f'Overall Accuracy: {accuracy:.2f}%\nTotal Samples: {total_samples}',
                fontsize=10, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def calculate_metrics(y_true, y_pred, class_names):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    
    metrics_df = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })
    
    return metrics_df