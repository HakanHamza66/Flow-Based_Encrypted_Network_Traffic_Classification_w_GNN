import torch
from data_loader import collect_labels, load_and_convert_all
from gnn_model import SimpleGNN
from train import train
from random import shuffle
from data_loader import collect_labels, load_and_convert_all, get_common_feature_set
from svm_utils import load_meta_features, train_svm
import time
from datetime import datetime
from confusion_matrix_analyzer import get_predictions, plot_confusion_matrix, calculate_metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import os

NUM_LAYERS_OPTIONS = [5]
HIDDEN_CHANNELS_OPTIONS = [512]
EPOCHS = 80
LEARNING_RATE = 0.001

csv_dir = "Data"
app_map, cat_map = collect_labels(csv_dir)
common_features = get_common_feature_set(csv_dir)
graph_list = load_and_convert_all(csv_dir, app_map, cat_map, common_features)
in_channels = graph_list[0].x.shape[1]

shuffle(graph_list)
train_size = int(0.7 * len(graph_list))
val_size = int(0.15 * len(graph_list))
train_data = graph_list[:train_size]
val_data = graph_list[train_size:train_size + val_size]
test_data = graph_list[train_size + val_size:]

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Create results file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = f"experiment_results_{timestamp}.txt"

# Run experiments for each parameter combination
for num_layers in NUM_LAYERS_OPTIONS:
    for hidden_channels in HIDDEN_CHANNELS_OPTIONS:
        print(f"\n{'='*50}")
        print(f"Testing with NUM_LAYERS={num_layers}, HIDDEN_CHANNELS={hidden_channels}")
        print(f"{'='*50}")
        
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Initialize model with current parameters
        model = SimpleGNN(
            in_channels=in_channels, 
            hidden_channels=hidden_channels, 
            out_channels=len(app_map),
            num_layers=num_layers
        )
        
        # Move model to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Train the model and get metrics
        start_time = time.time()
        metrics = train(model, train_data, val_data, epochs=EPOCHS, lr=LEARNING_RATE)
        end_time = time.time()
        training_time = end_time - start_time
          # Generate confusion matrix and metrics for applications
        app_names = [name for name, idx in sorted(app_map.items(), key=lambda x: x[1])]
        print("\n📊 Generating confusion matrix and metrics for applications...")
        
        # Get predictions and analyze
        y_true, y_pred = get_predictions(model, test_data)
        cm = confusion_matrix(y_true, y_pred)
        
        # Save confusion matrix plot with timestamp
        print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        plot_path = f'plots/confusion_matrix_{timestamp}.png'
        plot_confusion_matrix(cm, app_names, plot_path)
        print(f"\n✨ Confusion matrix saved to: {plot_path}")
        
        # Calculate and display per-class metrics
        metrics_df = calculate_metrics(y_true, y_pred, app_names)
        metrics_path = f'plots/class_metrics_{timestamp}.csv'
        metrics_df.to_csv(metrics_path, index=False)
        print(f"📈 Detailed class metrics saved to: {metrics_path}")
        
        # Display summary metrics
        total_correct = np.sum(np.diag(cm))
        total_samples = np.sum(cm)
        accuracy = total_correct / total_samples * 100
        print(f"\n🎯 Overall Test Accuracy: {accuracy:.2f}%")
        print(f"📊 Total Test Samples: {total_samples}")
          # Save results to file
        with open(results_file, 'a') as f:
            f.write(f"\nExperiment Configuration:\n")
            f.write(f"NUM_LAYERS: {num_layers}\n")
            f.write(f"HIDDEN_CHANNELS: {hidden_channels}\n")
            f.write(f"EPOCHS: {EPOCHS}\n")
            f.write(f"LEARNING_RATE: {LEARNING_RATE}\n")
            f.write(f"Training Time: {training_time:.2f} seconds\n")
            
            # Write training metrics
            if metrics:
                for metric_name, value in metrics.items():
                    f.write(f"{metric_name}: {value:.4f}\n")
            
            # Write confusion matrix metrics
            f.write("\nTest Set Results:\n")
            f.write(f"Overall Test Accuracy: {accuracy:.2f}%\n")
            f.write(f"Total Test Samples: {total_samples}\n")
            
            # Write per-class metrics
            f.write("\nPer-class Metrics:\n")
            f.write(metrics_df.to_string())
            f.write("\n")
            
            # Write confusion matrix
            f.write("\nConfusion Matrix:\n")
            class_names_str = "".join([f"{name:>15}" for name in app_names])
            f.write(f"{'':15}{class_names_str}\n")
            for i, row in enumerate(cm):
                f.write(f"{app_names[i]:15}" + "".join([f"{x:15d}" for x in row]) + "\n")
            
            f.write(f"\n{'='*50}\n")

print(f"\nAll experiments completed! Results saved to {results_file}")

# Run SVM part after all GNN experiments
print("\nRunning SVM comparison...")
X, y = load_meta_features("Data", label_type="appName")
svm_metrics = train_svm(X, y, label_name="appName")

print("\n📊 SVM Performance Summary:")
for k, v in svm_metrics.items():
    print(f"{k.capitalize()}: {v:.4f}")

# Save SVM results to file
with open(results_file, 'a') as f:
    f.write("\nSVM Results:\n")
    for k, v in svm_metrics.items():
        f.write(f"{k.capitalize()}: {v:.4f}\n")