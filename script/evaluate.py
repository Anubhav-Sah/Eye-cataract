# evaluate.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from preprocess import create_data_loaders
from train_model import CNNModel

def evaluate_model(model, test_loader, device='cpu'):
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    all_outputs = []  # To optionally store the raw sigmoid outputs
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            # Store raw outputs if needed for further analysis
            all_outputs.extend(outputs.cpu().numpy().flatten())
            # Convert sigmoid output probabilities to binary predictions (0 or 1)
            preds = (outputs.cpu().numpy() > 0.5).astype(int)
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate overall accuracy
    accuracy = np.mean(all_preds == all_labels)
    print(f"Test Accuracy: {accuracy*100:.4f}")
    
    # Calculate confusion matrix and classification report
    cm = confusion_matrix(all_labels, all_preds)
    cr = classification_report(all_labels, all_preds, target_names=['Normal', 'Cataract'])
    
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr)
    
    # Plot confusion matrix
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Normal', 'Cataract'], rotation=45)
    plt.yticks(tick_marks, ['Normal', 'Cataract'])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()
    
    # Compute and plot cumulative accuracy over test samples
    # This shows how accuracy evolves as more test samples are considered.
    correct = (all_preds == all_labels).astype(int)
    cumulative_accuracy = np.cumsum(correct) / np.arange(1, len(correct) + 1)
    
    plt.figure(figsize=(10, 5))
    plt.plot(cumulative_accuracy, 'b-', label="Cumulative Accuracy")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Cumulative Accuracy")
    plt.title("Cumulative Accuracy over Test Samples")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Set the base directory for your dataset
    base_dir = os.path.join(os.getcwd(), '..', 'dataset')
    
    # Create test data loader
    _, _, test_loader = create_data_loaders(base_dir, img_size=224, batch_size=32)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize and load the trained model
    model = CNNModel()
    model_path = os.path.join(os.getcwd(), '..', 'models', 'cataract_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Evaluate the model on the test data
    evaluate_model(model, test_loader, device=device)
