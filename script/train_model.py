# train_model.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from preprocess import create_data_loaders  # Make sure preprocess.py is in the correct location

# Define the CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Convolutional Blocks
        self.features = nn.Sequential(
            # Block 1: Input 3 -> 32 filters
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # Reduces dimensions by half

            # Block 2: 32 -> 32 filters
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 32 -> 64 filters
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 64 -> 128 filters
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # Calculate the feature map size after four max pooling layers.
        # Assuming input image size is 224x224, after four poolings, size becomes 224/16 = 14.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()  # For binary classification
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Training function
def train_model(model, train_loader, val_loader, num_epochs=20, learning_rate=1e-4, device='cpu'):
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            # Reshape labels to match output shape: [batch_size, 1]
            labels = labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)

        # Evaluate on the validation set
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('../models', exist_ok=True)
            model_path = '../models/cataract_model.pth'
            torch.save(model.state_dict(), model_path)
            print(f"Best model saved at epoch {epoch+1} with validation loss {val_loss:.4f}")
            
    print("Training complete.")

if __name__ == '__main__':
    # Set device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set the base directory where your dataset is located
    base_dir = os.path.join(os.getcwd(), '..', 'dataset')
    
    # Create data loaders (train, validation, test)
    train_loader, val_loader, test_loader = create_data_loaders(base_dir, img_size=224, batch_size=32)
    
    # Initialize the model
    model = CNNModel()
    
    # Train the model
    train_model(model, train_loader, val_loader, num_epochs=25, learning_rate=1e-4, device=device)
