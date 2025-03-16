# preprocess.py

import os
from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CataractDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths (list): List of file paths to images.
            labels (list): Corresponding labels (e.g., 1 for cataract, 0 for normal).
            transform (callable, optional): Optional transform to be applied
                on an image.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        # Open the image and convert to RGB
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def create_data_loaders(base_dir, img_size=224, batch_size=32, test_size=0.2, val_size=0.1):
    """
    Splits data into train, validation, and test sets; applies resizing, normalization, and augmentation.

    Args:
        base_dir (str): Base directory where the dataset folders ('cataract' and 'normal') are stored.
        img_size (int): Target size for resizing images (img_size x img_size).
        batch_size (int): Batch size for DataLoaders.
        test_size (float): Fraction of data to reserve for testing.
        val_size (float): Fraction of the training data to reserve for validation.
    
    Returns:
        Tuple of DataLoaders: (train_loader, val_loader, test_loader)
    """
    # Gather image paths
    cataract_paths = glob(os.path.join(base_dir, 'cataract', '*'))
    normal_paths = glob(os.path.join(base_dir, 'normal', '*'))

    # Create labels: 1 for cataract images, 0 for normal images
    cataract_labels = [1] * len(cataract_paths)
    normal_labels = [0] * len(normal_paths)

    # Combine paths and labels
    all_paths = cataract_paths + normal_paths
    all_labels = cataract_labels + normal_labels

    # First split into training and test sets
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        all_paths, all_labels, test_size=test_size, stratify=all_labels, random_state=42
    )

    # Split training further into training and validation sets
    # Calculate validation fraction relative to remaining data
    val_fraction = val_size / (1 - test_size)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=val_fraction, stratify=train_labels, random_state=42
    )

    # Define transformation pipelines
    # For training: apply resizing, data augmentation, and normalization
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                             std=[0.229, 0.224, 0.225])
    ])

    # For validation and testing: only resize and normalize
    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Create Dataset objects
    train_dataset = CataractDataset(train_paths, train_labels, transform=train_transform)
    val_dataset   = CataractDataset(val_paths, val_labels, transform=val_test_transform)
    test_dataset  = CataractDataset(test_paths, test_labels, transform=val_test_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    # Set the base directory to your dataset folder
    # Example structure: project_root/dataset/cataract and project_root/dataset/normal
    base_dir = os.path.join(os.getcwd(), '..', 'dataset')
    train_loader, val_loader, test_loader = create_data_loaders(base_dir)
    print("Train, validation, and test loaders created successfully.")
    print(f"Train set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(val_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")
