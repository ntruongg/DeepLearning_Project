import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

def get_data_loaders(data_dir, batch_size=32, image_size=224):
    # Define transforms with augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load train dataset
    train_path = os.path.join(data_dir, 'train')
    if os.path.exists(train_path) and os.listdir(train_path):
        train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
        num_classes = len(train_dataset.classes)
    else:
        raise FileNotFoundError("Train folder not found or empty")

    # Load val dataset if exists
    val_path = os.path.join(data_dir, 'val')
    if os.path.exists(val_path) and os.listdir(val_path):
        val_dataset = datasets.ImageFolder(root=val_path, transform=val_test_transform)
    else:
        # Split train into train/val
        total_size = len(train_dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        # Re-apply transforms
        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = val_test_transform

    # Load test dataset if exists
    test_path = os.path.join(data_dir, 'test')
    if os.path.exists(test_path) and os.listdir(test_path):
        test_dataset = datasets.ImageFolder(root=test_path, transform=val_test_transform)
    else:
        # Use val as test
        test_dataset = val_dataset

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, num_classes