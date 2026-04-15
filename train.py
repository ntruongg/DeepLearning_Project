import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleCNN, ResNetModel, DenseNetModel
from data_loader import get_data_loaders
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleCNN, ResNetModel, DenseNetModel
from data_loader import get_data_loaders
import argparse
from torch.optim.lr_scheduler import StepLR

def train_model(model, train_loader, val_loader, num_epochs=10, device='cuda', patience=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    model.to(device)

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')

        scheduler.step()

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), f'best_{args.model}_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'resnet', 'densenet'])
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, test_loader, num_classes = get_data_loaders(args.data_dir)

    if args.model == 'cnn':
        model = SimpleCNN(num_classes)
    elif args.model == 'resnet':
        model = ResNetModel(num_classes)
    else:
        model = DenseNetModel(num_classes)

    trained_model = train_model(model, train_loader, val_loader, args.epochs, device)

    # Save final model
    torch.save(trained_model.state_dict(), f'{args.model}_model.pth')