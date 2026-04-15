import torch
from torchvision import transforms
from PIL import Image
from model import SimpleCNN, ResNetModel, DenseNetModel
import argparse
import os

def load_class_names():
    if os.path.exists('class_names.txt'):
        with open('class_names.txt', 'r') as f:
            return [line.strip() for line in f.readlines()]
    else:
        return [f"Class {i}" for i in range(39)]  # Assuming 39 classes

def load_model(model_type, num_classes, model_path):
    if model_type == 'cnn':
        model = SimpleCNN(num_classes)
        if os.path.exists('best_cnn_model.pth'):
            model_path = 'best_cnn_model.pth'
    elif model_type == 'resnet':
        model = ResNetModel(num_classes)
        if os.path.exists('best_resnet_model.pth'):
            model_path = 'best_resnet_model.pth'
    else:
        model = DenseNetModel(num_classes)
        if os.path.exists('best_densenet_model.pth'):
            model_path = 'best_densenet_model.pth'
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def predict_image(model, image_path, device='cpu'):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['cnn', 'resnet', 'densenet'])
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--num_classes', type=int, default=39)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model, args.num_classes, args.model_path).to(device)

    prediction = predict_image(model, args.image_path, device)
    class_names = load_class_names()
    class_name = class_names[prediction] if prediction < len(class_names) else f'Class {prediction}'
    print(f'Predicted class: {class_name}')