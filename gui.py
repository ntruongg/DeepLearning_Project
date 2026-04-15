import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
from torchvision import transforms
from model import SimpleCNN, ResNetModel, DenseNetModel
import os

class TrafficSignGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Vietnamese Traffic Sign Recognition")
        self.root.geometry("600x500")

        # Model selection
        self.model_var = tk.StringVar(value="cnn")
        tk.Label(root, text="Select Model:").pack(pady=5)
        models_frame = tk.Frame(root)
        models_frame.pack()
        tk.Radiobutton(models_frame, text="CNN", variable=self.model_var, value="cnn").pack(side=tk.LEFT)
        tk.Radiobutton(models_frame, text="ResNet", variable=self.model_var, value="resnet").pack(side=tk.LEFT)
        tk.Radiobutton(models_frame, text="DenseNet", variable=self.model_var, value="densenet").pack(side=tk.LEFT)

        # Image selection
        tk.Label(root, text="Select Image:").pack(pady=5)
        self.image_button = tk.Button(root, text="Choose Image", command=self.select_image)
        self.image_button.pack()

        # Image display
        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

        # Predict button
        self.predict_button = tk.Button(root, text="Predict", command=self.predict)
        self.predict_button.pack(pady=10)

        # Result display
        self.result_label = tk.Label(root, text="", font=("Arial", 14))
        self.result_label.pack(pady=10)

        # Load class names
        self.class_names = []
        if os.path.exists('class_names.txt'):
            with open('class_names.txt', 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
        else:
            self.class_names = [f"Class {i}" for i in range(self.num_classes)]

        # Load models
        self.num_classes = 39  # Assuming 39 classes from the dataset
        self.models = {}
        self.load_models()

        # Image path
        self.image_path = None

    def load_models(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # CNN
        if os.path.exists('best_cnn_model.pth'):
            model_path = 'best_cnn_model.pth'
        elif os.path.exists('cnn_model.pth'):
            model_path = 'cnn_model.pth'
        else:
            model_path = None
        if model_path:
            self.models['cnn'] = SimpleCNN(self.num_classes)
            self.models['cnn'].load_state_dict(torch.load(model_path, map_location='cpu'))
            self.models['cnn'].to(device)
            self.models['cnn'].eval()

        # ResNet
        if os.path.exists('best_resnet_model.pth'):
            model_path = 'best_resnet_model.pth'
        elif os.path.exists('resnet_model.pth'):
            model_path = 'resnet_model.pth'
        else:
            model_path = None
        if model_path:
            self.models['resnet'] = ResNetModel(self.num_classes)
            self.models['resnet'].load_state_dict(torch.load(model_path, map_location='cpu'))
            self.models['resnet'].to(device)
            self.models['resnet'].eval()
 
        # DenseNet
        if os.path.exists('best_densenet_model.pth'):
            model_path = 'best_densenet_model.pth'
        elif os.path.exists('densenet_model.pth'):
            model_path = 'densenet_model.pth'
        else:
            model_path = None
        if model_path:
            self.models['densenet'] = DenseNetModel(self.num_classes)
            self.models['densenet'].load_state_dict(torch.load(model_path, map_location='cpu'))
            self.models['densenet'].to(device)
            self.models['densenet'].eval()

    def select_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if self.image_path:
            image = Image.open(self.image_path)
            image = image.resize((200, 200), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo

    def predict(self):
        if not self.image_path:
            messagebox.showerror("Error", "Please select an image first.")
            return

        model_name = self.model_var.get()
        if model_name not in self.models:
            messagebox.showerror("Error", f"Model {model_name} not found. Please train the model first.")
            return

        # Transform image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = Image.open(self.image_path).convert('RGB')
        image = transform(image).unsqueeze(0)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        image = image.to(device)

        model = self.models[model_name]
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            predicted_class = predicted.item()

        self.result_label.config(text=f"Predicted Class: {self.class_names[predicted_class] if predicted_class < len(self.class_names) else f'Class {predicted_class}'}")

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficSignGUI(root)
    root.mainloop()