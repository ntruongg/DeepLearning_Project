# Vietnamese Traffic Sign Recognition

This project implements traffic sign recognition for Vietnamese signs using CNN, ResNet, and DenseNet models.

## Dataset

Download the Vietnamese Traffic Signs dataset from Kaggle: https://www.kaggle.com/datasets/nguyenquyhcmus/vietnamese-traffic-signs

### Download Instructions

1. Install Kaggle CLI:
   ```
   pip install kaggle
   ```

2. Download the dataset:
   ```
   kaggle datasets download -d nguyenquyhcmus/vietnamese-traffic-signs
   unzip vietnamese-traffic-signs.zip
   ```

3. Organize the data into `train/`, `val/`, `test/` folders with class subfolders.

## Installation

```
pip install -r requirements.txt
```

## Training

Train a model (data should be in 'data' folder):

```
python train.py --model cnn --epochs 20
```

Models: `cnn`, `resnet`, `densenet`

## GUI Testing

Run the GUI for easy testing:

```
python gui.py
```

Select model, choose an image, and click Predict to see the result with class name.

## Command Line Testing

Use inference.py for command line prediction:

```
python inference.py --model resnet --model_path resnet_model.pth --image_path path/to/image.jpg
```

This will output the predicted class name.

## Improving Accuracy

If the model doesn't recognize signs correctly:

1. **Train longer**: Increase epochs (e.g., 50-100)
2. **Use better models**: DenseNet or ResNet perform better than simple CNN
3. **Data augmentation**: Added random flips, rotations, color jitter for training
4. **Fine-tune pre-trained models**: ResNet and DenseNet use ImageNet weights
5. **Check data quality**: Ensure images are clear and correctly labeled
6. **Hyperparameter tuning**: Adjust learning rate, batch size

## Training Tips

- Use GPU if available for faster training
- Monitor validation accuracy to avoid overfitting
- Early stopping saves the best model automatically (best_*_model.pth)