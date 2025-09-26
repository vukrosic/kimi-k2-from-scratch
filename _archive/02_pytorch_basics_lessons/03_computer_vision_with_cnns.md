# Computer Vision with CNNs

## Learning Objectives
- Master Convolutional Neural Networks (CNNs)
- Understand image preprocessing and augmentation
- Learn transfer learning and pre-trained models
- Practice with real computer vision tasks

## Understanding Convolutions

### Basic Convolution Operation
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Create a simple 2D convolution
conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)

# Create a sample image (5x5)
image = torch.tensor([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

print(f"Input shape: {image.shape}")

# Apply convolution
output = conv2d(image)
print(f"Output shape: {output.shape}")

# Manual convolution example
def manual_conv2d(input_tensor, kernel):
    """Manual 2D convolution implementation"""
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = kernel.shape
    
    # Calculate output dimensions
    out_height = height - kernel_height + 1
    out_width = width - kernel_width + 1
    
    # Initialize output tensor
    output = torch.zeros(batch_size, out_channels, out_height, out_width)
    
    # Perform convolution
    for b in range(batch_size):
        for c_out in range(out_channels):
            for h in range(out_height):
                for w in range(out_width):
                    # Extract patch
                    patch = input_tensor[b, :, h:h+kernel_height, w:w+kernel_width]
                    # Apply kernel and sum
                    output[b, c_out, h, w] = torch.sum(patch * kernel[c_out])
    
    return output

# Test manual convolution
kernel = torch.randn(1, 1, 3, 3)
manual_output = manual_conv2d(image, kernel)
print(f"Manual convolution output shape: {manual_output.shape}")
```

### Convolution Visualization
```python
def visualize_convolution():
    # Create a simple image with a pattern
    image = torch.zeros(1, 1, 7, 7)
    image[0, 0, 2:5, 2:5] = 1  # Create a 3x3 white square
    
    # Create different kernels
    kernels = {
        'Identity': torch.tensor([[[[0, 0, 0], [0, 1, 0], [0, 0, 0]]]], dtype=torch.float32),
        'Edge Detection': torch.tensor([[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]], dtype=torch.float32),
        'Blur': torch.tensor([[[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]], dtype=torch.float32) / 9,
        'Sharpen': torch.tensor([[[[0, -1, 0], [-1, 5, -1], [0, -1, 0]]]], dtype=torch.float32)
    }
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    
    # Original image
    axes[0, 0].imshow(image[0, 0].numpy(), cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Apply each kernel
    for i, (name, kernel) in enumerate(kernels.items()):
        conv = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        conv.weight.data = kernel
        
        output = conv(image)
        
        axes[0, i+1].imshow(output[0, 0].detach().numpy(), cmap='gray')
        axes[0, i+1].set_title(f'{name} Kernel')
        axes[0, i+1].axis('off')
        
        # Show kernel
        axes[1, i+1].imshow(kernel[0, 0].numpy(), cmap='gray')
        axes[1, i+1].set_title(f'{name} Kernel')
        axes[1, i+1].axis('off')
    
    plt.tight_layout()
    plt.show()

# Run visualization
visualize_convolution()
```

## Building CNN Architectures

### Simple CNN for Image Classification
```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # Assuming 32x32 input
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
    
    def forward(self, x):
        # First conv block
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Second conv block
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Third conv block
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Create model
model = SimpleCNN(num_classes=10)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Advanced CNN with Residual Connections
```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out

class ResNetCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetCNN, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

# Create ResNet model
resnet_model = ResNetCNN(num_classes=10)
print(f"ResNet parameters: {sum(p.numel() for p in resnet_model.parameters()):,}")
```

## Image Preprocessing and Augmentation

### Data Transforms
```python
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

# Basic transforms
basic_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Training transforms with augmentation
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation transforms (no augmentation)
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom transform for specific tasks
class CustomTransform:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, image):
        # Convert to tensor
        image = transforms.ToTensor()(image)
        
        # Custom normalization
        image = (image - self.mean) / self.std
        
        # Add noise
        noise = torch.randn_like(image) * 0.1
        image = image + noise
        
        return image

# Apply transforms to dataset
def get_data_loaders(batch_size=32):
    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transforms
    )
    
    val_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=val_transforms
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return train_loader, val_loader
```

### Advanced Augmentation Techniques
```python
import torchvision.transforms.functional as TF
import random

class AdvancedAugmentation:
    def __init__(self):
        self.rotation_angles = [-15, -10, -5, 5, 10, 15]
        self.brightness_factors = [0.8, 0.9, 1.0, 1.1, 1.2]
        self.contrast_factors = [0.8, 0.9, 1.0, 1.1, 1.2]
    
    def __call__(self, image):
        # Random rotation
        if random.random() > 0.5:
            angle = random.choice(self.rotation_angles)
            image = TF.rotate(image, angle)
        
        # Random brightness adjustment
        if random.random() > 0.5:
            brightness = random.choice(self.brightness_factors)
            image = TF.adjust_brightness(image, brightness)
        
        # Random contrast adjustment
        if random.random() > 0.5:
            contrast = random.choice(self.contrast_factors)
            image = TF.adjust_contrast(image, contrast)
        
        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
        
        # Random vertical flip (for some datasets)
        if random.random() > 0.8:
            image = TF.vflip(image)
        
        return image

# Mixup augmentation
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
```

## Transfer Learning

### Using Pre-trained Models
```python
import torchvision.models as models

# Load pre-trained ResNet
def get_pretrained_resnet(num_classes=10, freeze_backbone=False):
    # Load pre-trained model
    model = models.resnet18(pretrained=True)
    
    # Freeze backbone if specified
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace the final layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

# Load pre-trained VGG
def get_pretrained_vgg(num_classes=10, freeze_backbone=False):
    model = models.vgg16(pretrained=True)
    
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
    
    # Modify classifier
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    
    return model

# Load pre-trained EfficientNet
def get_pretrained_efficientnet(num_classes=10, freeze_backbone=False):
    model = models.efficientnet_b0(pretrained=True)
    
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
    
    # Replace classifier
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    
    return model

# Create models
resnet_model = get_pretrained_resnet(num_classes=10, freeze_backbone=True)
vgg_model = get_pretrained_vgg(num_classes=10, freeze_backbone=False)
efficientnet_model = get_pretrained_efficientnet(num_classes=10, freeze_backbone=True)
```

### Fine-tuning Strategies
```python
def fine_tune_model(model, epochs=10, learning_rate=0.001):
    # Different learning rates for different parts
    backbone_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'classifier' in name or 'fc' in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)
    
    # Create optimizer with different learning rates
    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': learning_rate * 0.1},
        {'params': classifier_params, 'lr': learning_rate}
    ])
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    return optimizer, scheduler

# Progressive unfreezing
def progressive_unfreezing(model, epoch, total_epochs):
    if epoch < total_epochs // 3:
        # Freeze all layers except classifier
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif epoch < 2 * total_epochs // 3:
        # Unfreeze last few layers
        for param in model.features[-2:].parameters():
            param.requires_grad = True
    else:
        # Unfreeze all layers
        for param in model.parameters():
            param.requires_grad = True
```

## Practice Exercises

### Exercise 1: CIFAR-10 Classification
Build a CNN that:
- Classifies CIFAR-10 images
- Uses data augmentation
- Implements early stopping
- Achieves >85% accuracy

### Exercise 2: Custom Dataset Classification
Create a model that:
- Works with your own image dataset
- Uses transfer learning
- Implements data augmentation
- Includes model evaluation metrics

### Exercise 3: Object Detection
Implement a simple object detector that:
- Uses a pre-trained backbone
- Adds detection heads
- Trains on a custom dataset
- Evaluates detection performance

### Exercise 4: Image Segmentation
Build a segmentation model that:
- Uses U-Net architecture
- Implements skip connections
- Trains on segmentation data
- Visualizes segmentation results

## Common Computer Vision Patterns

### Model Ensemble
```python
class ModelEnsemble(nn.Module):
    def __init__(self, models):
        super(ModelEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
    
    def forward(self, x):
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # Average predictions
        ensemble_pred = torch.stack(predictions).mean(dim=0)
        return ensemble_pred

# Create ensemble
models = [
    get_pretrained_resnet(num_classes=10),
    get_pretrained_vgg(num_classes=10),
    get_pretrained_efficientnet(num_classes=10)
]
ensemble = ModelEnsemble(models)
```

### Test Time Augmentation (TTA)
```python
def test_time_augmentation(model, image, num_augmentations=5):
    model.eval()
    predictions = []
    
    # Original image
    with torch.no_grad():
        pred = model(image)
        predictions.append(pred)
    
    # Augmented images
    for _ in range(num_augmentations):
        # Random augmentation
        augmented = TF.rotate(image, random.uniform(-10, 10))
        augmented = TF.adjust_brightness(augmented, random.uniform(0.9, 1.1))
        
        with torch.no_grad():
            pred = model(augmented)
            predictions.append(pred)
    
    # Average predictions
    final_pred = torch.stack(predictions).mean(dim=0)
    return final_pred
```

## AI Learning Prompt

**Copy this prompt into ChatGPT or any AI chatbot:**

"I'm learning computer vision with CNNs in PyTorch. I understand basic neural networks but I'm struggling with:

1. How convolutions work and why they're effective for images
2. Building CNN architectures for different tasks
3. Image preprocessing and data augmentation techniques
4. Transfer learning and using pre-trained models
5. Fine-tuning strategies and hyperparameter tuning
6. Common computer vision patterns and best practices

Please:
- Explain convolution operations with visual examples
- Show me how to build different CNN architectures
- Help me understand data augmentation and its importance
- Walk me through transfer learning step by step
- Give me exercises to practice with real datasets
- Explain common pitfalls and optimization techniques

I want to build effective computer vision models for real-world applications. Please provide hands-on examples and help me understand the design principles."

## Key Takeaways
- Convolutions are powerful for extracting spatial features from images
- CNN architectures should be designed based on the specific task
- Data augmentation is crucial for preventing overfitting
- Transfer learning can significantly improve performance with limited data
- Fine-tuning strategies depend on dataset size and similarity to pre-trained data
- Ensemble methods and TTA can improve model robustness

## Next Steps
Master computer vision with CNNs and you'll be ready for:
- Object detection and localization
- Image segmentation
- Generative models for images
- Video analysis and understanding
- Medical image analysis
- Autonomous driving applications
