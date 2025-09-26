# Neural Networks Deep Dive

## Learning Objectives
- Master different types of neural network layers
- Understand activation functions and their properties
- Learn about loss functions and optimization
- Practice with advanced network architectures

## Neural Network Layers

### Fully Connected (Linear) Layers
```python
import torch
import torch.nn as nn

# Basic linear layer
linear_layer = nn.Linear(in_features=784, out_features=128)
print(f"Weight shape: {linear_layer.weight.shape}")  # [128, 784]
print(f"Bias shape: {linear_layer.bias.shape}")      # [128]

# Forward pass
input_tensor = torch.randn(32, 784)  # Batch of 32 samples
output = linear_layer(input_tensor)
print(f"Output shape: {output.shape}")  # [32, 128]
```

### Convolutional Layers
```python
# 2D Convolutional layer
conv2d = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
print(f"Conv2d weight shape: {conv2d.weight.shape}")  # [64, 3, 3, 3]

# Forward pass
input_image = torch.randn(1, 3, 32, 32)  # Batch, Channels, Height, Width
output = conv2d(input_image)
print(f"Conv2d output shape: {output.shape}")  # [1, 64, 32, 32]

# 1D Convolutional layer (for sequences)
conv1d = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
sequence_input = torch.randn(32, 128, 100)  # Batch, Features, Sequence Length
output = conv1d(sequence_input)
print(f"Conv1d output shape: {output.shape}")  # [32, 64, 100]
```

### Recurrent Layers
```python
# LSTM layer
lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=2, batch_first=True)
print(f"LSTM parameters: {sum(p.numel() for p in lstm.parameters())}")

# Forward pass
sequence_input = torch.randn(32, 50, 128)  # Batch, Sequence Length, Features
output, (hidden, cell) = lstm(sequence_input)
print(f"LSTM output shape: {output.shape}")  # [32, 50, 64]
print(f"Hidden state shape: {hidden.shape}")  # [2, 32, 64]

# GRU layer (simpler than LSTM)
gru = nn.GRU(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
output, hidden = gru(sequence_input)
print(f"GRU output shape: {output.shape}")  # [32, 50, 64]
```

### Pooling and Normalization Layers
```python
# Max pooling
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
input_tensor = torch.randn(1, 64, 32, 32)
pooled = max_pool(input_tensor)
print(f"MaxPool output shape: {pooled.shape}")  # [1, 64, 16, 16]

# Average pooling
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
avg_pooled = avg_pool(input_tensor)
print(f"AvgPool output shape: {avg_pooled.shape}")  # [1, 64, 16, 16]

# Batch normalization
batch_norm = nn.BatchNorm2d(num_features=64)
normalized = batch_norm(input_tensor)
print(f"BatchNorm output shape: {normalized.shape}")  # [1, 64, 32, 32]

# Layer normalization
layer_norm = nn.LayerNorm(normalized_shape=64)
layer_normalized = layer_norm(input_tensor.view(1, 64, -1))
print(f"LayerNorm output shape: {layer_normalized.shape}")  # [1, 64, 1024]
```

## Activation Functions

### Common Activation Functions
```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Define activation functions
activations = {
    'ReLU': nn.ReLU(),
    'LeakyReLU': nn.LeakyReLU(0.1),
    'Sigmoid': nn.Sigmoid(),
    'Tanh': nn.Tanh(),
    'GELU': nn.GELU(),
    'Swish': nn.SiLU(),  # SiLU is the same as Swish
}

# Test on sample data
x = torch.linspace(-5, 5, 100)
results = {}

for name, activation in activations.items():
    results[name] = activation(x)

# Plot activation functions
plt.figure(figsize=(12, 8))
for i, (name, y) in enumerate(results.items()):
    plt.subplot(2, 3, i+1)
    plt.plot(x.numpy(), y.numpy())
    plt.title(name)
    plt.grid(True)
    plt.xlabel('Input')
    plt.ylabel('Output')

plt.tight_layout()
plt.show()
```

### Custom Activation Functions
```python
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
    
    def forward(self, x):
        return x * torch.sigmoid(x)

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
    
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))

# Usage
swish = Swish()
mish = Mish()

x = torch.randn(10)
print(f"Swish output: {swish(x)}")
print(f"Mish output: {mish(x)}")
```

## Loss Functions

### Classification Losses
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Binary classification
binary_criterion = nn.BCELoss()
binary_logits = torch.randn(10)
binary_targets = torch.randint(0, 2, (10,)).float()
binary_probs = torch.sigmoid(binary_logits)
binary_loss = binary_criterion(binary_probs, binary_targets)

# Multi-class classification
ce_criterion = nn.CrossEntropyLoss()
logits = torch.randn(10, 5)  # 10 samples, 5 classes
targets = torch.randint(0, 5, (10,))
ce_loss = ce_criterion(logits, targets)

# Focal loss (for imbalanced datasets)
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

focal_criterion = FocalLoss()
focal_loss = focal_criterion(logits, targets)
```

### Regression Losses
```python
# Mean Squared Error
mse_criterion = nn.MSELoss()
predictions = torch.randn(10, 1)
targets = torch.randn(10, 1)
mse_loss = mse_criterion(predictions, targets)

# Mean Absolute Error
mae_criterion = nn.L1Loss()
mae_loss = mae_criterion(predictions, targets)

# Huber Loss (robust to outliers)
huber_criterion = nn.SmoothL1Loss()
huber_loss = huber_criterion(predictions, targets)

# Custom loss function
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
    
    def forward(self, predictions, targets):
        mse = F.mse_loss(predictions, targets)
        mae = F.l1_loss(predictions, targets)
        return 0.7 * mse + 0.3 * mae

custom_criterion = CustomLoss()
custom_loss = custom_criterion(predictions, targets)
```

## Optimizers

### Common Optimizers
```python
import torch.optim as optim

# Stochastic Gradient Descent
sgd_optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# Adam optimizer
adam_optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-4)

# AdamW optimizer (better weight decay)
adamw_optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# RMSprop optimizer
rmsprop_optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)

# Learning rate schedulers
scheduler = optim.lr_scheduler.StepLR(adam_optimizer, step_size=30, gamma=0.1)
cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(adam_optimizer, T_max=100)
```

### Custom Optimizer
```python
class CustomOptimizer(optim.Optimizer):
    def __init__(self, params, lr=0.01, beta=0.9):
        defaults = dict(lr=lr, beta=beta)
        super(CustomOptimizer, self).__init__(params, defaults)
    
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                
                exp_avg = state['exp_avg']
                beta = group['beta']
                lr = group['lr']
                
                state['step'] += 1
                exp_avg.mul_(beta).add_(grad, alpha=1-beta)
                
                p.data.add_(exp_avg, alpha=-lr)
```

## Advanced Network Architectures

### Residual Network (ResNet) Block
```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out
```

### Attention Mechanism
```python
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (self.head_dim * heads == embed_size), "Embed size needs to be divisible by heads"
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
    
    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        
        # Attention calculation
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        
        out = self.fc_out(out)
        return out
```

## Practice Exercises

### Exercise 1: Custom CNN Architecture
Build a convolutional neural network that:
- Uses multiple Conv2d layers with different kernel sizes
- Implements batch normalization and dropout
- Includes residual connections
- Classifies CIFAR-10 dataset

### Exercise 2: LSTM for Sequence Prediction
Create an LSTM model that:
- Predicts the next value in a time series
- Uses multiple LSTM layers with dropout
- Implements early stopping
- Visualizes predictions vs actual values

### Exercise 3: Attention-based Model
Implement a model with:
- Self-attention mechanism
- Multi-head attention
- Positional encoding
- Task-specific output layer

### Exercise 4: Custom Loss Function
Design a loss function that:
- Combines multiple loss components
- Handles class imbalance
- Includes regularization terms
- Is differentiable and efficient

## Common Patterns and Best Practices

### Model Initialization
```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

# Apply initialization
model.apply(init_weights)
```

### Gradient Clipping
```python
# Clip gradients to prevent exploding gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Model Checkpointing
```python
def save_checkpoint(model, optimizer, epoch, loss, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath, model, optimizer):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss
```

## AI Learning Prompt

**Copy this prompt into ChatGPT or any AI chatbot:**

"I'm learning about neural network architectures and components in PyTorch. I understand basic PyTorch but I'm struggling with:

1. Different types of layers and when to use them
2. How activation functions affect network behavior
3. Choosing appropriate loss functions for different tasks
4. Understanding optimizers and their hyperparameters
5. Building complex architectures like ResNet and attention mechanisms
6. Best practices for model initialization and training

Please:
- Explain each component with practical examples
- Show me how to build different types of networks
- Help me understand the mathematics behind each component
- Walk me through implementing advanced architectures
- Give me exercises to practice building custom models
- Explain common pitfalls and how to avoid them

I want to build sophisticated neural networks for real-world applications. Please provide hands-on examples and help me understand the design principles."

## Key Takeaways
- Different layers serve different purposes in neural networks
- Activation functions introduce non-linearity and affect gradient flow
- Loss functions should match your specific task and data distribution
- Optimizers control how the network learns and converges
- Advanced architectures like ResNet and attention solve specific problems
- Proper initialization and training practices are crucial for success

## Next Steps
Master these neural network concepts and you'll be ready for:
- Computer vision with CNNs
- Natural language processing with transformers
- Generative models and GANs
- Reinforcement learning
- Model optimization and deployment
