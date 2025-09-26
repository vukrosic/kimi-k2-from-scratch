# PyTorch Fundamentals

## Learning Objectives
- Understand what PyTorch is and why it's used
- Learn about tensors and basic operations
- Master PyTorch's autograd system
- Practice with simple neural networks

## What is PyTorch?
PyTorch is an open-source machine learning library that provides:
- **Dynamic computation graphs** - Build networks on-the-fly
- **GPU acceleration** - Leverage CUDA for fast training
- **Pythonic interface** - Intuitive and easy to use
- **Research-friendly** - Flexible for experimentation
- **Production-ready** - TorchScript for deployment

## Installation and Setup

### Installing PyTorch
```bash
# CPU version
pip install torch torchvision torchaudio

# GPU version (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Check installation
python -c "import torch; print(torch.__version__)"
```

### Verifying GPU Support
```python
import torch

# Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

## Tensors - The Building Blocks

### Creating Tensors
```python
import torch
import numpy as np

# From Python lists
tensor1 = torch.tensor([1, 2, 3, 4])
tensor2 = torch.tensor([[1, 2], [3, 4]])

# From NumPy arrays
numpy_array = np.array([1, 2, 3, 4])
tensor3 = torch.from_numpy(numpy_array)

# Special tensors
zeros = torch.zeros(3, 4)           # 3x4 tensor of zeros
ones = torch.ones(2, 3)             # 2x3 tensor of ones
random_tensor = torch.randn(2, 3)   # 2x3 tensor with random values
identity = torch.eye(3)             # 3x3 identity matrix

# With specific data type and device
tensor_gpu = torch.tensor([1, 2, 3], dtype=torch.float32, device='cuda')
```

### Tensor Properties
```python
tensor = torch.randn(2, 3, 4)

print(f"Shape: {tensor.shape}")
print(f"Size: {tensor.size()}")
print(f"Number of dimensions: {tensor.ndim}")
print(f"Number of elements: {tensor.numel()}")
print(f"Data type: {tensor.dtype}")
print(f"Device: {tensor.device}")
print(f"Requires gradient: {tensor.requires_grad}")
```

### Tensor Operations
```python
# Basic arithmetic
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

print(a + b)    # [5, 7, 9]
print(a - b)    # [-3, -3, -3]
print(a * b)    # [4, 10, 18]
print(a / b)    # [0.25, 0.4, 0.5]

# Matrix operations
A = torch.randn(3, 4)
B = torch.randn(4, 5)

# Matrix multiplication
C = torch.matmul(A, B)  # or A @ B
print(f"Result shape: {C.shape}")  # [3, 5]

# Element-wise operations
x = torch.randn(2, 3)
y = torch.randn(2, 3)

print(torch.exp(x))      # Exponential
print(torch.log(x + 1))  # Natural logarithm
print(torch.sin(x))      # Sine
print(torch.sqrt(x + 1)) # Square root
```

## Autograd - Automatic Differentiation

### Understanding Gradients
```python
# Enable gradient tracking
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# Define a function
z = x**2 + y**2 + 2*x*y

# Compute gradients
z.backward()

print(f"dz/dx = {x.grad}")  # 2*x + 2*y = 2*2 + 2*3 = 10
print(f"dz/dy = {y.grad}")  # 2*y + 2*x = 2*3 + 2*2 = 10
```

### Gradient Computation Example
```python
# More complex example
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = torch.sum(x**2)  # Sum of squares

y.backward()
print(f"Gradients: {x.grad}")  # [2, 4, 6] (2*x for each element)
```

### Gradient Accumulation
```python
# Clear gradients before new computation
x = torch.tensor(2.0, requires_grad=True)

# First computation
y1 = x**2
y1.backward()
print(f"First gradient: {x.grad}")  # 4

# Second computation (gradients accumulate)
y2 = x**3
y2.backward()
print(f"Accumulated gradient: {x.grad}")  # 4 + 12 = 16

# Clear gradients
x.grad.zero_()
print(f"After clearing: {x.grad}")  # 0
```

## Building Your First Neural Network

### Simple Linear Model
```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define the model
class SimpleLinearModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)

# Create model instance
model = SimpleLinearModel(input_size=1, output_size=1)

# Create sample data
x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_train = torch.tensor([[2.0], [4.0], [6.0], [8.0]])  # y = 2x

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    # Forward pass
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

# Test the model
with torch.no_grad():
    test_input = torch.tensor([[5.0]])
    prediction = model(test_input)
    print(f'Prediction for x=5: {prediction.item():.2f}')
```

### Multi-Layer Perceptron
```python
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Create model
model = MLP(input_size=784, hidden_size=128, output_size=10)

# Print model architecture
print(model)
```

## Practice Exercises

### Exercise 1: Tensor Operations
Create a program that:
- Generates two random tensors of shape (3, 4)
- Performs element-wise multiplication
- Computes the mean and standard deviation
- Finds the maximum and minimum values

### Exercise 2: Gradient Computation
Implement a function that:
- Takes a tensor as input
- Computes the gradient of x³ + 2x² + 3x + 1
- Verifies the result manually
- Plots the function and its derivative

### Exercise 3: Simple Regression
Build a linear regression model that:
- Predicts house prices based on size
- Uses synthetic data (size vs price)
- Implements training loop with loss tracking
- Visualizes predictions vs actual values

### Exercise 4: Classification Model
Create a binary classifier that:
- Uses the sigmoid activation function
- Implements binary cross-entropy loss
- Trains on a simple 2D dataset
- Plots decision boundary

## Common PyTorch Patterns

### Model Training Template
```python
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        print(f'Epoch {epoch} completed. Average Loss: {running_loss/len(train_loader):.4f}')
```

### Model Evaluation Template
```python
def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    avg_loss = test_loss / len(test_loader)
    
    print(f'Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return accuracy, avg_loss
```

## AI Learning Prompt

**Copy this prompt into ChatGPT or any AI chatbot:**

"I'm learning PyTorch fundamentals and need help understanding:

1. What tensors are and how they differ from NumPy arrays
2. How to perform basic tensor operations and manipulations
3. How PyTorch's autograd system works for automatic differentiation
4. How to build and train simple neural networks
5. The difference between training and evaluation modes
6. Common PyTorch patterns and best practices

Please:
- Explain each concept with clear examples
- Show me how to implement basic neural networks
- Help me understand gradient computation and backpropagation
- Walk me through the training loop step by step
- Give me exercises to practice tensor operations
- Explain when to use different activation functions and loss functions

I want to build a solid foundation in PyTorch for deep learning. Please provide hands-on examples and help me understand the underlying concepts."

## Key Takeaways
- Tensors are the fundamental data structure in PyTorch
- Autograd enables automatic differentiation for gradient computation
- Neural networks are built by stacking layers and defining forward passes
- Training involves forward pass, loss computation, backward pass, and optimization
- PyTorch provides a flexible, Pythonic interface for deep learning
- Practice with simple examples before moving to complex architectures

## Next Steps
Master these fundamentals and you'll be ready for:
- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs)
- Advanced optimization techniques
- Transfer learning and pre-trained models
- Building production-ready models
