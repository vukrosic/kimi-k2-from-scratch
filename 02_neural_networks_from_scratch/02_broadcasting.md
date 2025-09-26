# Broadcasting in NumPy

## 🎯 Learning Objectives

By the end of this tutorial, you will be able to:
- Understand what broadcasting is and why it's powerful
- Apply broadcasting rules to perform operations on arrays of different shapes
- Use broadcasting to write efficient, readable code
- Implement neural network operations using broadcasting
- Debug common broadcasting errors

## 📚 Prerequisites

- Basic NumPy knowledge (arrays, shapes, basic operations)
- Understanding of matrix operations
- Familiarity with array shapes and dimensions

## 🚀 Getting Started

### What is Broadcasting?

Broadcasting is NumPy's ability to perform operations on arrays of different shapes without explicitly reshaping them. It's like having NumPy automatically "stretch" smaller arrays to match larger ones for operations.

### Why is Broadcasting Important?

- **Efficiency**: No need to create large arrays just for operations
- **Memory savings**: Avoids unnecessary data duplication
- **Cleaner code**: Write more readable and concise operations
- **Neural networks**: Essential for batch processing and layer operations

## 📖 Core Concepts

### 1. Broadcasting Rules

NumPy follows these rules when broadcasting:

1. **Rule 1**: Arrays with fewer dimensions are padded with size 1 dimensions on the left
2. **Rule 2**: Arrays with size 1 in any dimension can be "stretched" to match other arrays
3. **Rule 3**: Arrays must be compatible in all dimensions

### 2. Basic Broadcasting Examples

```python
import numpy as np

# Example 1: Scalar broadcasting
array_2d = np.array([[1, 2, 3],
                     [4, 5, 6]])
scalar = 10

# The scalar is broadcast to match the array shape
result = array_2d + scalar
print("Scalar broadcasting:")
print(result)
# Output: [[11, 12, 13], [14, 15, 16]]
```

```python
# Example 2: 1D array with 2D array
array_2d = np.array([[1, 2, 3],
                     [4, 5, 6]])
array_1d = np.array([10, 20, 30])

# The 1D array is broadcast to match each row
result = array_2d + array_1d
print("1D with 2D broadcasting:")
print(result)
# Output: [[11, 22, 33], [14, 25, 36]]
```

### 3. Understanding Broadcasting Steps

Let's trace through a broadcasting operation step by step:

```python
# Step-by-step broadcasting example
A = np.array([[1, 2, 3],
              [4, 5, 6]])  # Shape: (2, 3)

B = np.array([10, 20, 30])  # Shape: (3,)

# Step 1: Pad B with size 1 dimensions
# B becomes shape (1, 3)
B_padded = B.reshape(1, 3)

# Step 2: Stretch B to match A's shape
# B becomes shape (2, 3)
B_stretched = np.tile(B_padded, (2, 1))

# Step 3: Perform the operation
result = A + B_stretched
print("Step-by-step result:")
print(result)
```

### 4. Common Broadcasting Patterns

#### Pattern 1: Adding Bias to Neural Network Layers
```python
# Neural network layer output (batch_size, features)
layer_output = np.array([[1, 2, 3],
                        [4, 5, 6]])  # Shape: (2, 3)

# Bias vector (features,)
bias = np.array([0.1, 0.2, 0.3])  # Shape: (3,)

# Broadcasting automatically handles this
result = layer_output + bias
print("Neural network bias addition:")
print(result)
# Output: [[1.1, 2.2, 3.3], [4.1, 5.2, 6.3]]
```

#### Pattern 2: Scaling Features
```python
# Feature matrix (samples, features)
features = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])  # Shape: (3, 3)

# Scaling factors (features,)
scales = np.array([2, 3, 4])  # Shape: (3,)

# Broadcast scaling
scaled_features = features * scales
print("Feature scaling:")
print(scaled_features)
# Output: [[2, 6, 12], [8, 15, 24], [14, 24, 36]]
```

#### Pattern 3: Batch Processing
```python
# Input batch (batch_size, input_features)
inputs = np.array([[1, 2],
                   [3, 4],
                   [5, 6]])  # Shape: (3, 2)

# Weights (input_features, output_features)
weights = np.array([[0.1, 0.2],
                    [0.3, 0.4]])  # Shape: (2, 2)

# Matrix multiplication with broadcasting
outputs = inputs @ weights
print("Batch processing:")
print(outputs)
# Output: [[0.7, 1.0], [1.5, 2.2], [2.3, 3.4]]
```

## 🛠️ Hands-on Examples

### Example 1: Image Processing with Broadcasting

```python
def normalize_image(image, mean, std):
    """
    Normalize image using broadcasting
    
    Args:
        image: Image array (height, width, channels)
        mean: Mean values for each channel (channels,)
        std: Standard deviation for each channel (channels,)
    
    Returns:
        normalized_image: Normalized image
    """
    # Broadcasting: mean and std are broadcast across height and width
    normalized = (image - mean) / std
    return normalized

# Example usage
image = np.random.randn(32, 32, 3)  # 32x32 RGB image
mean = np.array([0.485, 0.456, 0.406])  # ImageNet mean
std = np.array([0.229, 0.224, 0.225])   # ImageNet std

normalized_image = normalize_image(image, mean, std)
print(f"Original image shape: {image.shape}")
print(f"Normalized image shape: {normalized_image.shape}")
```

### Example 2: Neural Network Layer with Broadcasting

```python
def dense_layer(inputs, weights, bias):
    """
    Dense layer implementation using broadcasting
    
    Args:
        inputs: Input batch (batch_size, input_features)
        weights: Weight matrix (input_features, output_features)
        bias: Bias vector (output_features,)
    
    Returns:
        outputs: Layer outputs (batch_size, output_features)
    """
    # Matrix multiplication
    outputs = inputs @ weights
    
    # Broadcasting: bias is added to each sample
    outputs = outputs + bias
    
    return outputs

# Example usage
batch_size = 4
input_features = 3
output_features = 2

inputs = np.random.randn(batch_size, input_features)
weights = np.random.randn(input_features, output_features)
bias = np.random.randn(output_features)

outputs = dense_layer(inputs, weights, bias)
print(f"Inputs shape: {inputs.shape}")
print(f"Weights shape: {weights.shape}")
print(f"Bias shape: {bias.shape}")
print(f"Outputs shape: {outputs.shape}")
```

### Example 3: Attention Mechanism with Broadcasting

```python
def scaled_dot_product_attention(Q, K, V):
    """
    Scaled dot-product attention using broadcasting
    
    Args:
        Q: Query matrix (batch_size, seq_len, d_model)
        K: Key matrix (batch_size, seq_len, d_model)
        V: Value matrix (batch_size, seq_len, d_model)
    
    Returns:
        output: Attention output
    """
    d_model = Q.shape[-1]
    
    # Compute attention scores
    scores = Q @ K.transpose(-2, -1)  # (batch_size, seq_len, seq_len)
    scores = scores / np.sqrt(d_model)
    
    # Apply softmax (broadcasting across the last dimension)
    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    
    # Apply attention to values
    output = attention_weights @ V
    
    return output

# Example usage
batch_size = 2
seq_len = 4
d_model = 3

Q = np.random.randn(batch_size, seq_len, d_model)
K = np.random.randn(batch_size, seq_len, d_model)
V = np.random.randn(batch_size, seq_len, d_model)

attention_output = scaled_dot_product_attention(Q, K, V)
print(f"Attention output shape: {attention_output.shape}")
```

## 🎯 Practice Exercises

### Exercise 1: Basic Broadcasting
```python
# Practice these broadcasting operations
A = np.array([[1, 2, 3],
              [4, 5, 6]])  # Shape: (2, 3)

B = np.array([10, 20, 30])  # Shape: (3,)
C = np.array([[1], [2]])    # Shape: (2, 1)

# 1. A + B
# 2. A + C
# 3. A * B
# 4. A * C
```

### Exercise 2: Neural Network Operations
```python
# Implement a simple neural network layer
def relu_layer(inputs, weights, bias):
    """
    ReLU layer with broadcasting
    
    Args:
        inputs: (batch_size, input_features)
        weights: (input_features, output_features)
        bias: (output_features,)
    
    Returns:
        outputs: (batch_size, output_features)
    """
    # Your code here
    pass

# Test your implementation
inputs = np.array([[1, 2, 3],
                   [4, 5, 6]])
weights = np.array([[0.1, 0.2],
                    [0.3, 0.4],
                    [0.5, 0.6]])
bias = np.array([0.1, 0.2])

# Call your function and print the result
```

### Exercise 3: Data Preprocessing
```python
# Normalize a dataset using broadcasting
def normalize_dataset(data, mean, std):
    """
    Normalize dataset features
    
    Args:
        data: (samples, features)
        mean: (features,)
        std: (features,)
    
    Returns:
        normalized_data: (samples, features)
    """
    # Your code here
    pass

# Test with sample data
data = np.random.randn(100, 5)  # 100 samples, 5 features
mean = np.mean(data, axis=0)    # Mean of each feature
std = np.std(data, axis=0)      # Std of each feature

# Normalize and verify
```

## 🔍 Common Mistakes and Debugging

### Mistake 1: Incompatible Shapes
```python
# This will cause a broadcasting error
A = np.array([[1, 2, 3],
              [4, 5, 6]])  # Shape: (2, 3)
B = np.array([1, 2])       # Shape: (2,)

# Error: operands could not be broadcast together
# result = A + B  # This will fail
```

**Solution**: Check shapes and understand broadcasting rules
```python
print(f"A shape: {A.shape}")
print(f"B shape: {B.shape}")

# B needs to be shape (3,) to broadcast with A
B_correct = np.array([1, 2, 3])  # Shape: (3,)
result = A + B_correct
print("Correct result:", result)
```

### Mistake 2: Confusing Broadcasting with Reshaping
```python
# Broadcasting doesn't change the original arrays
A = np.array([[1, 2, 3],
              [4, 5, 6]])
B = np.array([10, 20, 30])

result = A + B
print("A after broadcasting:", A)  # A is unchanged
print("B after broadcasting:", B)  # B is unchanged
print("Result:", result)
```

### Mistake 3: Performance Issues with Large Arrays
```python
# Inefficient: Creating large arrays
large_array = np.ones((1000, 1000))
small_array = np.array([1, 2, 3])

# This creates a large temporary array
result = large_array + small_array  # Broadcasting is efficient

# But this would be inefficient:
# large_small = np.tile(small_array, (1000, 1000))  # Don't do this
# result = large_array + large_small
```

## 🧠 AI Learning Prompts

Use these prompts with ChatGPT or other AI assistants:

### Prompt 1: Understanding Broadcasting Rules
```
"I'm learning NumPy broadcasting. Can you explain the three broadcasting rules with examples? Show me how NumPy determines if two arrays can be broadcast together and what the resulting shape will be."
```

### Prompt 2: Debugging Broadcasting Errors
```
"I'm getting a broadcasting error: 'operands could not be broadcast together'. My arrays have shapes (3, 4) and (2,). Can you help me understand why this fails and how to fix it?"
```

### Prompt 3: Neural Network Broadcasting
```
"I'm implementing a neural network layer and need to add bias to the outputs. The outputs have shape (batch_size, features) and bias has shape (features,). Can you show me how broadcasting works here and write a function that does this correctly?"
```

## 📊 Key Takeaways

1. **Broadcasting is automatic** - NumPy handles shape compatibility
2. **Memory efficient** - No need to create large temporary arrays
3. **Essential for neural networks** - Batch processing and layer operations
4. **Follows strict rules** - Arrays must be compatible in all dimensions
5. **Performance benefit** - Faster than explicit reshaping
6. **Common in data science** - Feature scaling, normalization, etc.

## 🚀 Next Steps

After mastering broadcasting, you're ready for:
- **Reshaping**: Changing array dimensions for different operations
- **Advanced NumPy**: More complex operations and optimizations
- **Neural Network Implementation**: Building complete models

## 📚 Additional Resources

### Books
- "Python for Data Analysis" by Wes McKinney
- "Numerical Python" by Robert Johansson
- "Deep Learning" by Ian Goodfellow

### Online Resources
- [NumPy Broadcasting Documentation](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [NumPy Broadcasting Tutorial](https://numpy.org/doc/stable/user/theory.broadcasting.html)
- [Broadcasting Examples](https://numpy.org/doc/stable/user/basics.broadcasting.html#broadcasting-examples)

### Practice Platforms
- [NumPy Broadcasting Exercises](https://www.w3resource.com/python-exercises/numpy/)
- [Kaggle Learn](https://www.kaggle.com/learn/intro-to-programming)
- [Google Colab](https://colab.research.google.com/)

---

**Ready for the next challenge?** Move on to [Reshaping](03_reshaping.md) to learn how to change array dimensions for different operations!

Happy coding! 🐍✨
