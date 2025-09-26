# Reshaping Arrays in NumPy

## 🎯 Learning Objectives

By the end of this tutorial, you will be able to:
- Understand different methods for reshaping NumPy arrays
- Use reshape, transpose, and other reshaping operations effectively
- Apply reshaping to solve real-world problems in machine learning
- Implement data preprocessing pipelines using reshaping
- Debug common reshaping errors and understand memory layout

## 📚 Prerequisites

- Basic NumPy knowledge (arrays, shapes, basic operations)
- Understanding of matrix operations and broadcasting
- Familiarity with array dimensions and indexing

## 🚀 Getting Started

### What is Reshaping?

Reshaping is the process of changing the dimensions of an array without changing its data. It's like rearranging the same elements into different shapes - like taking a 1D line of numbers and arranging them into a 2D grid or 3D cube.

### Why is Reshaping Important?

- **Data preprocessing**: Convert data between different formats
- **Neural networks**: Reshape inputs for different layer types
- **Image processing**: Convert between different image representations
- **Memory efficiency**: Optimize data layout for operations
- **API compatibility**: Match expected input shapes

## 📖 Core Concepts

### 1. Basic Reshaping with `reshape()`

```python
import numpy as np

# Create a 1D array
arr_1d = np.array([1, 2, 3, 4, 5, 6])
print(f"Original shape: {arr_1d.shape}")

# Reshape to 2D
arr_2d = arr_1d.reshape(2, 3)
print(f"Reshaped to 2D: {arr_2d.shape}")
print(arr_2d)
# Output: [[1, 2, 3], [4, 5, 6]]

# Reshape to 3D
arr_3d = arr_1d.reshape(2, 1, 3)
print(f"Reshaped to 3D: {arr_3d.shape}")
print(arr_3d)
# Output: [[[1, 2, 3]], [[4, 5, 6]]]
```

### 2. Using -1 for Automatic Dimension Calculation

```python
# Let NumPy calculate one dimension
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# Reshape to 2 rows, let NumPy calculate columns
arr_2d = arr.reshape(2, -1)
print(f"Shape: {arr_2d.shape}")  # (2, 4)
print(arr_2d)

# Reshape to 4 columns, let NumPy calculate rows
arr_2d = arr.reshape(-1, 4)
print(f"Shape: {arr_2d.shape}")  # (2, 4)
print(arr_2d)

# Flatten to 1D
arr_flat = arr.reshape(-1)
print(f"Flattened shape: {arr_flat.shape}")  # (8,)
```

### 3. Transpose Operations

```python
# Create a 2D array
arr = np.array([[1, 2, 3],
                [4, 5, 6]])
print("Original:")
print(arr)
print(f"Shape: {arr.shape}")

# Transpose using .T
arr_transposed = arr.T
print("Transposed:")
print(arr_transposed)
print(f"Shape: {arr_transposed.shape}")

# Transpose using transpose()
arr_transposed2 = np.transpose(arr)
print("Transposed (method 2):")
print(arr_transposed2)
```

### 4. Multi-dimensional Transpose

```python
# Create a 3D array
arr_3d = np.array([[[1, 2, 3],
                    [4, 5, 6]],
                   [[7, 8, 9],
                    [10, 11, 12]]])
print(f"Original shape: {arr_3d.shape}")  # (2, 2, 3)

# Transpose different dimensions
arr_transposed = np.transpose(arr_3d, (1, 0, 2))
print(f"Transposed shape: {arr_transposed.shape}")  # (2, 2, 3)

# Transpose all dimensions
arr_transposed_all = np.transpose(arr_3d, (2, 1, 0))
print(f"All transposed shape: {arr_transposed_all.shape}")  # (3, 2, 2)
```

## 🛠️ Hands-on Examples

### Example 1: Image Data Reshaping

```python
def preprocess_image(image, target_size):
    """
    Reshape image for neural network input
    
    Args:
        image: Image array (height, width, channels)
        target_size: Target size (height, width)
    
    Returns:
        processed_image: Reshaped image
    """
    # Flatten image
    flattened = image.reshape(-1)
    
    # Reshape to target size (assuming same number of channels)
    channels = image.shape[2]
    processed = flattened.reshape(target_size[0], target_size[1], channels)
    
    return processed

# Example usage
image = np.random.randn(32, 32, 3)  # 32x32 RGB image
target_size = (28, 28)

processed_image = preprocess_image(image, target_size)
print(f"Original shape: {image.shape}")
print(f"Processed shape: {processed_image.shape}")
```

### Example 2: Batch Processing for Neural Networks

```python
def prepare_batch(data, batch_size):
    """
    Prepare data for batch processing
    
    Args:
        data: Input data (samples, features)
        batch_size: Number of samples per batch
    
    Returns:
        batches: List of batches
    """
    num_samples = data.shape[0]
    batches = []
    
    for i in range(0, num_samples, batch_size):
        batch = data[i:i+batch_size]
        batches.append(batch)
    
    return batches

# Example usage
data = np.random.randn(100, 10)  # 100 samples, 10 features
batch_size = 32

batches = prepare_batch(data, batch_size)
print(f"Number of batches: {len(batches)}")
print(f"First batch shape: {batches[0].shape}")
print(f"Last batch shape: {batches[-1].shape}")
```

### Example 3: Convolutional Neural Network Data Preparation

```python
def prepare_cnn_data(images, labels):
    """
    Prepare data for CNN input
    
    Args:
        images: Image data (samples, height, width, channels)
        labels: Labels (samples,)
    
    Returns:
        X: Reshaped images for CNN
        y: One-hot encoded labels
    """
    # Reshape images for CNN (add batch dimension if needed)
    X = images.reshape(-1, images.shape[1], images.shape[2], images.shape[3])
    
    # One-hot encode labels
    num_classes = len(np.unique(labels))
    y = np.eye(num_classes)[labels]
    
    return X, y

# Example usage
images = np.random.randn(1000, 28, 28, 1)  # 1000 grayscale images
labels = np.random.randint(0, 10, 1000)    # 10 classes

X, y = prepare_cnn_data(images, labels)
print(f"Images shape: {X.shape}")
print(f"Labels shape: {y.shape}")
```

### Example 4: Sequence Data for RNNs

```python
def prepare_sequence_data(data, sequence_length):
    """
    Prepare sequence data for RNN input
    
    Args:
        data: Time series data (time_steps, features)
        sequence_length: Length of each sequence
    
    Returns:
        sequences: Reshaped sequences (samples, sequence_length, features)
        targets: Target values for each sequence
    """
    num_sequences = len(data) - sequence_length
    sequences = []
    targets = []
    
    for i in range(num_sequences):
        sequence = data[i:i+sequence_length]
        target = data[i+sequence_length]
        
        sequences.append(sequence)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)

# Example usage
time_series = np.random.randn(100, 5)  # 100 time steps, 5 features
sequence_length = 10

sequences, targets = prepare_sequence_data(time_series, sequence_length)
print(f"Sequences shape: {sequences.shape}")
print(f"Targets shape: {targets.shape}")
```

## 🎯 Practice Exercises

### Exercise 1: Basic Reshaping
```python
# Practice these reshaping operations
arr = np.arange(24)  # [0, 1, 2, ..., 23]

# 1. Reshape to (4, 6)
# 2. Reshape to (6, 4)
# 3. Reshape to (2, 3, 4)
# 4. Reshape to (3, 2, 4)
# 5. Flatten back to 1D
```

### Exercise 2: Image Processing
```python
# Reshape image data for different purposes
image = np.random.randn(64, 64, 3)  # 64x64 RGB image

# 1. Flatten the image to 1D
# 2. Reshape to (3, 64, 64) - channels first
# 3. Reshape to (1, 64, 64, 3) - add batch dimension
# 4. Reshape to (3, 4096) - each channel as a row
```

### Exercise 3: Neural Network Data Preparation
```python
# Prepare data for different neural network types
data = np.random.randn(1000, 784)  # 1000 samples, 784 features (28x28 flattened)

# 1. Reshape for CNN: (1000, 28, 28, 1)
# 2. Reshape for RNN: (1000, 28, 28) - treat as sequences
# 3. Reshape for Transformer: (1000, 28, 28) - treat as tokens
# 4. Keep for Dense layer: (1000, 784)
```

## 🔍 Common Mistakes and Debugging

### Mistake 1: Incompatible Reshape Dimensions
```python
# This will cause an error
arr = np.array([1, 2, 3, 4, 5, 6])

# Error: cannot reshape array of size 6 into shape (2, 4)
# arr_2d = arr.reshape(2, 4)  # This will fail
```

**Solution**: Check total elements match
```python
print(f"Original size: {arr.size}")
print(f"Target size: {2 * 4}")

# Correct reshaping
arr_2d = arr.reshape(2, 3)  # 2 * 3 = 6 ✓
print("Correct reshape:", arr_2d)
```

### Mistake 2: Confusing Reshape with Transpose
```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

# Reshape changes the layout
reshaped = arr.reshape(3, 2)
print("Reshaped:")
print(reshaped)

# Transpose changes the orientation
transposed = arr.T
print("Transposed:")
print(transposed)
```

### Mistake 3: Memory Layout Issues
```python
# Reshape creates a view (no copy)
arr = np.array([[1, 2, 3],
                [4, 5, 6]])
reshaped = arr.reshape(6)

# Modifying the view affects the original
reshaped[0] = 999
print("Original after modifying view:")
print(arr)  # Original is changed!

# To avoid this, use copy()
arr_copy = arr.copy()
reshaped_copy = arr_copy.reshape(6)
reshaped_copy[0] = 888
print("Original after modifying copy:")
print(arr)  # Original is unchanged
```

## 🧠 AI Learning Prompts

Use these prompts with ChatGPT or other AI assistants:

### Prompt 1: Understanding Reshaping
```
"I'm learning NumPy reshaping. Can you explain the difference between reshape() and transpose()? Give me examples showing how they change the data layout and when to use each one."
```

### Prompt 2: Debugging Reshape Errors
```
"I'm getting a reshape error: 'cannot reshape array of size 12 into shape (3, 5)'. Can you help me understand why this fails and show me how to calculate compatible shapes?"
```

### Prompt 3: Neural Network Data Preparation
```
"I'm preparing data for a neural network. I have images with shape (1000, 28, 28, 1) and need to feed them to different layer types. Can you show me how to reshape this data for CNN, RNN, and Dense layers?"
```

## 📊 Key Takeaways

1. **Reshaping changes dimensions** without changing data
2. **Total elements must match** - can't reshape 6 elements into 8
3. **Reshape creates views** - be careful with memory layout
4. **Transpose changes orientation** - different from reshape
5. **Essential for neural networks** - different layers need different shapes
6. **Use -1 for automatic calculation** - let NumPy figure out one dimension

## 🚀 Next Steps

After mastering reshaping, you're ready for:
- **Advanced NumPy**: More complex operations and optimizations
- **Neural Network Implementation**: Building complete models
- **Data Preprocessing**: Real-world data preparation pipelines

## 📚 Additional Resources

### Books
- "Python for Data Analysis" by Wes McKinney
- "Numerical Python" by Robert Johansson
- "Deep Learning" by Ian Goodfellow

### Online Resources
- [NumPy Reshaping Documentation](https://numpy.org/doc/stable/user/basics.rec.html)
- [NumPy Array Manipulation](https://numpy.org/doc/stable/reference/routines.array-manipulation.html)
- [Reshaping Tutorial](https://numpy.org/doc/stable/user/quickstart.html#shape-manipulation)

### Practice Platforms
- [NumPy Reshaping Exercises](https://www.w3resource.com/python-exercises/numpy/)
- [Kaggle Learn](https://www.kaggle.com/learn/intro-to-programming)
- [Google Colab](https://colab.research.google.com/)

---

**Congratulations!** You've completed the NumPy Matrix Operations series! You now have a solid foundation in:
- Matrix operations and linear algebra
- Broadcasting for efficient computations
- Reshaping for data preparation

**Ready for the next challenge?** Move on to building neural networks from scratch or dive into PyTorch for deep learning!

Happy coding! 🐍✨
