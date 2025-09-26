# Matrix Operations with NumPy

## 🎯 Learning Objectives

By the end of this tutorial, you will be able to:
- Understand the fundamentals of matrix operations
- Perform basic matrix arithmetic with NumPy
- Implement matrix multiplication, addition, and subtraction
- Apply matrix operations to solve real-world problems
- Build a simple neuron using matrix operations

## 📚 Prerequisites

- Basic Python knowledge (variables, functions, loops)
- Understanding of lists and arrays
- NumPy installed (`pip install numpy`)

## 🚀 Getting Started

### What are Matrix Operations?

Matrix operations are mathematical operations performed on matrices (2D arrays of numbers). They are fundamental to:
- **Machine Learning**: Neural networks use matrix multiplication extensively
- **Computer Graphics**: 3D transformations and rotations
- **Data Analysis**: Statistical computations and data manipulation
- **Scientific Computing**: Solving systems of equations

### Why NumPy?

NumPy (Numerical Python) is the foundation of scientific computing in Python. It provides:
- **Efficient array operations**: Much faster than Python lists
- **Broadcasting**: Automatic handling of different array shapes
- **Mathematical functions**: Built-in operations for arrays
- **Memory efficiency**: Optimized C implementations

## 📖 Core Concepts

### 1. Creating Matrices

```python
import numpy as np

# Create matrices from lists
matrix_a = np.array([[1, 2, 3],
                     [4, 5, 6]])

matrix_b = np.array([[7, 8],
                     [9, 10],
                     [11, 12]])

print("Matrix A shape:", matrix_a.shape)  # (2, 3)
print("Matrix B shape:", matrix_b.shape)  # (3, 2)
```

### 2. Basic Matrix Operations

#### Addition and Subtraction
```python
# Matrices must have the same shape
matrix_c = np.array([[1, 2],
                     [3, 4]])

matrix_d = np.array([[5, 6],
                     [7, 8]])

# Addition
result_add = matrix_c + matrix_d
print("Addition result:")
print(result_add)
# Output: [[6, 8], [10, 12]]

# Subtraction
result_sub = matrix_c - matrix_d
print("Subtraction result:")
print(result_sub)
# Output: [[-4, -4], [-4, -4]]
```

#### Scalar Operations
```python
# Multiply matrix by scalar
scalar = 3
result_scalar = matrix_c * scalar
print("Scalar multiplication:")
print(result_scalar)
# Output: [[3, 6], [9, 12]]
```

### 3. Matrix Multiplication

Matrix multiplication is the most important operation for neural networks.

#### Element-wise Multiplication (Hadamard Product)
```python
# Element-wise multiplication (same shape required)
result_elementwise = matrix_c * matrix_d
print("Element-wise multiplication:")
print(result_elementwise)
# Output: [[5, 12], [21, 32]]
```

#### True Matrix Multiplication (Dot Product)
```python
# True matrix multiplication using @ or np.dot()
result_dot = matrix_c @ matrix_d
# Alternative: result_dot = np.dot(matrix_c, matrix_d)
print("Matrix multiplication:")
print(result_dot)
# Output: [[19, 22], [43, 50]]

# Let's verify the calculation:
# [1*5 + 2*7, 1*6 + 2*8] = [19, 22]
# [3*5 + 4*7, 3*6 + 4*8] = [43, 50]
```

### 4. Matrix Properties

```python
# Transpose
matrix_transpose = matrix_c.T
print("Transpose:")
print(matrix_transpose)
# Output: [[1, 3], [2, 4]]

# Identity matrix
identity = np.eye(3)  # 3x3 identity matrix
print("Identity matrix:")
print(identity)

# Matrix determinant (for square matrices)
det = np.linalg.det(matrix_c)
print(f"Determinant: {det}")

# Matrix inverse (for square matrices)
inverse = np.linalg.inv(matrix_c)
print("Inverse:")
print(inverse)
```

## 🛠️ Hands-on Examples

### Example 1: Building a Simple Neuron

Let's implement a single neuron using matrix operations:

```python
def simple_neuron(inputs, weights, bias):
    """
    Simple neuron: output = inputs @ weights + bias
    
    Args:
        inputs: Input vector (1D array)
        weights: Weight vector (1D array)
        bias: Bias scalar
    
    Returns:
        output: Neuron output
    """
    # Matrix multiplication: inputs @ weights
    weighted_sum = np.dot(inputs, weights)
    
    # Add bias
    output = weighted_sum + bias
    
    return output

# Example usage
inputs = np.array([0.5, 0.3, 0.8])  # 3 input features
weights = np.array([0.2, 0.4, 0.1])  # 3 weights
bias = 0.1

neuron_output = simple_neuron(inputs, weights, bias)
print(f"Neuron output: {neuron_output}")
```

### Example 2: Batch Processing

Process multiple inputs at once:

```python
def batch_neuron(inputs_batch, weights, bias):
    """
    Process multiple inputs at once
    
    Args:
        inputs_batch: 2D array (batch_size, input_features)
        weights: 1D array (input_features,)
        bias: scalar
    
    Returns:
        outputs: 1D array (batch_size,)
    """
    # inputs_batch @ weights automatically handles the batch dimension
    weighted_sums = inputs_batch @ weights
    outputs = weighted_sums + bias
    return outputs

# Example usage
batch_inputs = np.array([[0.5, 0.3, 0.8],
                        [0.2, 0.9, 0.1],
                        [0.7, 0.4, 0.6]])

weights = np.array([0.2, 0.4, 0.1])
bias = 0.1

batch_outputs = batch_neuron(batch_inputs, weights, bias)
print(f"Batch outputs: {batch_outputs}")
```

### Example 3: Two-Layer Network

```python
def two_layer_network(inputs, weights1, bias1, weights2, bias2):
    """
    Simple two-layer neural network
    
    Args:
        inputs: Input vector
        weights1: First layer weights (input_size, hidden_size)
        bias1: First layer bias (hidden_size,)
        weights2: Second layer weights (hidden_size, output_size)
        bias2: Second layer bias (output_size,)
    
    Returns:
        output: Final network output
    """
    # First layer
    hidden = inputs @ weights1 + bias1
    hidden = np.maximum(0, hidden)  # ReLU activation
    
    # Second layer
    output = hidden @ weights2 + bias2
    
    return output

# Example usage
inputs = np.array([0.5, 0.3, 0.8])

# First layer: 3 inputs -> 4 hidden units
weights1 = np.random.randn(3, 4) * 0.1
bias1 = np.zeros(4)

# Second layer: 4 hidden -> 2 outputs
weights2 = np.random.randn(4, 2) * 0.1
bias2 = np.zeros(2)

output = two_layer_network(inputs, weights1, bias1, weights2, bias2)
print(f"Network output: {output}")
```

## 🎯 Practice Exercises

### Exercise 1: Matrix Operations Practice
```python
# Create these matrices and perform the operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 1. Calculate A + B
# 2. Calculate A - B
# 3. Calculate A * B (element-wise)
# 4. Calculate A @ B (matrix multiplication)
# 5. Calculate A.T (transpose)
```

### Exercise 2: Linear Transformation
```python
# Create a 2D point and apply a rotation matrix
point = np.array([1, 0])  # Point on x-axis

# Rotation matrix for 90 degrees
angle = np.pi / 2
rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                           [np.sin(angle), np.cos(angle)]])

# Apply rotation
rotated_point = point @ rotation_matrix
print(f"Original point: {point}")
print(f"Rotated point: {rotated_point}")
```

### Exercise 3: System of Equations
```python
# Solve: 2x + 3y = 7
#        4x + 5y = 11

# Coefficient matrix
A = np.array([[2, 3], [4, 5]])
# Right-hand side
b = np.array([7, 11])

# Solution: x = A^(-1) @ b
solution = np.linalg.inv(A) @ b
print(f"Solution: x = {solution[0]}, y = {solution[1]}")
```

## 🔍 Common Mistakes and Debugging

### Mistake 1: Shape Mismatch
```python
# This will cause an error
A = np.array([[1, 2], [3, 4]])  # Shape: (2, 2)
B = np.array([[1, 2, 3]])       # Shape: (1, 3)

# Error: operands could not be broadcast together
# result = A + B  # This will fail
```

**Solution**: Check shapes before operations
```python
print(f"A shape: {A.shape}")
print(f"B shape: {B.shape}")
```

### Mistake 2: Confusing Element-wise vs Matrix Multiplication
```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Element-wise multiplication
element_wise = A * B
print("Element-wise:", element_wise)

# Matrix multiplication
matrix_mult = A @ B
print("Matrix multiplication:", matrix_mult)
```

### Mistake 3: Forgetting to Add Bias
```python
# Common mistake in neural networks
inputs = np.array([0.5, 0.3, 0.8])
weights = np.array([0.2, 0.4, 0.1])

# Missing bias
output = inputs @ weights  # No bias added
print(f"Output without bias: {output}")

# Correct way
bias = 0.1
output = inputs @ weights + bias
print(f"Output with bias: {output}")
```

## 🧠 AI Learning Prompts

Use these prompts with ChatGPT or other AI assistants:

### Prompt 1: Understanding Matrix Operations
```
"I'm learning matrix operations with NumPy. Can you explain the difference between element-wise multiplication and matrix multiplication? Give me examples with 2x2 matrices and show the step-by-step calculations."
```

### Prompt 2: Debugging Shape Errors
```
"I'm getting a shape mismatch error when trying to multiply matrices. My matrices are A with shape (3, 4) and B with shape (2, 3). Can you help me understand what's wrong and how to fix it?"
```

### Prompt 3: Neural Network Implementation
```
"I want to implement a simple neural network layer using matrix operations. Can you help me write a function that takes inputs, weights, and bias, and returns the output? Include examples and explain each step."
```

## 📊 Key Takeaways

1. **Matrix operations are fundamental** to machine learning and scientific computing
2. **NumPy provides efficient implementations** of matrix operations
3. **Shape compatibility** is crucial for matrix operations
4. **Matrix multiplication (@)** is different from element-wise multiplication (*)
5. **Neural networks** are built using matrix operations
6. **Batch processing** allows efficient computation of multiple inputs

## 🚀 Next Steps

After mastering matrix operations, you're ready for:
- **Broadcasting**: How NumPy handles different array shapes
- **Reshaping**: Changing array dimensions for different operations
- **Advanced NumPy**: More complex operations and optimizations

## 📚 Additional Resources

### Books
- "Python for Data Analysis" by Wes McKinney
- "Numerical Python" by Robert Johansson
- "Deep Learning" by Ian Goodfellow (Chapter 2)

### Online Resources
- [NumPy Documentation](https://numpy.org/doc/stable/)
- [NumPy Tutorial](https://numpy.org/doc/stable/user/quickstart.html)
- [Matrix Operations in NumPy](https://numpy.org/doc/stable/reference/routines.linalg.html)

### Practice Platforms
- [NumPy Exercises](https://www.w3resource.com/python-exercises/numpy/)
- [Kaggle Learn](https://www.kaggle.com/learn/intro-to-programming)
- [Google Colab](https://colab.research.google.com/) for hands-on practice

---

**Ready for the next challenge?** Move on to [Broadcasting](02_broadcasting.md) to learn how NumPy handles different array shapes automatically!

Happy coding! 🐍✨
