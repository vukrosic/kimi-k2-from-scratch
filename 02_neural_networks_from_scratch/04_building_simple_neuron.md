# Building a Simple Neuron with NumPy

## 🎯 Learning Objectives

By the end of this tutorial, you will be able to:
- Understand the mathematical foundation of artificial neurons
- Implement a single neuron from scratch using only NumPy
- Build and train a perceptron for binary classification
- Implement multiple neurons and simple neural networks
- Understand activation functions and their role in neural networks
- Apply neurons to solve real-world problems like XOR and classification
- Debug common neuron implementation issues

## 📚 Prerequisites

- Matrix operations with NumPy
- Understanding of broadcasting and reshaping
- Basic knowledge of linear algebra
- Familiarity with Python functions and classes

## 🚀 Getting Started

### What is a Neuron?

An artificial neuron is a mathematical model inspired by biological neurons. It takes multiple inputs, applies weights, adds a bias, and passes the result through an activation function to produce an output.

### The Mathematical Model

A neuron can be represented mathematically as:

```
output = activation_function(Σ(input_i × weight_i) + bias)
```

Or in vector form:
```
output = activation_function(inputs @ weights + bias)
```

### Why Build Neurons from Scratch?

- **Deep Understanding**: Grasp the fundamental concepts
- **No Black Box**: Know exactly what happens inside
- **Foundation**: Build more complex networks later
- **Debugging**: Understand and fix issues
- **Customization**: Modify and experiment freely

## 📖 Core Concepts

### 1. The Basic Neuron Structure

```python
import numpy as np
import matplotlib.pyplot as plt

class SimpleNeuron:
    def __init__(self, num_inputs, activation='sigmoid'):
        """
        Initialize a simple neuron
        
        Args:
            num_inputs: Number of input features
            activation: Activation function ('sigmoid', 'relu', 'tanh', 'linear')
        """
        # Initialize weights randomly (Xavier initialization)
        self.weights = np.random.randn(num_inputs) * np.sqrt(2.0 / num_inputs)
        self.bias = 0.0
        self.activation = activation
        
    def forward(self, inputs):
        """
        Forward pass through the neuron
        
        Args:
            inputs: Input vector (1D array)
            
        Returns:
            output: Neuron output
        """
        # Linear combination: inputs @ weights + bias
        z = np.dot(inputs, self.weights) + self.bias
        
        # Apply activation function
        if self.activation == 'sigmoid':
            output = self._sigmoid(z)
        elif self.activation == 'relu':
            output = self._relu(z)
        elif self.activation == 'tanh':
            output = self._tanh(z)
        elif self.activation == 'linear':
            output = z
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
            
        return output
    
    def _sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow
    
    def _relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def _tanh(self, x):
        """Tanh activation function"""
        return np.tanh(x)
    
    def get_parameters(self):
        """Get neuron parameters"""
        return {'weights': self.weights, 'bias': self.bias}
    
    def set_parameters(self, weights, bias):
        """Set neuron parameters"""
        self.weights = weights
        self.bias = bias

# Example usage
neuron = SimpleNeuron(num_inputs=3, activation='sigmoid')
inputs = np.array([0.5, 0.3, 0.8])
output = neuron.forward(inputs)
print(f"Neuron output: {output}")
```

### 2. Understanding Activation Functions

```python
def plot_activation_functions():
    """Plot different activation functions"""
    x = np.linspace(-5, 5, 100)
    
    # Define activation functions
    sigmoid = 1 / (1 + np.exp(-x))
    relu = np.maximum(0, x)
    tanh = np.tanh(x)
    linear = x
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot each activation function
    axes[0, 0].plot(x, sigmoid, 'b-', linewidth=2)
    axes[0, 0].set_title('Sigmoid')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(x, relu, 'r-', linewidth=2)
    axes[0, 1].set_title('ReLU')
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(x, tanh, 'g-', linewidth=2)
    axes[1, 0].set_title('Tanh')
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(x, linear, 'm-', linewidth=2)
    axes[1, 1].set_title('Linear')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

# Uncomment to plot activation functions
# plot_activation_functions()
```

### 3. Batch Processing with Neurons

```python
class BatchNeuron:
    def __init__(self, num_inputs, activation='sigmoid'):
        """Initialize neuron for batch processing"""
        self.weights = np.random.randn(num_inputs) * np.sqrt(2.0 / num_inputs)
        self.bias = 0.0
        self.activation = activation
        
    def forward(self, inputs):
        """
        Forward pass for batch of inputs
        
        Args:
            inputs: Input matrix (batch_size, num_inputs)
            
        Returns:
            outputs: Output vector (batch_size,)
        """
        # Matrix multiplication: inputs @ weights + bias
        # Broadcasting handles the bias addition
        z = np.dot(inputs, self.weights) + self.bias
        
        # Apply activation function
        if self.activation == 'sigmoid':
            outputs = self._sigmoid(z)
        elif self.activation == 'relu':
            outputs = self._relu(z)
        elif self.activation == 'tanh':
            outputs = self._tanh(z)
        elif self.activation == 'linear':
            outputs = z
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
            
        return outputs
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _tanh(self, x):
        return np.tanh(x)

# Example usage
neuron = BatchNeuron(num_inputs=3, activation='sigmoid')
batch_inputs = np.array([[0.5, 0.3, 0.8],
                        [0.2, 0.9, 0.1],
                        [0.7, 0.4, 0.6]])
outputs = neuron.forward(batch_inputs)
print(f"Batch outputs: {outputs}")
```

## 🛠️ Hands-on Examples

### Example 1: Binary Classification with Perceptron

```python
class Perceptron:
    def __init__(self, num_inputs, learning_rate=0.1):
        """Initialize perceptron for binary classification"""
        self.weights = np.random.randn(num_inputs) * 0.1
        self.bias = 0.0
        self.learning_rate = learning_rate
        
    def forward(self, inputs):
        """Forward pass with step activation"""
        z = np.dot(inputs, self.weights) + self.bias
        return self._step_function(z)
    
    def _step_function(self, x):
        """Step activation function"""
        return np.where(x >= 0, 1, 0)
    
    def train(self, X, y, epochs=100):
        """
        Train the perceptron
        
        Args:
            X: Training inputs (samples, features)
            y: Training labels (samples,)
            epochs: Number of training epochs
        """
        errors = []
        
        for epoch in range(epochs):
            epoch_errors = 0
            
            for i in range(len(X)):
                # Forward pass
                prediction = self.forward(X[i])
                
                # Calculate error
                error = y[i] - prediction
                
                if error != 0:
                    epoch_errors += 1
                    
                    # Update weights and bias
                    self.weights += self.learning_rate * error * X[i]
                    self.bias += self.learning_rate * error
            
            errors.append(epoch_errors)
            
            # Stop if no errors
            if epoch_errors == 0:
                print(f"Converged after {epoch + 1} epochs")
                break
                
        return errors
    
    def predict(self, X):
        """Make predictions on new data"""
        return np.array([self.forward(x) for x in X])

# Example: AND gate
def train_and_gate():
    """Train perceptron to learn AND gate"""
    # AND gate truth table
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([0, 0, 0, 1])
    
    # Create and train perceptron
    perceptron = Perceptron(num_inputs=2, learning_rate=0.1)
    errors = perceptron.train(X, y, epochs=100)
    
    # Test the trained perceptron
    predictions = perceptron.predict(X)
    
    print("AND Gate Results:")
    print("Inputs | Target | Prediction")
    print("-" * 30)
    for i in range(len(X)):
        print(f"{X[i]} |   {y[i]}    |     {predictions[i]}")
    
    print(f"\nFinal weights: {perceptron.weights}")
    print(f"Final bias: {perceptron.bias}")
    
    return perceptron, errors

# Train AND gate
and_perceptron, and_errors = train_and_gate()
```

### Example 2: XOR Problem with Multiple Neurons

```python
class XORNetwork:
    def __init__(self):
        """Initialize network for XOR problem"""
        # Two hidden neurons
        self.hidden1 = SimpleNeuron(num_inputs=2, activation='sigmoid')
        self.hidden2 = SimpleNeuron(num_inputs=2, activation='sigmoid')
        
        # Output neuron
        self.output = SimpleNeuron(num_inputs=2, activation='sigmoid')
        
        # Manually set weights for XOR (learned through training)
        self._set_xor_weights()
    
    def _set_xor_weights(self):
        """Set weights for XOR problem"""
        # Hidden layer 1: OR gate
        self.hidden1.set_parameters(
            weights=np.array([1.0, 1.0]),
            bias=-0.5
        )
        
        # Hidden layer 2: NAND gate
        self.hidden2.set_parameters(
            weights=np.array([-1.0, -1.0]),
            bias=1.5
        )
        
        # Output layer: AND gate
        self.output.set_parameters(
            weights=np.array([1.0, 1.0]),
            bias=-1.5
        )
    
    def forward(self, inputs):
        """Forward pass through the network"""
        # Hidden layer
        h1_output = self.hidden1.forward(inputs)
        h2_output = self.hidden2.forward(inputs)
        
        # Output layer
        hidden_outputs = np.array([h1_output, h2_output])
        final_output = self.output.forward(hidden_outputs)
        
        return final_output
    
    def predict(self, X):
        """Make predictions on batch of inputs"""
        return np.array([self.forward(x) for x in X])

# Test XOR network
def test_xor_network():
    """Test the XOR network"""
    xor_net = XORNetwork()
    
    # XOR truth table
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    
    predictions = xor_net.predict(X)
    
    print("XOR Network Results:")
    print("Inputs | Prediction")
    print("-" * 20)
    for i in range(len(X)):
        print(f"{X[i]} |    {predictions[i]:.3f}")
    
    return xor_net

# Test XOR
xor_network = test_xor_network()
```

### Example 3: Multi-Class Classification

```python
class MultiClassNeuron:
    def __init__(self, num_inputs, num_classes, activation='sigmoid'):
        """Initialize neuron for multi-class classification"""
        self.num_classes = num_classes
        self.activation = activation
        
        # One set of weights for each class
        self.weights = np.random.randn(num_classes, num_inputs) * 0.1
        self.bias = np.zeros(num_classes)
        
    def forward(self, inputs):
        """Forward pass for multi-class classification"""
        # Calculate scores for each class
        scores = np.dot(self.weights, inputs) + self.bias
        
        # Apply activation function
        if self.activation == 'sigmoid':
            outputs = self._sigmoid(scores)
        elif self.activation == 'softmax':
            outputs = self._softmax(scores)
        else:
            outputs = scores
            
        return outputs
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _softmax(self, x):
        """Softmax activation for probability distribution"""
        exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return exp_x / np.sum(exp_x)
    
    def predict(self, inputs):
        """Predict class with highest score"""
        outputs = self.forward(inputs)
        return np.argmax(outputs)
    
    def predict_batch(self, X):
        """Predict classes for batch of inputs"""
        return np.array([self.predict(x) for x in X])

# Example: 3-class classification
def test_multiclass_neuron():
    """Test multi-class neuron"""
    # Create sample data
    X = np.array([[1, 2],
                  [2, 3],
                  [3, 1],
                  [4, 2]])
    y = np.array([0, 1, 2, 0])  # 3 classes
    
    # Create and test neuron
    neuron = MultiClassNeuron(num_inputs=2, num_classes=3, activation='softmax')
    
    print("Multi-class Neuron Results:")
    print("Inputs | Class 0 | Class 1 | Class 2 | Prediction")
    print("-" * 55)
    
    for i in range(len(X)):
        outputs = neuron.forward(X[i])
        prediction = neuron.predict(X[i])
        print(f"{X[i]} | {outputs[0]:.3f} | {outputs[1]:.3f} | {outputs[2]:.3f} |     {prediction}")
    
    return neuron

# Test multi-class neuron
multiclass_neuron = test_multiclass_neuron()
```

### Example 4: Neural Network with Multiple Layers

```python
class MultiLayerNetwork:
    def __init__(self, layer_sizes, activation='sigmoid'):
        """
        Initialize multi-layer neural network
        
        Args:
            layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
            activation: Activation function for all layers
        """
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.layers = []
        
        # Create layers
        for i in range(len(layer_sizes) - 1):
            layer = SimpleNeuron(
                num_inputs=layer_sizes[i],
                activation=activation
            )
            self.layers.append(layer)
    
    def forward(self, inputs):
        """Forward pass through all layers"""
        current_input = inputs
        
        for layer in self.layers:
            current_input = layer.forward(current_input)
            
        return current_input
    
    def predict(self, X):
        """Make predictions on batch of inputs"""
        return np.array([self.forward(x) for x in X])
    
    def get_all_outputs(self, inputs):
        """Get outputs from all layers (for analysis)"""
        outputs = [inputs]
        current_input = inputs
        
        for layer in self.layers:
            current_input = layer.forward(current_input)
            outputs.append(current_input)
            
        return outputs

# Example: 2-3-1 network
def test_multilayer_network():
    """Test multi-layer network"""
    # Create network: 2 inputs -> 3 hidden -> 1 output
    network = MultiLayerNetwork([2, 3, 1], activation='sigmoid')
    
    # Test inputs
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    
    print("Multi-layer Network Results:")
    print("Inputs | Output")
    print("-" * 20)
    
    for i in range(len(X)):
        output = network.forward(X[i])
        print(f"{X[i]} | {output:.3f}")
    
    # Get all layer outputs for first input
    all_outputs = network.get_all_outputs(X[0])
    print(f"\nLayer outputs for input {X[0]}:")
    for i, output in enumerate(all_outputs):
        print(f"Layer {i}: {output}")
    
    return network

# Test multi-layer network
multilayer_network = test_multilayer_network()
```

### Example 5: Real-World Application: Iris Classification

```python
def load_iris_data():
    """Load and prepare Iris dataset"""
    # Simplified Iris dataset (sepal length, sepal width, petal length, petal width)
    X = np.array([[5.1, 3.5, 1.4, 0.2],
                  [4.9, 3.0, 1.4, 0.2],
                  [4.7, 3.2, 1.3, 0.2],
                  [4.6, 3.1, 1.5, 0.2],
                  [5.0, 3.6, 1.4, 0.2],
                  [5.4, 3.9, 1.7, 0.4],
                  [4.6, 3.4, 1.4, 0.3],
                  [5.0, 3.4, 1.5, 0.2],
                  [4.4, 2.9, 1.4, 0.2],
                  [4.9, 3.1, 1.5, 0.1],
                  [7.0, 3.2, 4.7, 1.4],
                  [6.4, 3.2, 4.5, 1.5],
                  [6.9, 3.1, 4.9, 1.5],
                  [5.5, 2.3, 4.0, 1.3],
                  [6.5, 2.8, 4.6, 1.5],
                  [5.7, 2.8, 4.5, 1.3],
                  [6.3, 3.3, 4.7, 1.6],
                  [4.9, 2.4, 3.3, 1.0],
                  [6.6, 2.9, 4.6, 1.3],
                  [5.2, 2.7, 3.9, 1.4]])
    
    # Labels: 0 = setosa, 1 = versicolor
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    
    return X, y

def normalize_data(X):
    """Normalize data to [0, 1] range"""
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    return (X - X_min) / (X_max - X_min)

def test_iris_classification():
    """Test neuron on Iris dataset"""
    # Load and normalize data
    X, y = load_iris_data()
    X_normalized = normalize_data(X)
    
    # Create neuron for binary classification
    neuron = SimpleNeuron(num_inputs=4, activation='sigmoid')
    
    # Test on first few samples
    print("Iris Classification Results:")
    print("Sample | Features | Target | Output")
    print("-" * 45)
    
    for i in range(10):
        output = neuron.forward(X_normalized[i])
        print(f"  {i+1:2d}   | {X_normalized[i]} |   {y[i]}    | {output:.3f}")
    
    return neuron, X_normalized, y

# Test Iris classification
iris_neuron, X_norm, y_labels = test_iris_classification()
```

## 🎯 Practice Exercises

### Exercise 1: Implement Different Activation Functions
```python
# Implement these activation functions and their derivatives
def sigmoid(x):
    """Sigmoid activation function"""
    # Your code here
    pass

def sigmoid_derivative(x):
    """Derivative of sigmoid function"""
    # Your code here
    pass

def relu(x):
    """ReLU activation function"""
    # Your code here
    pass

def relu_derivative(x):
    """Derivative of ReLU function"""
    # Your code here
    pass

def tanh(x):
    """Tanh activation function"""
    # Your code here
    pass

def tanh_derivative(x):
    """Derivative of tanh function"""
    # Your code here
    pass
```

### Exercise 2: Build a Perceptron for OR Gate
```python
# Implement a perceptron that learns the OR gate
def train_or_gate():
    """Train perceptron to learn OR gate"""
    # Your code here
    pass

# Test your implementation
```

### Exercise 3: Create a 3-Input Neuron
```python
# Build a neuron that takes 3 inputs and implements a specific logic
def three_input_neuron():
    """Create neuron for 3-input logic"""
    # Your code here
    pass

# Test with different input combinations
```

### Exercise 4: Implement a Neural Network for XOR
```python
# Build a 2-layer network that solves XOR problem
def xor_network():
    """Implement XOR network"""
    # Your code here
    pass

# Test the network
```

## 🔍 Common Mistakes and Debugging

### Mistake 1: Weight Initialization
```python
# Bad: All weights initialized to 0
class BadNeuron:
    def __init__(self, num_inputs):
        self.weights = np.zeros(num_inputs)  # All weights are 0
        self.bias = 0.0

# Good: Random weight initialization
class GoodNeuron:
    def __init__(self, num_inputs):
        self.weights = np.random.randn(num_inputs) * 0.1  # Small random weights
        self.bias = 0.0
```

### Mistake 2: Activation Function Overflow
```python
# Bad: No clipping for sigmoid
def bad_sigmoid(x):
    return 1 / (1 + np.exp(-x))  # Can overflow for large x

# Good: Clipped sigmoid
def good_sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Prevents overflow
```

### Mistake 3: Shape Mismatch in Batch Processing
```python
# Bad: Inconsistent shapes
def bad_batch_forward(self, inputs):
    # inputs shape: (batch_size, features)
    # weights shape: (features,)
    # This might not work as expected
    return inputs * self.weights  # Element-wise multiplication

# Good: Proper matrix multiplication
def good_batch_forward(self, inputs):
    # inputs shape: (batch_size, features)
    # weights shape: (features,)
    # Result shape: (batch_size,)
    return np.dot(inputs, self.weights) + self.bias
```

## 🧠 AI Learning Prompts

Use these prompts with ChatGPT or other AI assistants:

### Prompt 1: Understanding Neuron Mathematics
```
"I'm learning about artificial neurons. Can you explain the mathematical model of a neuron step by step? Show me how inputs, weights, bias, and activation functions work together, and give me examples with specific numbers."
```

### Prompt 2: Implementing Activation Functions
```
"I need to implement different activation functions in NumPy. Can you help me write sigmoid, ReLU, and tanh functions, including their derivatives? Also explain when to use each one and what problems they solve."
```

### Prompt 3: Debugging Neuron Implementation
```
"I'm building a neuron from scratch but it's not working correctly. My neuron takes 2 inputs and should output 1 for positive inputs and 0 for negative inputs, but it's giving wrong results. Can you help me debug this?"
```

### Prompt 4: XOR Problem Solution
```
"I understand that a single neuron can't solve XOR, but I want to build a network that can. Can you help me design a 2-layer network with 2 hidden neurons that solves XOR? Show me the weights and explain why this works."
```

## 📊 Key Takeaways

1. **Neurons are mathematical models** that process inputs through weights, bias, and activation
2. **Activation functions** determine the neuron's output and behavior
3. **Weight initialization** is crucial for proper learning
4. **Single neurons** can solve linearly separable problems
5. **Multiple neurons** are needed for complex problems like XOR
6. **Batch processing** allows efficient computation of multiple inputs
7. **Proper implementation** requires understanding of NumPy operations

## 🚀 Next Steps

After mastering simple neurons, you're ready for:
- **Backpropagation**: How to train neurons and networks
- **Multi-layer Networks**: Building deeper architectures
- **Training Algorithms**: Gradient descent and optimization
- **Real-world Applications**: Image recognition, NLP, etc.

## 📚 Fun Additional Resources - not necessary for this course

### Books
- "Deep Learning" by Ian Goodfellow (Chapter 6)
- "Neural Networks and Deep Learning" by Michael Nielsen
- "Pattern Recognition and Machine Learning" by Christopher Bishop

### Online Resources
- [3Blue1Brown Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [CS231n Course Notes](http://cs231n.github.io/neural-networks-1/)

### Practice Platforms
- [Neural Network Playground](https://playground.tensorflow.org/)
- [Kaggle Learn](https://www.kaggle.com/learn/intro-to-programming)
- [Google Colab](https://colab.research.google.com/)

---

**Ready for the next challenge?** Move on to [Backpropagation](05_backpropagation.md) to learn how to train your neurons and networks!

Happy coding! 🐍✨
