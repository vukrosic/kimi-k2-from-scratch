# Tutorial 1: Understanding RMSNorm in DeepSeek-V3

## What is RMSNorm?

RMSNorm (Root Mean Square Normalization) is a normalization technique used in transformer models like DeepSeek-V3. It normalizes the input by dividing by the root mean square of the input values, followed by scaling with learnable parameters. It's similar to LayerNorm but computationally more efficient as it avoids computing mean and variance separately.

In DeepSeek-V3, it's implemented as `DeepseekV3RMSNorm`, which is equivalent to T5's LayerNorm.

### Key Benefits:
- Faster than LayerNorm (no mean subtraction).
- Helps stabilize training in deep networks.
- Used before attention and MLP layers.

## Code Implementation

Here's the core implementation from the model:

```python
import torch
import torch.nn as nn

class DeepseekV3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
```

### Line-by-Line Breakdown

Let's dissect the code step by step.

#### Initialization (`__init__` method):
- `super().__init__()`: Calls the parent `nn.Module` constructor to set up the module.
- `self.weight = nn.Parameter(torch.ones(hidden_size))`: Creates a learnable parameter tensor of ones with shape `(hidden_size,)`. This weight scales the normalized output and is updated during training.
- `self.variance_epsilon = eps`: Sets a small constant (default 1e-6) to prevent division by zero in the normalization.

This setup ensures the norm starts as identity (multiplying by 1) but learns to adjust scales per dimension.

#### Forward Pass (`forward` method):
- `input_dtype = hidden_states.dtype`: Saves the input's data type (e.g., float16 for efficiency) to restore it later.
- `hidden_states = hidden_states.to(torch.float32)`: Temporarily casts to float32 for accurate computations, avoiding precision issues in lower dtypes.
- `variance = hidden_states.pow(2).mean(-1, keepdim=True)`: 
  - `hidden_states.pow(2)`: Squares each element to compute the mean squared value.
  - `.mean(-1, keepdim=True)`: Averages along the last dimension (features for each token), keeping the dimension for broadcasting. This gives the variance (RMS^2) per token per batch item, shape `(batch_size, seq_len, 1)`.
- `hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)`:
  - `variance + self.variance_epsilon`: Adds epsilon for stability.
  - `torch.rsqrt(...)`: Computes 1 / sqrt(variance), the RMS normalization factor.
  - Multiplies the original hidden_states by this factor, normalizing so each token's features have RMS ≈ 1.
- `return self.weight * hidden_states.to(input_dtype)`: Scales by the learned weights and casts back to the original dtype.

This process normalizes without mean subtraction, making it faster than LayerNorm.

## Step-by-Step Example Walkthrough

Let's trace through the example code:

1. Define dimensions: `hidden_size = 512`, `batch_size=2`, `seq_len=10`. This simulates a small batch of 2 sequences, each with 10 tokens, each token having 512 features.
2. `hidden_states = torch.randn(batch_size, seq_len, hidden_size)`: Generates random input tensor, e.g., values around -1 to 1.
3. `norm = DeepseekV3RMSNorm(hidden_size)`: Instantiates the norm layer with 512 weights initialized to 1.
4. `normalized = norm(hidden_states)`: 
   - Inside forward: Cast to fp32.
   - Compute variance: For each of the 20 tokens (2x10), average of 512 squared features, say ~1.0 for random data.
   - rsqrt(variance + 1e-6) ≈ 1 / sqrt(1) = 1, so initial normalization is near-identity.
   - Multiply by weights (all 1s), cast back.
5. Print shapes: All `(2, 10, 512)`, confirming no shape change. Weights shape `(512,)` shows per-feature scaling.

In training, weights learn to emphasize important features. For visualization, you could print `torch.norm(normalized, dim=-1)` to see per-token norms close to 1 * weight norms.

## Example Usage

```python
# Example with dummy input
hidden_size = 512
batch_size, seq_len = 2, 10
hidden_states = torch.randn(batch_size, seq_len, hidden_size)

norm = DeepseekV3RMSNorm(hidden_size)
normalized = norm(hidden_states)

print(f"Input shape: {hidden_states.shape}")
print(f"Normalized shape: {normalized.shape}")
print(f"Weight shape: {norm.weight.shape}")
```

To verify, add:
```python
print(f"RMS before: {torch.sqrt((hidden_states ** 2).mean(-1)).mean()}")
print(f"RMS after: {torch.sqrt((normalized ** 2).mean(-1)).mean()}")
```
RMS after should be ≈1 initially, demonstrating normalization.

## Why Use RMSNorm in DeepSeek-V3?

DeepSeek-V3 uses RMSNorm for efficiency in its large-scale MoE architecture. It's applied pre-attention and pre-MLP to prevent gradient explosion/vanishing.

Next Tutorial: Rotary Position Embeddings.
