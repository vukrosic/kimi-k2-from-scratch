# Tutorial 2: Rotary Position Embeddings (RoPE) in DeepSeek-V3

## What is Rotary Position Embedding (RoPE)?

RoPE is a position encoding method for transformers that encodes relative positional information by rotating the query and key vectors in the attention mechanism. Unlike absolute positional embeddings, RoPE is more efficient for long sequences and allows extrapolation beyond training lengths with scaling techniques.

In DeepSeek-V3, it's implemented via `DeepseekV3RotaryEmbedding` and variants for scaling (linear, dynamic NTK, YaRN).

### Key Benefits:
- Captures relative positions naturally.
- Supports longer context lengths with scaling.
- Reduces the need for absolute position embeddings.

## Code Implementation

Core Rotary Embedding class:

```python
import torch
import torch.nn as nn
import math

class DeepseekV3RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )
        self.max_seq_len_cached = None

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(t.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )
```

### Line-by-Line Breakdown

Let's break down the Rotary Embedding class and the application function.

#### Initialization (`__init__` method):
- `super().__init__()`: Inherits from `nn.Module`.
- `self.dim = dim`: Stores the head dimension (e.g., 64 or 128).
- `self.max_position_embeddings = max_position_embeddings`: Maximum sequence length for precomputing (default 2048).
- `self.base = base`: Base frequency for rotations (default 10000).
- `inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))`: 
  - `torch.arange(0, self.dim, 2)`: Even indices from 0 to dim-2 (for pairs in rotations).
  - Divide by dim and raise base to that power: Creates decreasing frequencies θ_i = base^(-2i/dim).
  - `1.0 / ...`: Inverse frequencies for outer product with positions.
- `self.register_buffer("inv_freq", inv_freq, persistent=False)`: Registers as non-learnable buffer.
- `self._set_cos_sin_cache(...)`: Immediately caches cos/sin for max length.
- `self.max_seq_len_cached = None`: Tracks cached length.

This precomputes frequencies for efficient rotation angles.

#### Cache Setup (`_set_cos_sin_cache` method):
- `self.max_seq_len_cached = seq_len`: Updates cache length.
- `t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)`: Positions 0 to seq_len-1.
- `freqs = torch.outer(t, self.inv_freq.to(t.device))`: Angles θ_m,i = m * θ_i for each position m and pair i. Shape (seq_len, dim/2).
- `emb = torch.cat((freqs, freqs), dim=-1)`: Duplicates for sin/cos pairs, shape (seq_len, dim).
- `self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)`: Cosine of angles.
- `self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)`: Sine of angles.

Caches for reuse, avoiding recomputation.

#### Forward (`forward` method):
- If cache too small, update it.
- Slice to seq_len and cast to x's dtype: Returns (cos, sin) of shape (seq_len, dim).

#### Applying RoPE (`apply_rotary_pos_emb` function):
This rotates Q and K using cos/sin. Here's the implementation:

```python
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    
    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    
    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
```

- `cos/sin[position_ids].unsqueeze(unsqueeze_dim)`: Selects angles for positions, unsqueezes for broadcasting (dim=1 for heads).
- Reshape Q/K to (b, h, s, d/2, 2): Pairs (x_re, x_im) for complex multiplication.
- `transpose(4, 3)`: Swaps to apply rotation: e^(iθ) * (x_re + i x_im) = (x_re cos - x_im sin, x_re sin + x_im cos).
- `rotate_half(q)`: Shifts to (-x_im, x_re) for the imaginary part.
- Final: q * cos + rotated * sin implements the rotation.

## Example Usage

```python
# Dummy setup
dim = 64  # Head dimension
seq_len = 20
batch_size, num_heads = 2, 8

# Create embedding
rope = DeepseekV3RotaryEmbedding(dim)

# Dummy query and key (bs, heads, seq, dim)
q = torch.randn(batch_size, num_heads, seq_len, dim)
k = torch.randn(batch_size, num_heads, seq_len, dim)

# Position IDs
position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

# Get cos/sin
cos, sin = rope(q, seq_len=seq_len)

# Apply RoPE
q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1)

print(f"Original Q shape: {q.shape}")
print(f"Rotated Q shape: {q_rot.shape}")
```

This rotates the vectors, embedding positional info via rotations. The relative angle between positions m and n is (m-n)*θ, naturally captured in dot products.

## Step-by-Step Example Walkthrough

Trace the example:

1. Setup: `dim=64`, `seq_len=20`, `batch_size=2`, `num_heads=8`. Simulates multi-head attention.
2. `rope = DeepseekV3RotaryEmbedding(dim)`: Creates with inv_freq shape (32,), caches cos/sin (2048, 64).
3. `q = torch.randn(2,8,20,64)`, `k` similar: Random queries/keys.
4. `position_ids = torch.arange(20).unsqueeze(0).expand(2, -1)`: Positions 0-19 for each batch.
5. `cos, sin = rope(q, seq_len=20)`: Returns sliced (20,64) tensors.
6. `q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1)`:
   - cos/sin become (2,1,20,64) after unsqueeze and position select (but since position_ids=0:19, same as cos/sin).
   - Reshape q to (2,8,20,32,2), transpose to apply rotation formula.
   - rotate_half swaps and negates half: For position 0, no rotation; for higher positions, rotates by increasing angles.
   - Result: q_rot has positional encoding baked in via rotations.

To visualize, compute dot product q_rot @ k_rot.T; relative positions affect similarity more than absolute.

## Why Use RoPE in DeepSeek-V3?

DeepSeek-V3 employs RoPE for its efficiency in handling long contexts (up to 128K tokens) with variants like YaRN for better extrapolation, crucial for its MoE-based architecture.

Next Tutorial: MLP Layer.
