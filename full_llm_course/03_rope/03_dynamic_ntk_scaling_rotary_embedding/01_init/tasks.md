# Lesson 4.3.1: Tasks - RoPE Dynamic NTK Scaling Initialization

## Theory Exercises

1. **NTK Scaling Formula**: Derive/explain base_new = base * ((s * seq / max - (s - 1)) ** (dim / (dim - 2))). Why ** (dim / (dim - 2))? (Approximates NTK preservation, from theory for kernel similarity in long seq.)

2. **Why Dynamic?**: Contrast with linear: Linear scales all pos uniformly, dynamic adjusts base based on actual seq_len, better for varying lengths. When seq <= max, no change (s=1 effective).

3. **NTK Preservation**: What is NTK in transformers? (Neural Tangent Kernel, linearizes attention for long seq; scaling maintains short-seq kernel for long.)

4. **Limitations**: Recompute inv_freq for each long seq (O(dim) cost); suitable for generation (incremental), not random long batches.

## Code Tasks

1. **Basic Class Creation**:
   - Define DeepseekV3DynamicNTKScalingRotaryEmbedding as in teaching (stub _set_cos_sin_cache from standard).
   - rope = DeepseekV3DynamicNTKScalingRotaryEmbedding(dim=64, scaling_factor=4.0, max_position_embeddings=1024)
   - Print rope.scaling_factor (4.0), rope.inv_freq.shape (32,), same as standard rope_std = DeepseekV3RotaryEmbedding(64)

   ```python
   import torch
   import torch.nn as nn

   class DeepseekV3RotaryEmbedding(nn.Module):
       # Stub standard
       def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
           super().__init__()
           self.dim = dim
           self.base = base
           inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
           self.register_buffer("inv_freq", inv_freq)

   class DeepseekV3DynamicNTKScalingRotaryEmbedding(DeepseekV3RotaryEmbedding):
       def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
           self.scaling_factor = scaling_factor
           super().__init__(dim, max_position_embeddings, base, device)

   rope_std = DeepseekV3RotaryEmbedding(64)
   rope_ntk = DeepseekV3DynamicNTKScalingRotaryEmbedding(64, scaling_factor=4.0)
   print(rope_ntk.scaling_factor)
   print(torch.allclose(rope_ntk.inv_freq, rope_std.inv_freq))  # True
   ```

2. **Stub Override**:
   - Override _set_cos_sin_cache in NTK: If seq_len > self.max_position_embeddings, compute alpha = (self.scaling_factor * seq_len / self.max_position_embeddings - (self.scaling_factor - 1)) ** (self.dim / (self.dim - 2))
   - base_new = self.base * alpha
   - inv_freq_new = 1.0 / (base_new ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
   - self.register_buffer("inv_freq", inv_freq_new)
   - Then standard t, outer, etc.
   - For seq=1024 <= max=2048, no change; for seq=4096 >2048, s=2, compute base_new.

3. **Init Comparison**:
   - Create standard and NTK with s=1.0 (identical init).
   - With s=2.0, init same (adjustment in cache), but call _set_cos_sin_cache(1000) (no change), then 3000 (adjust base for NTK).

4. **Formula Verification**:
   - dim=128, max=2048, s=2, seq=4096
   - alpha = (2 * 4096 / 2048 - (2 - 1)) ** (128 / 126) = (4 -1) ** (128/126) = 3 ** 1.016 ≈3.03
   - base_new = 10000 * 3.03 ≈30300
   - inv_freq_new smaller for high i (slower high freq).

## Quiz

1. What is adjusted in dynamic NTK?  
   a) scaling_factor b) base for long seq c) dim

2. True/False: For seq <= max_position_embeddings, identical to standard.

3. Exponent in alpha? (Short: dim / (dim - 2))

## Advanced Task

Implement the alpha formula, compute base_new for dim=64, max=2k, s=8, seq=32k. Then inv_freq_new vs original. Simulate cos for pos=31k: θ = 31k * inv_new, should be within trained range. Compare to linear scaling for same.

Submit code, outputs, and answers.
