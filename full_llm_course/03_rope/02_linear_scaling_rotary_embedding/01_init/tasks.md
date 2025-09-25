# Lesson 4.2.1: Tasks - RoPE Linear Scaling Initialization

## Theory Exercises

1. **Scaling Mechanism**: Explain how t / scaling_factor in _set_cos_sin_cache extends context. For scaling_factor=2, pos=4000 acts like pos=2000 in standard. Why linear? (Simple, preserves relative ratios approximately.)

2. **Mathematical Equivalence**: Show scaled θ = (m / s) * ω = m * (base^{-2d/D} / s) = m * ω' where ω' = ω / s. Equivalent to base^s (larger base, slower rotation).

3. **When to Use Linear Scaling**: For a model trained on 2k context, how to infer on 8k? Set scaling_factor=4. Limitations: Relative distances distorted for very large s (e.g., s=32, short seq look same as long).

4. **Inheritance Benefits**: Why subclass standard? Reuse inv_freq, forward, only override cache set. Discuss vs reimplementing all.

## Code Tasks

1. **Basic Class Creation**:
   - Define DeepseekV3LinearScalingRotaryEmbedding as in teaching (stub _set_cos_sin_cache from standard).
   - rope = DeepseekV3LinearScalingRotaryEmbedding(dim=64, scaling_factor=2.0, max_position_embeddings=1024)
   - Print rope.scaling_factor (2.0), rope.inv_freq.shape (32,), compare to standard rope_std = DeepseekV3RotaryEmbedding(64) (same inv_freq)

   ```python
   import torch
   import torch.nn as nn

   class DeepseekV3RotaryEmbedding(nn.Module):
       # Stub standard for comparison
       def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
           super().__init__()
           self.dim = dim
           self.base = base
           inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
           self.register_buffer("inv_freq", inv_freq)

   class DeepseekV3LinearScalingRotaryEmbedding(DeepseekV3RotaryEmbedding):
       def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
           self.scaling_factor = scaling_factor
           super().__init__(dim, max_position_embeddings, base, device)

   rope_std = DeepseekV3RotaryEmbedding(64)
   rope_scale = DeepseekV3LinearScalingRotaryEmbedding(64, scaling_factor=2.0)
   print(rope_scale.scaling_factor)
   print(torch.allclose(rope_scale.inv_freq, rope_std.inv_freq))  # True
   ```

2. **Stub Cache with Scaling**:
   - Override _set_cos_sin_cache in linear: t = t / self.scaling_factor, then standard.
   - Call rope_scale._set_cos_sin_cache(4, 'cpu', torch.float32)
   - Manually: t_standard = torch.arange(4); freqs_std = torch.outer(t_standard, inv_freq)
   - t_scaled = t_standard / 2.0; freqs_scale = torch.outer(t_scaled, inv_freq)
   - Verify freqs_scale[3] == freqs_std[1.5] approx (but integer, check [2] == std[1])

3. **Compare Standard vs Scaled**:
   - For seq_len=4, compute cos/sin for standard and scaled (s=2).
   - For pos=3 in scaled: angles = 3/2 * inv_freq = 1.5 * inv_freq, vs standard pos=1.5 (but since integer, compare pos=3 scaled to pos=1 standard + half, but focus on t scaling).

4. **Extrapolation Test Stub**:
   - Assume trained on max=2k, set scaling=4 for 8k.
   - Compute cache for seq=8000, but slice to 2000; angles for pos=7999 = 7999/4 * inv_freq ≈ 1999.75 * inv_freq, close to trained range.

## Quiz

1. What does linear scaling change?  
   a) inv_freq b) t in cache c) base

2. True/False: scaling_factor >1 extends effective context length.

3. Default scaling_factor? (Short: 1.0)

## Advanced Task

Implement linear scaling without / scaling_factor: Adjust inv_freq *= scaling_factor (faster freqs? No: to slow rotation, inv_freq /= scaling_factor). Show equivalence: For s=2, inv_freq_scaled = inv_freq / 2, then t * inv_freq_scaled = (t/2) * inv_freq. Test numerically for seq=4, dim=8, verify cos/sin same.

Submit code, outputs, and answers.
