# Lesson 4.3.2: Tasks - RoPE Dynamic NTK Scaling _set_cos_sin_cache

## Theory Exercises

1. **Dynamic Adjustment**: Explain the if seq_len > max_position_embeddings condition. For seq=1024 <= max=2048, no change; for seq=4096, compute alpha, base_new. Why only then? (For short seq, use trained frequencies.)

2. **Alpha Formula**: For dim=64, max=2k, s=2, seq=4k, compute alpha = (2*4k/2k -1) ** (64/62) = 3 ** 1.032 ≈3.1. base_new =10000*3.1=31k. How does larger base slow rotations?

3. **Inv_Freq Update**: Why register new inv_freq? Overwrites original for this cache. For multiple calls, consistent? (Yes, since seq increasing in generation.)

4. **NTK Theory**: Why dim/(dim-2)? (Approximation from NTK derivation for rotary kernel; as dim→∞, →1, linear-like.)

## Code Tasks

1. **Override and Basic Test**:
   - Define full DeepseekV3DynamicNTKScalingRotaryEmbedding with overridden _set_cos_sin_cache (if seq > max, compute alpha, base_new, inv_freq_new, register, then standard t outer etc.).
   - rope = DeepseekV3DynamicNTKScalingRotaryEmbedding(dim=32, scaling_factor=2.0, max_position_embeddings=8)
   - Call rope._set_cos_sin_cache(6)  # <8, no change, use original inv_freq
   - Print rope.inv_freq[0] (1.0)
   - Call rope._set_cos_sin_cache(10)  # >8, update
   - Print new rope.inv_freq[0] (smaller than original? Wait, base_new > base, inv_freq_new < inv_freq for all i)

   ```python
   import torch
   import torch.nn as nn

   class DeepseekV3RotaryEmbedding(nn.Module):
       # Standard parent
       def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
           super().__init__()
           self.dim = dim
           self.max_position_embeddings = max_position_embeddings
           self.base = base
           inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
           self.register_buffer("inv_freq", inv_freq)
           self.max_seq_len_cached = None

       def _set_cos_sin_cache(self, seq_len, device, dtype):
           self.max_seq_len_cached = seq_len
           t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
           freqs = torch.outer(t, self.inv_freq.to(t.device))
           emb = torch.cat((freqs, freqs), dim=-1)
           self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
           self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

   class DeepseekV3DynamicNTKScalingRotaryEmbedding(DeepseekV3RotaryEmbedding):
       def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
           self.scaling_factor = scaling_factor
           super().__init__(dim, max_position_embeddings, base, device)

       def _set_cos_sin_cache(self, seq_len, device, dtype):
           self.max_seq_len_cached = seq_len

           if seq_len > self.max_position_embeddings:
               base = self.base * (
                   (self.scaling_factor * seq_len / self.max_position_embeddings)
                   - (self.scaling_factor - 1)
               ) ** (self.dim / (self.dim - 2))
               inv_freq = 1.0 / (
                   base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
               )
               self.register_buffer("inv_freq", inv_freq, persistent=False)

           t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
           freqs = torch.outer(t, self.inv_freq.to(t.device))
           emb = torch.cat((freqs, freqs), dim=-1)
           self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
           self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

   rope = DeepseekV3DynamicNTKScalingRotaryEmbedding(32, max_position_embeddings=8, scaling_factor=2.0)
   print('Original inv[0]:', rope.inv_freq[0].item())
   rope._set_cos_sin_cache(6, 'cpu', torch.float32)  # No change
   print('After short:', rope.inv_freq[0].item())
   rope._set_cos_sin_cache(10, 'cpu', torch.float32)  # Update
   print('After long:', rope.inv_freq[0].item())  # Smaller
   ```

2. **Alpha Computation**:
   - dim=64, max=2048, s=2, seq=4096
   - alpha = (2 * 4096 / 2048 - 1) ** (64 / 62) = 3 ** 1.032 ≈3.1
   - base_new = 10000 * 3.1 = 31000
   - exponents = torch.arange(0,64,2).float() / 64
   - inv_freq_new = 1 / (base_new ** exponents)
   - Print inv_freq_new[0] =1, inv_freq_new[-1] < original (slower high freq)

3. **Compare to Standard**:
   - rope_std = DeepseekV3RotaryEmbedding(32)
   - rope_ntk = ... (s=2, max=8)
   - _set_cos_sin_cache(6) for both (no change, same)
   - _set_cos_sin_cache(10) for ntk (update), std no
   - Print rope_ntk.inv_freq.mean() < rope_std.inv_freq.mean() (larger base, smaller inv)

4. **Cache After Update**:
   - After long call, cos_cached uses new inv_freq, print cos_cached[9,0] = cos(9 * new_inv[0]) = cos(9), but since new_inv smaller, angle smaller.

## Quiz

1. When is base_new computed?  
   a) Always b) seq > max_position_embeddings c) seq < max

2. True/False: Updating inv_freq overwrites the original buffer.

3. What is alpha for s=1? (Short: 1, no change)

## Advanced Task

Test NTK vs linear for long seq: For max=2k, seq=8k, s=4. Compute base_new for NTK, inv_freq_new. For linear, t /4. Show for high dim i, NTK adjusts less than linear (alpha ^ -2i/D smaller change for large i). Simulate cos for pos=7999, high dim.

Submit code, outputs, and answers.
