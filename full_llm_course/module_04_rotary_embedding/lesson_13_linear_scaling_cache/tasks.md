# Lesson 4.2.2: Tasks - RoPE Linear Scaling _set_cos_sin_cache

## Theory Exercises

1. **Scaling Application**: For scaling_factor=4, seq_len=8, how does t look? t = [0,0.25,0.5,0.75,1,1.25,1.5,1.75]. Angles for pos=7 = 1.75 * ω vs standard 7*ω (1/4 speed). Why slower rotation extends context?

2. **Equivalence to Base Change**: Show scaling s equivalent to base' = base^s. For s=2, base=10000, base' =10000^2 =1e8, inv_freq' = 1 / base'^{2i/D} = 1 / (base^{2s i/D}) = (1 / base^{2i/D}) ^ s = inv_freq ^ s? Wait, no: Actually, θ' = m * (base'^{-2i/D}) = m * (base^{-s * 2i/D}) = (m / s) * (base^{-2i/D} * s^{2i/D -1} wait, approximate for small i/D.

3. **Dynamic Update with Scaling**: If cached for seq=4 with s=2, then forward seq=8, recompute t /2 for 0-7. How does it blend old/new? (Recompute all, but consistent).

4. **Limitations Math**: For s=8, relative dist 1 at pos=8000: θ_rel = (8000 /8 - 7999/8) * ω = (1000 - 999.875) * ω = 0.125 * ω, vs short seq dist1 θ=1*ω (distorted).

## Code Tasks

1. **Override and Basic Test**:
   - Define full DeepseekV3LinearScalingRotaryEmbedding with overridden _set_cos_sin_cache (t / scaling_factor, then standard cat/cos/sin).
   - rope = DeepseekV3LinearScalingRotaryEmbedding(dim=8, scaling_factor=2.0, max_position_embeddings=2)
   - Call rope._set_cos_sin_cache(4, 'cpu', torch.float32)
   - t_scaled = torch.arange(4) / 2.0  # [0,0.5,1,1.5]
   - freqs = torch.outer(t_scaled, rope.inv_freq)  # (4,4)
   - emb = torch.cat((freqs, freqs), -1)
   - Verify rope.cos_cached.allclose(emb.cos())

   ```python
   import torch
   import torch.nn as nn

   class DeepseekV3RotaryEmbedding(nn.Module):
       # Standard parent
       def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
           super().__init__()
           self.dim = dim
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

   class DeepseekV3LinearScalingRotaryEmbedding(DeepseekV3RotaryEmbedding):
       def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
           self.scaling_factor = scaling_factor
           super().__init__(dim, max_position_embeddings, base, device)

       def _set_cos_sin_cache(self, seq_len, device, dtype):
           self.max_seq_len_cached = seq_len
           t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
           t = t / self.scaling_factor
           freqs = torch.outer(t, self.inv_freq.to(t.device))
           emb = torch.cat((freqs, freqs), dim=-1)
           self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
           self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

   rope = DeepseekV3LinearScalingRotaryEmbedding(8, scaling_factor=2.0)
   rope._set_cos_sin_cache(4, 'cpu', torch.float32)
   # Manual verification
   t_scaled = torch.arange(4) / 2.0
   freqs = torch.outer(t_scaled, rope.inv_freq)
   emb = torch.cat((freqs, freqs), dim=-1)
   print(torch.allclose(rope.cos_cached, emb.cos()))
   ```

2. **Compare Standard vs Scaled Cache**:
   - rope_std = DeepseekV3RotaryEmbedding(8)
   - rope_scale = DeepseekV3LinearScalingRotaryEmbedding(8, scaling_factor=2.0)
   - Both _set_cos_sin_cache(4)
   - For pos=3, standard θ3 = 3 * inv_freq, scaled θ3 = 1.5 * inv_freq
   - Print cos_std[3,0] = cos(3*inv[0]), cos_scale[3,0] = cos(1.5*inv[0])

3. **Dynamic Extension with Scaling**:
   - rope_scale._set_cos_sin_cache(2)  # Small cache
   - Call forward with x seq=5 (extends to 5, t/2 = [0,0.5,1,1.5,2])
   - Verify max_seq_len_cached=5, cos.shape=(5,8)

4. **Equivalence Test**:
   - For s=2, compute scaled cache for seq=4.
   - Standard cache for seq=2 (t=0,1), but to match scaled pos=3 (1.5), interpolate or note non-integer.

## Quiz

1. In overridden _set_cos_sin_cache, what is scaled?  
   a) inv_freq b) t c) emb

2. True/False: For scaling_factor=1, identical to standard.

3. Effect of scaling_factor>1? (Short: Slower rotations, longer effective context)

## Advanced Task

Test extrapolation: Assume trained on seq=2k with s=1, now s=4 for 8k. Compute cos for pos=7999 with s=4: θ = 7999/4 * ω ≈ 2000 * ω, within trained range. Compare to standard at pos=7999 (over-rotated). Simulate dot product Q_pos2000 · K_pos1999 vs scaled Q_pos8000 · K_pos7999 (similar cos rel θ=1*ω).

Submit code, outputs, and answers.
