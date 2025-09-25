# Lesson 4.1.1: Tasks - RoPE Standard Initialization

## Theory Exercises

1. **RoPE Concept**: Explain how rotation encodes relative positions. For dim=4, base=10000, pos=0 and pos=1, compute θ for d=0,2. Show Q_1 · K_0 = |Q||K| cos(θ) for relative dist 1.

2. **Frequencies Derivation**: Why inv_freq = 1 / base^{2i/D}? For i=0, D=128, base=10000, inv_freq[0] = 1 (low freq, slow rotation). For i=63, high freq (fast rotation for high dims).

3. **Caching Rationale**: Why precompute cos/sin up to max_position_embeddings? How does it support extrapolation (use same freqs for longer seq)?

4. **Base Choice**: Why base=10000? (Wavelength ~10k tokens; too small: over-rotation short seq, too large: under-rotation long seq). Discuss for Kimi-K2's 128k context.

## Code Tasks

1. **Manual Inv_Freq Computation**:
   - dim=128, base=10000, device=None (cpu).
   - arange = torch.arange(0, dim, 2).float()  # [0,2,...,126]
   - exponents = arange / dim  # [0, 0.0156, ..., 0.984]
   - powers = base ** exponents  # [1, 10000^0.0156 ≈1.43, ...]
   - inv_freq = 1.0 / powers  # [1, 0.7, ...]
   - Print inv_freq.shape (64,), inv_freq[0] (1.0), inv_freq[-1] (small ~1e-4)

   ```python
   import torch

   dim = 128
   base = 10000
   arange = torch.arange(0, dim, 2).float()
   exponents = arange / dim
   powers = base ** exponents
   inv_freq = 1.0 / powers
   print(inv_freq.shape)  # torch.Size([64])
   print(inv_freq[0])  # tensor(1.)
   print(inv_freq[-1])  # Small value
   ```

2. **Class Creation and Inspection**:
   - Define DeepseekV3RotaryEmbedding class __init__ as in teaching (stub _set_cos_sin_cache).
   - rope = DeepseekV3RotaryEmbedding(dim=64, max_position_embeddings=512, base=5000)
   - Print rope.dim, rope.base, rope.inv_freq.shape (32,)
   - Verify rope.register_buffer called (hasattr(rope, 'inv_freq'))

3. **Simulate Frequencies**:
   - For seq_len=3, t = torch.arange(3)  # [0,1,2]
   - freqs = torch.outer(t, rope.inv_freq)  # (3,32)
   - emb = torch.cat((freqs, freqs), dim=-1)  # (3,64)
   - Print emb[0, :4] (zeros for pos=0), emb[1,0:4] (inv_freq[0],0,inv_freq[1],0)

4. **JIT Compatibility**:
   - Why call _set_cos_sin_cache in init? Test: torch.jit.script(rope) (should work with pre-built cache).
   - Stub: def _set_cos_sin_cache(self, seq_len, device, dtype): pass
   - Discuss persistent=False (not saved in state_dict for small buffers).

## Quiz

1. Shape of inv_freq? For dim=128:  
   a) (128,) b) (64,) c) (1,64)

2. True/False: RoPE has learnable parameters for positions.

3. What is base default? (Short: 10000)

## Advanced Task

Implement RoPE rotation matrix for dim=4: For pos=1, matrix R such that Q_rot = Q @ R (block diagonal cos/sin). Use inv_freq, compute angles, build R with torch.cos/sin. Apply to Q = rand(1,1,4), verify Q_rot[0,0] = Q[0,0]cosθ - Q[0,1]sinθ, etc. Discuss why applied to Q/K pairs.

Submit code, outputs, and answers.
