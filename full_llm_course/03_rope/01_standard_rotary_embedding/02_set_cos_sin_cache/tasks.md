# Lesson 4.1.2: Tasks - RoPE _set_cos_sin_cache

## Theory Exercises

1. **Outer Product Explanation**: Why torch.outer(t, inv_freq)? For seq_len=4, D/2=2, inv_freq=[1,0.5], what is freqs? How does it create position-dependent angles?

2. **Permutation Note**: Why "different from paper"? Paper uses pair-wise (cos/sin per 2 dims), code uses full emb then cos/sin with even/odd split. Show equivalence for dim=4: Compute emb, cos/sin, vs paper's block rotations.

3. **Caching Extension**: If max_cached=4, forward seq_len=6, what happens? Recompute cache to 6, slice [:6] in forward. Why not append (recompute outer for efficiency)?

4. **Buffer Persistent=False**: Why? (Small size, JIT compatibility; not needed in state_dict for fixed freqs). When would persistent=True (e.g., learned pos)?

## Code Tasks

1. **Manual Cache Computation**:
   - dim=8 (D/2=4), seq_len=3, base=10000, compute inv_freq as in init.
   - t = torch.arange(3)
   - freqs = torch.outer(t, inv_freq)  # (3,4)
   - emb = torch.cat((freqs, freqs), dim=-1)  # (3,8)
   - cos = emb.cos(); sin = emb.sin()
   - Print cos.shape (3,8), cos[0] (all 1s for pos=0), cos[1,0:4] (cos(0*inv[0]), cos(0*inv[1]), ... wait no: for pos=1, cos(1*inv_freq)

   ```python
   import torch
   import math

   dim = 8
   seq_len = 3
   base = 10000
   inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
   t = torch.arange(seq_len)
   freqs = torch.outer(t, inv_freq)
   emb = torch.cat((freqs, freqs), dim=-1)
   cos_cached = emb.cos()
   sin_cached = emb.sin()
   print(cos_cached.shape)  # torch.Size([3, 8])
   print(cos_cached[0])  # tensor([1.,1.,1.,1.,1.,1.,1.,1.])
   ```

2. **Class Integration**:
   - Define full __init__ and _set_cos_sin_cache for DeepseekV3RotaryEmbedding (stub forward).
   - rope = DeepseekV3RotaryEmbedding(dim=16, max_position_embeddings=10)
   - Print hasattr(rope, 'cos_cached'), rope.cos_cached.shape (10,16)
   - Call rope._set_cos_sin_cache(5, device='cpu', dtype=torch.float32); print new shape (5,16)

3. **Verify Equivalence to Rotation**:
   - For pos=1, dim=4, compute cos/sin from cache.
   - Rotation for pair (d=0,1): [[cosθ, -sinθ], [sinθ, cosθ]], θ=1*inv_freq[0]
   - For Q = [q0,q1,q2,q3], Q_rot even = [q0 cos - q1 sin, q2 cos - q3 sin], odd similar.
   - But code applies in apply_rotary_pos_emb; here just compute cache values.

4. **Dynamic Extension Test**:
   - Init with max=4, check max_seq_len_cached=4
   - Call _set_cos_sin_cache(6, ...), check updated to 6, buffers resized.
   - Slice rope.cos_cached[:3] for seq=3.

## Quiz

1. Shape of freqs in _set_cos_sin_cache? For seq_len=10, dim=128:  
   a) (10,128) b) (10,64) c) (64,10)

2. True/False: emb = cat(freqs, freqs) covers odd dims with same angles as even.

3. Why to(dtype) for cos/sin? (Short: Match model precision, e.g., fp16)

## Advanced Task

Implement _set_cos_sin_cache without outer: For each pos, compute freqs_pos = pos * inv_freq, emb_pos = cat(freqs_pos, freqs_pos), stack to (seq_len, D). Time vs outer for seq_len=1024, dim=128 (outer faster vectorized). Discuss for very long seq (1M, memory for cache).

Submit code, outputs, and answers.
