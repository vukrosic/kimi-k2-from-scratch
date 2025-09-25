# Lesson 4.1.3: RoPE - Forward Method in Standard Rotary Embedding

## Theory

The forward method of DeepseekV3RotaryEmbedding returns the precomputed cosine and sine tensors for the given sequence length, used in attention to rotate Q and K. It checks if the cache is sufficient; if not, extends it. This design supports variable-length inputs efficiently, key for batched training/inference in Kimi-K2.

### Key Theory
- **Input**: x (dummy, for device/dtype), seq_len (current length, e.g., from kv_seq_len in attention).
- **Cache Check**: If no cache or seq_len > cached, call _set_cos_sin_cache to recompute larger cache.
- **Slicing**: Return cos/sin [:seq_len] to match input length.
- **Dtype Cast**: to(x.dtype) ensures compatibility with Q/K (e.g., fp16), avoiding extra casts in attention.
- **No Computation in Forward**: Pure lookup/slice; heavy work in cache set. O(1) time, O(seq_len * dim) memory slice.
- **Usage in Attention**: Returned cos, sin passed to apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids).

Mathematical: For current seq_len, cos[:seq_len, :] , sin[:seq_len, :] where cos[p, d] = cos(p * inv_freq[d//2]) for even d, etc.

Input: x (bs, heads, seq, head_dim) - used for device/dtype, seq_len optional (infer from x.shape[-2] if None).

## Code Walkthrough

From DeepseekV3RotaryEmbedding.forward:

```python
def forward(self, x, seq_len=None):
    # x: [bs, num_attention_heads, seq_len, head_size]
    if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
        self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
    return (
        self.cos_cached[:seq_len].to(dtype=x.dtype),
        self.sin_cached[:seq_len].to(dtype=x.dtype),
    )
```

- if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached: First call or longer seq, extend cache with _set_cos_sin_cache(seq_len, x.device, x.dtype).
  - Uses x.device (GPU/CPU), x.dtype (fp16/32) for cache.
- self.cos_cached[:seq_len]: Slice first seq_len rows (positions 0 to seq_len-1), shape (seq_len, dim).
- .to(dtype=x.dtype): Cast to match Q/K dtype, e.g., from fp32 cache to fp16.
- Return tuple (cos, sin), both (seq_len, dim).

If seq_len=None, infer from x.shape[2] (seq_len dim).

Example:
```python
x = torch.randn(1,8,10,64)  # bs=1, heads=8, seq=10, head_dim=64
cos, sin = rope(x)  # Calls cache if needed, returns (10,64) each
print(cos.shape)  # torch.Size([10, 64])
# For longer x seq=20, auto-extends cache to 20
```

## PyTorch Functions
- Slicing [:seq_len]: View, no copy.
- to(dtype): Cast, may copy if dtype change.
- No ops on x; dummy for context.

## Why This Matters
Efficient position encoding for variable lengths; dynamic caching avoids fixed max overhead. Essential for Kimi-K2's long-context attention.

Standard RoPE complete. Next: Scaling variants.
