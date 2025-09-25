# Lesson 4.1.2: RoPE - _set_cos_sin_cache in Standard Rotary Embedding

## Theory

The _set_cos_sin_cache method precomputes the cosine and sine values for rotary embeddings up to a given sequence length, storing them as buffers for fast lookup during forward. This avoids recomputing trig functions on every call, crucial for efficiency in training/inference. In Kimi-K2, it's called in init for max_position_embeddings and in forward if seq_len > cached.

### Key Theory
- **Computation**: For positions t=0 to seq_len-1, angles = t * inv_freq (outer product for all positions and frequencies).
- **Full Emb**: cat(angles, angles, dim=-1) to cover all dims (even: angles, odd: angles).
- **Trig**: cos(emb), sin(emb) for rotation components.
- **Different Permutation**: Code uses cat(freqs, freqs) then cos/sin, equivalent to paper's pair-wise rotation but with permuted dims for same math (Q even/odd split).
- **Caching Logic**: Update max_seq_len_cached; slice in forward [:seq_len]. Supports dynamic extension.
- **Buffers**: Registered for device move with model, persistent=False (small, not always saved).

Mathematical: For pos t, freqs_t = t * inv_freq (1 x D/2), emb_t = [freqs_t, freqs_t] (1 x D), cos_t = cos(emb_t), sin_t = sin(emb_t). Cache (seq_len x D).

## Code Walkthrough

From DeepseekV3RotaryEmbedding._set_cos_sin_cache:

```python
def _set_cos_sin_cache(self, seq_len, device, dtype):
    self.max_seq_len_cached = seq_len
    t = torch.arange(
        self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
    )
    freqs = torch.outer(t, self.inv_freq.to(t.device))
    # Different from paper, but it uses a different permutation in order to obtain the same calculation
    emb = torch.cat((freqs, freqs), dim=-1)
    self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
    self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
```

- self.max_seq_len_cached = seq_len: Update cache size.
- t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype): Positions [0..seq_len-1], matching inv_freq dtype (fp32).
- freqs = torch.outer(t, self.inv_freq.to(t.device)): (seq_len, D/2), angles = pos * freq.
- emb = torch.cat((freqs, freqs), dim=-1): (seq_len, D), duplicate for odd dims.
- emb.cos().to(dtype): Cosine, cast to desired dtype (e.g., fp16).
- register_buffer("cos_cached"/"sin_cached", ..., persistent=False): Store as buffer, not param; persistent=False for JIT/small size.

Called in init with seq_len=max_position_embeddings; in forward if seq_len > cached, recompute larger.

Example for seq_len=2, dim=4 (D/2=2):
- t=[0,1], inv_freq=[1, 0.5] (toy)
- freqs = [[0,0], [1,0.5]]
- emb = [[0,0,0,0], [1,0.5,1,0.5]]
- cos_cached ≈ [[1,1,1,1], [cos1, cos0.5, cos1, cos0.5]]

## PyTorch Functions
- torch.outer: Matrix of products (t outer inv_freq).
- torch.cat(dim=-1): Concat last dim.
- emb.cos()/sin(): Element-wise trig.
- to(dtype): Cast for precision/speed.
- register_buffer: Model tensor without grad.

## Why This Matters
Caching makes RoPE O(1) per token after init; permutation ensures compatibility with attention rotation apply.

Next: Forward method (return cos/sin slices).
