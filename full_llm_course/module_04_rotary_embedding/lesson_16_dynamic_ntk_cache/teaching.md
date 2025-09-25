# Lesson 4.3.2: RoPE Dynamic NTK Scaling - _set_cos_sin_cache in DynamicNTKScalingRotaryEmbedding

## Theory

The _set_cos_sin_cache in DynamicNTKScalingRotaryEmbedding overrides the standard to dynamically adjust the base and recompute inv_freq when the sequence length exceeds the trained max_position_embeddings. This ensures the Neural Tangent Kernel (NTK) of the attention mechanism is preserved for longer sequences, allowing better extrapolation in Kimi-K2 without retraining the positional encodings.

### Key Theory
- **Dynamic Adjustment**: If seq_len > max_position_embeddings, compute alpha = (scaling_factor * seq_len / max_position_embeddings - (scaling_factor - 1)) ** (dim / (dim - 2))
- base_new = base * alpha, then inv_freq_new = 1.0 / (base_new ** (arange(0, dim, 2) / dim))
- Register new inv_freq, then compute t, freqs, emb, cos/sin as standard.
- **NTK Preservation**: The exponent dim/(dim-2) approximates the NTK scaling from theory, making long-seq attention kernel similar to short-seq trained one.
- **When Triggered**: Only for long seq; short seq uses original inv_freq (no recompute).
- **Cost**: Recompute inv_freq O(dim), outer O(seq * dim/2); acceptable for generation (rare full recompute).

Mathematical: For long seq, ω_new = base_new^{-2i/D} = (base * alpha)^{-2i/D} = ω * alpha^{-2i/D}, adjusting high i (dims) less aggressively.

## Code Walkthrough

From DeepseekV3DynamicNTKScalingRotaryEmbedding._set_cos_sin_cache:

```python
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

    t = torch.arange(
        self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
    )

    freqs = torch.outer(t, self.inv_freq)
    # Different from paper, but it uses a different permutation in order to obtain the same calculation
    emb = torch.cat((freqs, freqs), dim=-1)
    self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
    self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
```

- self.max_seq_len_cached = seq_len: Update.
- if seq_len > self.max_position_embeddings: Compute base_new = base * alpha, alpha = (s * seq / max - (s - 1)) ** (dim / (dim - 2))
  - arange(0, dim, 2).float().to(device) / dim: Exponents.
  - base ** exponents: Powers.
  - inv_freq = 1.0 / powers
  - self.register_buffer("inv_freq", inv_freq, persistent=False): Update buffer (overwrites original).
- t = torch.arange(seq_len, ...): Positions.
- freqs = torch.outer(t, self.inv_freq): Using (possibly new) inv_freq.
- emb = cat(freqs, freqs, -1), cos/sin as standard.

For short seq, uses original inv_freq; long seq updates to new.

Example for seq=4096 > max=2048, s=2, dim=128: alpha ≈3, base_new≈3*10000, inv_freq_new smaller.

## PyTorch Functions
- ** for power, arange, outer, cat, cos/sin, register_buffer as standard.
- Conditional register_buffer: Updates buffer for long seq.

## Why This Matters
Dynamic NTK enables Kimi-K2 to extrapolate RoPE to much longer contexts while preserving learned attention patterns.

Next: Forward (inherits standard).
