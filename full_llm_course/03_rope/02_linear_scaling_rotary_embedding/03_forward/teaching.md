# Lesson 4.2.3: RoPE Linear Scaling - Forward Method in LinearScalingRotaryEmbedding

## Theory

The forward method in DeepseekV3LinearScalingRotaryEmbedding is identical to the standard variant, inheriting the logic for cache check, slicing, and dtype casting. The scaling effect is transparent: it uses the scaled cache from the overridden _set_cos_sin_cache, providing cos/sin with slower rotations for longer effective contexts in Kimi-K2 setups.

### Key Theory
- **Inheritance**: No override; calls parent forward, which uses self.cos_cached/sin_cached (scaled during cache set).
- **Scaling Transparency**: When forward calls _set_cos_sin_cache (if needed), it uses scaled t, so returned cos/sin have angles θ = (pos / s) * ω, slowing rotation.
- **Same Behavior**: Dynamic extension, slicing [:seq_len], to(x.dtype) all work as standard; scaling only affects the underlying cache values.
- **In Attention**: Returned cos/sin used in apply_rotary_pos_emb; scaling makes long positions behave like short ones.
- **No Additional Cost**: Forward remains O(1), scaling handled in cache (O(seq * dim/2) recompute if extended).

Mathematical: Same as standard, but cos[p, d] = cos( (p / s) * inv_freq[d//2] ) for even d.

Input/Output: Same as standard: cos, sin (seq_len, dim).

## Code Walkthrough

From DeepseekV3LinearScalingRotaryEmbedding.forward (inherited, no override):

```python
# Copied from standard
def forward(self, x, seq_len=None):
    if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
        self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
    return (
        self.cos_cached[:seq_len].to(dtype=x.dtype),
        self.sin_cached[:seq_len].to(dtype=x.dtype),
    )
```

- Cache check calls overridden _set_cos_sin_cache, applying t / scaling_factor.
- Slice and cast as standard.
- Result: Scaled cos/sin for rotation.

Example:
```python
rope = DeepseekV3LinearScalingRotaryEmbedding(dim=64, scaling_factor=2.0)
x = torch.randn(1,8,10,64)
cos, sin = rope(x)  # Uses scaled cache, cos[9] = cos(9/2 * inv_freq) = cos(4.5 * inv_freq)
print(cos.shape)  # (10,64)
# Same API as standard, but values scaled
```

Compare to standard: For same pos, scaled cos larger (smaller angle, closer to 1).

## PyTorch Functions
- Inherited: Same as standard forward.

## Why This Matters
Inheritance keeps code DRY; scaling only where needed (cache), forward API consistent for attention integration.

Linear scaling complete. Next: Dynamic NTK scaling variant.
