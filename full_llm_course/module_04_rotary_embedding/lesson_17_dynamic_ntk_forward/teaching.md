# Lesson 4.3.3: RoPE Dynamic NTK Scaling - Forward Method in DynamicNTKScalingRotaryEmbedding

## Theory

The forward method in DeepseekV3DynamicNTKScalingRotaryEmbedding is identical to the standard variant, inheriting the cache check, slicing, and dtype casting. The dynamic NTK effect is embedded in the cache: for long sequences, the overridden _set_cos_sin_cache adjusts the base and inv_freq, providing cos/sin with modified frequencies for better extrapolation in Kimi-K2.

### Key Theory
- **Inheritance**: No override; calls parent forward, using self.cos_cached/sin_cached (adjusted in cache set if long seq).
- **Dynamic Transparency**: When forward triggers _set_cos_sin_cache for seq > max, it uses the dynamic base/inv_freq, so returned cos/sin have NTK-preserving angles.
- **Same Behavior**: Short seq use original; long seq use adjusted inv_freq for that cache.
- **In Attention**: Returned cos/sin used in apply_rotary_pos_emb; dynamic base makes long-seq attention similar to short-seq trained.
- **No Additional Cost**: Forward O(1), adjustment in cache (O(dim + seq * dim/2) for long seq recompute).

Mathematical: Same as standard, but for long seq, cos[p, d] = cos(p * inv_freq_new[d//2]) with inv_freq_new from dynamic base.

Input/Output: Same as standard: cos, sin (seq_len, dim).

## Code Walkthrough

From DeepseekV3DynamicNTKScalingRotaryEmbedding.forward (inherited, no override):

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

- Cache check calls overridden _set_cos_sin_cache, which may update inv_freq for long seq.
- Slice and cast as standard.
- Result: cos/sin with dynamic frequencies for rotation.

Example:
```python
rope = DeepseekV3DynamicNTKScalingRotaryEmbedding(dim=64, scaling_factor=2.0, max_position_embeddings=2048)
x_short = torch.randn(1,8,1000,64)  # seq=1000 <2048, uses original inv_freq
cos_short, sin_short = rope(x_short)
x_long = torch.randn(1,8,3000,64)  # seq=3000 >2048, adjusts base/inv_freq
cos_long, sin_long = rope(x_long)
print(cos_long.shape)  # (3000,64)
# cos_long uses adjusted frequencies for better long-seq performance
```

Compare to standard: For long seq, dynamic cos closer to trained short-seq patterns.

## PyTorch Functions
- Inherited: Same as standard forward.

## Why This Matters
Dynamic NTK inherits efficiency; adjustment only for long seq, forward API consistent.

Dynamic NTK complete. Next: YaRN variant.
