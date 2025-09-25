# Lesson 4.3.1: RoPE Dynamic NTK Scaling - Initialization of DynamicNTKScalingRotaryEmbedding

## Theory

The DynamicNTKScalingRotaryEmbedding extends standard RoPE with dynamic adjustment of the base frequency for sequences longer than the trained max_position_embeddings. Based on NTK theory, it scales the base to maintain the kernel's properties, enabling better extrapolation to longer contexts in Kimi-K2 without degrading performance. It's more sophisticated than linear scaling, adjusting frequencies non-uniformly.

### Key Theory
- **NTK Scaling**: For seq_len > max_position_embeddings, recompute base = original_base * ((scaling_factor * seq_len / max) - (scaling_factor - 1)) ** (dim / (dim - 2)). This "stretches" high frequencies less than low ones, preserving NTK similarity.
- **When to Use**: For inference on much longer contexts (e.g., trained 4k, infer 128k); dynamic base change based on actual seq_len.
- **Inheritance**: Subclasses DeepseekV3RotaryEmbedding, overrides _set_cos_sin_cache for base/inv_freq update; init adds scaling_factor.
- **Credits**: From Reddit /u/bloc97 and /u/emozilla, copied from Llama impl.
- **Limitations**: Recomputes inv_freq for long seq (costly if frequent); best for generation where seq grows gradually.

Mathematical: If seq > max, base_new = base * alpha, alpha = (s * seq / max - (s - 1)) ^{D/(D-2)}, inv_freq_new = 1 / base_new ^{2i/D}. For s=1, no change.

Params: scaling_factor=1.0 (no scale).

## Code Walkthrough

From DeepseekV3DynamicNTKScalingRotaryEmbedding.__init__:

```python
class DeepseekV3DynamicNTKScalingRotaryEmbedding(DeepseekV3RotaryEmbedding):
    """DeepseekV3RotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)
```

- Inherits DeepseekV3RotaryEmbedding: Sets dim, max_position_embeddings, base, initial inv_freq, cache.
- self.scaling_factor = scaling_factor: Store (default 1.0).
- super().__init__(...): Parent init, including initial _set_cos_sin_cache with original base.
- Adjustment in overridden _set_cos_sin_cache: If seq_len > max_position_embeddings, update base and inv_freq.

For scaling_factor=8.0, long seq gets dynamically larger base.

Example:
```python
rope = DeepseekV3DynamicNTKScalingRotaryEmbedding(dim=64, scaling_factor=8.0)
# Init with standard base, but forward long seq adjusts
```

## PyTorch Functions
- Inheritance: super() calls parent.
- No new ops in init; dynamic in cache set.

## Why This Matters
Dynamic NTK allows Kimi-K2 to handle ultra-long contexts (e.g., 1M tokens) with trained short RoPE, preserving attention patterns via NTK.

Next: Overridden _set_cos_sin_cache for dynamic base.
