# Lesson 4.1.1: RoPE - Initialization of Standard Rotary Embedding

## Theory

Rotary Position Embeddings (RoPE) encode absolute positions as relative rotations in the query and key vectors of attention, allowing the model to extrapolate to longer sequences better than absolute embeddings. In Kimi-K2, RoPE is used in attention for positional information, applied only to Q and K (not V). The standard variant is the base implementation, with extensions for scaling in later lessons.

### Key Theory
- **RoPE Concept**: For position m, rotate pairs of dimensions (d, d+1) by angle θ_m^d = m * (base^{-2d/D}), where D=dim (head_dim). This injects relative distance via dot product properties: Q_m · K_n = |Q||K| cos( (m-n) θ ).
- **Frequencies**: Precompute inv_freq = 1 / base^{2i/D} for i=0 to D/2-1 (even dims). Odd dims get same.
- **Caching**: cos/sin precomputed up to max_position_embeddings (2048 default) for efficiency; extend on longer seq.
- **No Learnable Pos**: Fixed geometric, unlike learned absolute PE; base=10000 (wavelength 10k tokens).
- **Device/Dtype**: inv_freq to device; cache in default dtype.

Mathematical: freqs = pos * inv_freq, emb = cat(freqs, freqs), cos=cos(emb), sin=sin(emb). Rotation matrix for each pair.

## Code Walkthrough

From DeepseekV3RotaryEmbedding.__init__:

```python
def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
    super().__init__()
    self.dim = dim
    self.max_position_embeddings = max_position_embeddings
    self.base = base
    inv_freq = 1.0 / (
        self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
    )
    self.register_buffer("inv_freq", inv_freq, persistent=False)
    # Build here to make `torch.jit.trace` work.
    self._set_cos_sin_cache(
        seq_len=max_position_embeddings,
        device=self.inv_freq.device,
        dtype=torch.get_default_dtype(),
    )
    self.max_seq_len_cached = None
```

- self.dim, max_position_embeddings, base: Store config.
- torch.arange(0, dim, 2).float().to(device): Even indices 0,2,...,dim-2.
- / self.dim: Normalize [0, 2/dim, ..., (dim-2)/dim].
- ** (exponent): base to power, inv_freq = base^{-exponent}.
- register_buffer("inv_freq", ..., persistent=False): Non-learnable tensor, saved in state_dict but not optimized.
- _set_cos_sin_cache(max_position_embeddings, ...): Precompute cos/sin for 0 to max-1.
- self.max_seq_len_cached = None: Track for extension.

No params; all buffers. For JIT tracing, build cache early.

## PyTorch Functions
- torch.arange: Sequence tensor.
- ** operator: Power.
- register_buffer: Persistent tensor without grad.
- torch.get_default_dtype(): Model dtype (fp32/16).

## Why This Matters
RoPE's relative encoding enables Kimi-K2's long context (128k+); init sets frequencies for rotation angles.

Next: _set_cos_sin_cache method.
