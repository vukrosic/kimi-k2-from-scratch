# Lesson 4.2.2: RoPE Linear Scaling - _set_cos_sin_cache in LinearScalingRotaryEmbedding

## Theory

The _set_cos_sin_cache in LinearScalingRotaryEmbedding overrides the standard to apply linear scaling to the position tensor t before computing frequencies. This adjustment "slows down" the rotations, allowing the same frequency set (inv_freq) to cover longer sequences without over-rotating, enabling context length extrapolation in Kimi-K2 configurations.

### Key Theory
- **Scaling Application**: t_scaled = t / scaling_factor, then freqs = outer(t_scaled, inv_freq). For scaling_factor >1, angles θ = (pos / s) * ω smaller, rotations slower, effective wavelength *s.
- **Inheritance Override**: Calls parent logic after scaling t; same emb, cos/sin computation.
- **Dynamic Update**: Like standard, updates cache if seq_len > cached, using scaled t for new positions.
- **Benefits**: Simple way to extend context (e.g., s=8 for 2k to 16k); no change to trained model weights.
- **Drawbacks**: Linear scaling distorts relative positions non-uniformly; better for moderate extensions.

Mathematical: Standard freqs = pos * inv_freq, scaled freqs = (pos / s) * inv_freq = pos * (inv_freq / s). But code scales t, equivalent.

For s=2, pos=3: θ = 1.5 * ω vs standard 3*ω (half speed).

## Code Walkthrough

From DeepseekV3LinearScalingRotaryEmbedding._set_cos_sin_cache:

```python
def _set_cos_sin_cache(self, seq_len, device, dtype):
    self.max_seq_len_cached = seq_len
    t = torch.arange(
        self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
    )
    t = t / self.scaling_factor
    freqs = torch.outer(t, self.inv_freq)
    # Different from paper, but it uses a different permutation in order to obtain the same calculation
    emb = torch.cat((freqs, freqs), dim=-1)
    self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
    self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
```

- self.max_seq_len_cached = seq_len: Update as standard.
- t = torch.arange(seq_len, device, dtype): Positions [0..seq_len-1].
- t = t / self.scaling_factor: Scale down (e.g., /2 for s=2).
- freqs = torch.outer(t, self.inv_freq): Scaled angles (seq_len, D/2).
- emb = torch.cat((freqs, freqs), dim=-1): (seq_len, D).
- register_buffer cos/sin as standard.

Only difference: t scaling line. Called from init/forward like parent.

Example for seq_len=4, s=2, dim=4 (D/2=2), inv_freq=[1,0.5]:
- t = [0,1,2,3] /2 = [0,0.5,1,1.5]
- freqs = [[0,0], [0.5,0.25], [1,0.5], [1.5,0.75]]
- emb row 3: [1.5,0.75,1.5,0.75], cos/sin accordingly (slower than standard [3,1.5,3,1.5]).

## PyTorch Functions
- / self.scaling_factor: Element-wise divide on t.
- Rest same as standard: outer, cat, cos/sin, register.

## Why This Matters
Enables seamless context extension; single param s controls effective length without retraining RoPE.

Next: Forward (inherits standard, no change).
