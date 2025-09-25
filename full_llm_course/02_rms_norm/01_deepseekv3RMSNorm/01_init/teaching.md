# Lesson 3.1.1: RMSNorm - Initialization of DeepseekV3RMSNorm

## Theory

RMSNorm (Root Mean Square Normalization) is a variant of normalization used in Kimi-K2 for stabilizing activations in transformer layers. Unlike LayerNorm (which subtracts mean and divides by std), RMSNorm only divides by RMS (sqrt(mean(x^2) + eps)), omitting mean subtraction for simplicity and speed. It's equivalent to T5's LayerNorm without the mean term, reducing compute and improving performance in LLMs.

### Key Theory
- **Formula Preview**: \( y = \frac{x}{\sqrt{\frac{1}{d} \sum x_i^2 + \epsilon}} \cdot g \), where g is learnable scale (weight).
- **Why RMSNorm?**: No mean computation (saves ops, especially on GPU), empirically similar/better than LayerNorm in autoregressive models. Avoids mean shift issues in residual connections.
- **Params**: hidden_size (dim D), eps (small value, 1e-6, prevents div0).
- **Affine Transform**: Only scale (weight, ones init); no bias (common in LLMs).
- **Global Registration**: Added to ALL_LAYERNORM_LAYERS for HF utilities (e.g., gradient checkpointing).

From config: rms_norm_eps (default 1e-6).

Mathematical: RMS = sqrt( E[x^2] + eps ), y = x / RMS * weight.

## Code Walkthrough

From DeepseekV3RMSNorm.__init__:

```python
class DeepseekV3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        DeepseekV3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
```

- super().__init__(): Standard nn.Module init.
- self.weight: Learnable scale, initialized to ones (no change initially).
- self.variance_epsilon: Fixed eps for numerical stability.
- No bias param (pure scale).
- ~D params (e.g., 4096 floats).

Post-init: _init_weights applies normal_(0, std) to weight? No, for RMSNorm (LayerNorm-like), typically left as ones or small init; but in HF, ALL_LAYERNORM_LAYERS may handle differently—check code: for LayerNorm, weight.normal_(1, scale=0.02) in some impls, but here it's Parameter(ones), so post_init may adjust.

In transformers, for LayerNorm/RMSNorm, weight is often init to 1, bias 0.

## PyTorch Functions
- nn.Parameter: Registers weight for optimization.
- torch.ones: Uniform 1s for scale.

## Why This Matters
RMSNorm is lightweight (no mean), key for deep stacks in Kimi-K2 (30+ layers). Scale learns to adjust per dim.

Next: Forward pass (variance computation, normalization).
