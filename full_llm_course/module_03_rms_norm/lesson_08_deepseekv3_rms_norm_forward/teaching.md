# Lesson 3.1.2: RMSNorm - Forward Pass of DeepseekV3RMSNorm

## Theory

The forward pass of RMSNorm normalizes hidden states along the feature dimension (last dim), applying a learned scale. This stabilizes gradients and activations in deep transformers like Kimi-K2, where it's used pre-attention and post-attention. Computation is efficient: no mean subtraction, just RMS division.

### Key Theory
- **Step-by-Step**:
  1. Cast to fp32 for precision (avoids fp16 underflow in variance).
  2. Compute variance = mean(x^2, dim=-1, keepdim=True): Per-sample RMS^2.
  3. Normalize: x / sqrt(variance + eps) (rsqrt for efficiency).
  4. Scale: * weight (per-feature adjustment).
  5. Cast back to input dtype (e.g., fp16 for speed).
- **Why fp32 Intermediate?**: Variance in fp16 can be tiny, leading to large rsqrt; fp32 ensures stability.
- **Keepdim=True**: Maintains shape for broadcasting (e.g., (B,T,1) * (B,T,D)).
- **No Bias**: Pure normalization + scale; residuals handle shifts.

Mathematical: Let \( \sigma = \sqrt{ \frac{1}{D} \sum_{i=1}^D x_i^2 + \epsilon } \), then \( y_j = \frac{x_j}{\sigma} \cdot g_j \), for j=1 to D.

Input: hidden_states (B, T, D), Output: same shape.

## Code Walkthrough

From DeepseekV3RMSNorm.forward:

```python
def forward(self, hidden_states):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)
```

- input_dtype = hidden_states.dtype: Save original (e.g., fp16/bf16 for AMP).
- hidden_states = hidden_states.to(torch.float32): Cast for accurate computation.
- variance = hidden_states.pow(2).mean(-1, keepdim=True): x^2 mean over features, shape (B,T,1).
  - pow(2): Element-wise square.
  - mean(-1, keepdim=True): Reduce last dim, keep for broadcast.
- hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon): Normalize.
  - rsqrt(z) = 1/sqrt(z): Faster than sqrt then reciprocal.
  - + eps: Stability.
  - * : Broadcasts (B,T,D) * (B,T,1).
- return self.weight * hidden_states.to(input_dtype): Scale (broadcast (D,)) * normalized, cast back.

Efficiency: O(D) per token, vectorized.

Example:
```python
x = torch.tensor([[[1.0, 2.0]]])  # B=1,T=1,D=2
rms = DeepseekV3RMSNorm(2)
y = rms(x)
# variance = mean([1,4]) = 2.5, rms=sqrt(2.5+eps)≈1.58
# y ≈ [[1/1.58 *1, 2/1.58 *1]] ≈ [[0.63, 1.26]]
```

## PyTorch Functions
- to(dtype): Cast tensor.
- pow(2): x**2.
- mean(dim, keepdim): Reduce with dim.
- rsqrt: 1/sqrt (fused op).
- Broadcasting in * for shapes.

## Why This Matters
RMSNorm enables deep training without cov-shift; fp32 cast prevents fp16 issues in large models like Kimi-K2.

Next: RMSNorm section complete. RoPE follows.
