# Lesson 3.1.2: Tasks - RMSNorm Forward Pass

## Theory Exercises

1. **Step-by-Step Math**: For x = [3, 4] (D=2), eps=1e-6, weight=[1,1]. Compute variance, rsqrt(variance + eps), y. Verify ||y|| ≈1 (normalized).

2. **Dtype Handling**: Why cast to fp32? Simulate fp16: Small x (1e-4 scale), compute variance in fp16 vs fp32. When does fp16 underflow (variance~0, rsqrt~inf)?

3. **Broadcasting**: Explain shapes: hidden (B,T,D) * rsqrt (B,T,1) * weight (D,). Why keepdim=True? What if mean without keepdim (error in broadcast)?

4. **RMSNorm vs LayerNorm**: Implement LayerNorm forward (subtract mean, divide std). For random x, compare outputs (similar if mean~0). Discuss compute savings (no mean/var, just mean(x^2)).

## Code Tasks

1. **Step-by-Step Forward**:
   - Use DeepseekV3RMSNorm class.
   - x = torch.tensor([[[3.0, 4.0]]])  # B=1,T=1,D=2
   - Manually: input_dtype = x.dtype; x_fp32 = x.to(torch.float32)
   - var = x_fp32.pow(2).mean(-1, keepdim=True)  # [[5.0]]
   - rms = torch.rsqrt(var + 1e-6)  # ~0.447
   - normed = x_fp32 * rms  # [[1.34, 1.79]]
   - y = rms.weight * normed.to(input_dtype)  # Same if weight=1
   - Compare to rms(x), verify equality.

   ```python
   import torch
   import torch.nn as nn

   class DeepseekV3RMSNorm(nn.Module):
       # From teaching
       def __init__(self, hidden_size, eps=1e-6):
           super().__init__()
           self.weight = nn.Parameter(torch.ones(hidden_size))
           self.variance_epsilon = eps
       def forward(self, hidden_states):
           # From teaching
           input_dtype = hidden_states.dtype
           hidden_states = hidden_states.to(torch.float32)
           variance = hidden_states.pow(2).mean(-1, keepdim=True)
           hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
           return self.weight * hidden_states.to(input_dtype)

   rms = DeepseekV3RMSNorm(2)
   x = torch.tensor([[[3.0, 4.0]]])
   y_model = rms(x)
   # Manual steps here
   print(torch.allclose(y_model, y_manual))
   ```

2. **Dtype Effects**:
   - x = torch.randn(1,1,4096, dtype=torch.float16) * 1e-3  # Small fp16
   - Compute forward in fp16-only (modify class, no to(fp32)): See NaN or large values?
   - With fp32 cast: Stable, ||y||~1.
   - Print variance before/after cast.

3. **Broadcast Test**:
   - x = torch.randn(2,3,4)  # B=2,T=3,D=4
   - var_no_keep = x.pow(2).mean(-1)  # (2,3), error in * x
   - var_keep = x.pow(2).mean(-1, keepdim=True)  # (2,3,1), broadcasts ok.
   - Implement forward with/without keepdim, catch error.

4. **Compare to LayerNorm**:
   - ln = nn.LayerNorm(4096, eps=1e-6)
   - x = torch.randn(32,1024,4096)
   - y_rms = rms(x); y_ln = ln(x)
   - Compute mean(abs(y_rms - y_ln))  # Small if mean(x)~0
   - Time both: %timeit rms(x), %timeit ln(x)  # RMSNorm faster

## Quiz

1. Shape of variance?  
   a) (B,T,D) b) (B,T,1) c) (B,T)

2. True/False: rsqrt(variance + eps) is used for efficiency over /sqrt.

3. Why to(input_dtype) at end? (Short: Match input precision, e.g., fp16)

## Advanced Task

Implement fused RMSNorm (custom function with torch ops, no class). Test numerical stability: x with extreme values (1e10, 1e-10), compare to standard. Profile GPU time for B=32,T=1024,D=4096. Discuss if fp32 cast necessary in bf16 (better range).

Submit code, outputs, and answers.
