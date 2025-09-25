# Lesson 3.1.1: Tasks - RMSNorm Initialization

## Theory Exercises

1. **RMSNorm vs LayerNorm**: Derive formulas. LayerNorm: subtract mean, divide std. RMSNorm: only divide RMS. Why omit mean? (Saves ~D adds, empirical equivalence in LLMs; mean ~0 in residuals.)

2. **Epsilon Role**: Why eps=1e-6? What if eps=0 (div0)? Discuss numerical stability for small hidden_size (e.g., D=64) vs large (4096).

3. **Affine Params**: Why only weight (scale), no bias? How does weight learn (per-dim adjustment)? Init to ones: Why not zeros (would zero output)?

4. **Registration in HF**: Why add to ALL_LAYERNORM_LAYERS? (For utilities like gradient checkpointing, where LayerNorm variants are treated similarly.)

## Code Tasks

1. **Basic Initialization**:
   - Define DeepseekV3RMSNorm class as in teaching.
   - rms = DeepseekV3RMSNorm(hidden_size=4096, eps=1e-5)
   - Print rms.weight.shape (torch.Size([4096])), rms.variance_epsilon (1e-5)
   - Verify type(rms.weight) == nn.Parameter, rms.weight.data.allclose(torch.ones(4096))

   ```python
   import torch
   import torch.nn as nn

   class DeepseekV3RMSNorm(nn.Module):
       def __init__(self, hidden_size, eps=1e-6):
           super().__init__()
           self.weight = nn.Parameter(torch.ones(hidden_size))
           self.variance_epsilon = eps

   rms = DeepseekV3RMSNorm(4096, eps=1e-5)
   print(rms.weight.shape)
   print(rms.variance_epsilon)
   print(torch.allclose(rms.weight.data, torch.ones(4096)))
   ```

2. **Param Count**:
   - Compute params: sum(p.numel() for p in rms.parameters()) == 4096
   - Compare to nn.LayerNorm(4096, eps=1e-5): Also 4096 (weight) + 4096 (bias) = 8192. RMSNorm saves half (no bias).

3. **Custom Eps Experiment**:
   - Create rms1 = DeepseekV3RMSNorm(64, eps=1e-8), rms2 with eps=1e-2
   - Forward dummy x = torch.randn(1,10,64) * 0.01 (small values)
   - Compute rms in forward (mean(x.pow(2)) + eps).sqrt(); see effect on small x (large eps stabilizes less? Wait, eps adds to variance).

4. **Stub with _init_weights**:
   - Add to class: def post_init(self): self.weight.data.normal_(1.0, 0.02)  # Small perturbation around 1
   - Create, call post_init, print weight.mean() ≈1, std≈0.02

## Quiz

1. Shape of self.weight?  
   a) Scalar b) (hidden_size,) c) (1, hidden_size)

2. True/False: RMSNorm has bias param like LayerNorm.

3. Default eps? (Short: 1e-6)

## Advanced Task

Implement LayerNorm equivalent without mean: class T5LayerNorm(nn.Module): ... (divide by std = sqrt(var + eps), but var = mean((x-mean)^2)). Compare compute (F.mean, F.var vs F.mean(x.pow(2))) in code timing for D=4096, B=32, T=1024. Discuss why RMSNorm faster.

Submit code, outputs, and answers.
