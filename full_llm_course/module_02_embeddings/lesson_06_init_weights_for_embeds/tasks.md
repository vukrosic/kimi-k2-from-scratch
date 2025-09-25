# Lesson 2.3: Tasks - Embeddings Initialization in _init_weights

## Theory Exercises

1. **Init Goals**: Explain vanishing/exploding gradients. How does normal(0, 0.02) help? Contrast with uniform init or zero init for embeddings.

2. **Variance Preservation**: For embeddings, why unit variance? Derive approximate std for uniform(-a,a): std = a/sqrt(3). For std=0.02, what a?

3. **Padding in Init**: Why zero weight[padding_idx] after normal_? How does it affect loss computation (ignore pad tokens)?

4. **Fixed vs Adaptive Init**: Why fixed std=0.02 in transformers vs fan-in (1/sqrt(D))? Discuss for large D=4096.

## Code Tasks

1. **Apply _init_weights**:
   - Create embed = nn.Embedding(100, 64, padding_idx=0)
   - Save initial weight norm: torch.norm(embed.weight.data)
   - Define _init_weights function as in teaching (std=0.02)
   - embed.apply(_init_weights)  # Recursive, but single module
   - Print new norm ≈ sqrt(100*64) * 0.02 ≈ 1.28 (unit variance per dim)

   ```python
   import torch
   import torch.nn as nn

   def _init_weights(module, std=0.02):
       if isinstance(module, nn.Embedding):
           module.weight.data.normal_(0.0, std)
           if module.padding_idx is not None:
               module.weight.data[module.padding_idx].zero_()

   embed = nn.Embedding(100, 64, padding_idx=0)
   print('Initial norm:', torch.norm(embed.weight.data))
   embed.apply(lambda m: _init_weights(m))
   print('After init norm:', torch.norm(embed.weight.data))
   print('Padding zero:', torch.all(embed.weight.data[0] == 0))
   ```

2. **Variance Check**:
   - Before/after init: Compute var = torch.var(embed.weight.data).item()
   - Expected after: ≈ 0.02**2 = 0.0004
   - For 10 runs (seed), average var.

3. **Custom Std Experiment**:
   - Init with std=0.1 vs 0.02.
   - Forward random input_ids (B=1,T=5), compute hidden var per dim.
   - Which leads to larger initial activations? (0.1: ~10x larger)

4. **Post-Init in Model**:
   - Use SimpleModel from previous: Add post_init = lambda: self.apply(_init_weights)
   - Create model, call post_init, verify embed_tokens weight var ≈0.0004

## Quiz

1. What dist for embed weight in _init_weights?  
   a) Uniform(0,1) b) Normal(0, initializer_range) c) Zeros

2. True/False: _init_weights zeros padding_idx after normal_.

3. Role of post_init()? (Short: Apply _init_weights recursively)

## Advanced Task

Implement adaptive init for embeddings: std = 1 / sqrt(hidden_size) (Xavier-like). Compare training stability in a toy model (embed + Linear + loss on random data, 10 steps). Plot loss curves for fixed 0.02 vs adaptive. Discuss for Kimi-K2's large D.

Submit code, outputs, and answers.
