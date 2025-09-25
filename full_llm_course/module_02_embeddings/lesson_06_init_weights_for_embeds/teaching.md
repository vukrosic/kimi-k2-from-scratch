# Lesson 2.3: Embeddings - Initialization in _init_weights

## Theory

Model initialization is critical for stable training in deep networks. Random weights with appropriate variance prevent vanishing/exploding gradients. In HuggingFace, _init_weights (from PreTrainedModel) standardizes this across modules, including embeddings. For Kimi-K2, it's called in post_init() after __init__.

### Key Theory
- **Initialization Goals**: Keep activations/grads with unit variance across layers (He/Kaiming for ReLU, Xavier for tanh/sigmoid). For embeddings, normal(0, std) where std = config.initializer_range (typically 0.02).
- **For Embeddings**: As nn.Embedding (like Linear without bias), weight ~ N(0, std^2). Ensures initial embeds have similar magnitudes, aiding early training.
- **Padding Handling**: If padding_idx, weight[padding_idx] = 0 (set in nn.Embedding), then _init_weights skips or zeros it.
- **Global Application**: Recursive on all submodules; for Linear/Embedding: weight.normal_(0, std), bias.zero_ if present.

Why 0.02? Empirical for transformers; small to avoid large initial logits.

Mathematical: Weight W ~ N(0, σ^2), σ = initializer_range. For fan-in init, σ = 1/sqrt(D_in), but fixed for simplicity.

## Code Walkthrough

From PreTrainedModel._init_weights:

```python
def _init_weights(self, module):
    std = self.config.initializer_range
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
```

- For nn.Embedding: normal_(0, std) on weight.
- Padding: Explicitly zero if padding_idx (though nn.Embedding does this on creation).
- Called via self.post_init() at end of __init__: self._init_weights(self) (recursive).

In DeepseekV3Model.__init__: After setting embed_tokens, self.post_init() applies this.

Example:
```python
# After creation
embed_tokens.weight.data.normal_(0, 0.02)
# Verify: torch.mean(embed_tokens.weight.data**2) ≈ 0.02**2 = 0.0004
```

## PyTorch Functions
- tensor.normal_(mean, std): In-place normal dist.
- Recursive traversal via module.apply(_init_weights).

## Why This Matters
Bad init leads to slow convergence or NaNs. Fixed std works well for transformers; custom (e.g., small_init for MoE) possible.

Embeddings section complete. Next: RMSNorm.
