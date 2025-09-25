# Lesson 2.2: Embeddings - Get and Set Input Embeddings

## Theory

In HuggingFace Transformers, models provide methods to access and replace embedding layers for flexibility. This allows users to customize embeddings (e.g., resize vocab, use different initializations) without rewriting the core model. In Kimi-K2, these are defined in DeepseekV3Model and inherited by DeepseekV3ForCausalLM.

### Key Theory
- **Modularity**: get_input_embeddings returns the embedding module; set_input_embeddings replaces it, updating shapes if needed (e.g., new vocab_size).
- **Use Cases**: 
  - Resizing model for new tokenizer (add/remove tokens).
  - Injecting pre-trained embeds from another model.
  - Fine-tuning with frozen or custom embeds.
- **Shape Consistency**: New embeds must match hidden_size; vocab_size can change, but LM head may need adjustment (tied weights complicate this).
- **_init_weights Integration**: After setting, call model.post_init() to re-initialize weights.

These methods ensure the embedding layer is pluggable, promoting reusability.

## Code Walkthrough

From DeepseekV3Model:

```python
def get_input_embeddings(self):
    return self.embed_tokens

def set_input_embeddings(self, value):
    self.embed_tokens = value
```

- get_input_embeddings: Simple return of self.embed_tokens (nn.Embedding instance).
- set_input_embeddings: Direct assignment. No auto-resizing; user ensures compatibility.
- In DeepseekV3ForCausalLM: Delegates to self.model.get_input_embeddings() / set_input_embeddings(value), maintaining consistency.
- _init_weights (from PreTrainedModel): Called in post_init(), applies normal_(0, config.initializer_range) to embed_tokens.weight if Embedding.

Example Usage:
```python
# Get
embeds = model.get_input_embeddings()
print(type(embeds))  # <class 'torch.nn.modules.embedding.Embedding'>

# Set new (e.g., larger vocab)
new_embeds = nn.Embedding(new_vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
model.set_input_embeddings(new_embeds)
model.post_init()  # Re-init weights
```

If tied, setting affects LM head implicitly.

## PyTorch Functions
- Direct attribute access; no special torch ops.
- post_init() triggers _init_weights for normalization.

## Why This Matters
Enables model adaptation (e.g., multilingual). In production, resize_token_embeddings method builds on this for full vocab resize.

Next: Embeddings in _init_weights (initialization details).
