# Lesson 2.2: Tasks - Get and Set Input Embeddings

## Theory Exercises

1. **Modularity Benefits**: Why provide get/set methods? How do they support HF ecosystem (e.g., AutoModel, tokenizers)? Discuss risks of mismatched shapes.

2. **Use Cases Deep Dive**:
   - Resizing: How to add 100 new tokens? What happens to LM head if tied?
   - Custom Embeds: Example for vision-language model (e.g., CLIP embeds as inputs_embeds).
   - Freezing: Set new embeds, set requires_grad=False for fine-tuning.

3. **Tied Weights Complication**: In Kimi-K2, lm_head.weight tied to embed_tokens.weight. If you set new embeds with different vocab_size, what breaks? How to handle (resize lm_head separately)?

4. **Post-Init Reinitialization**: Why call post_init() after set? What does _init_weights do for Embedding (normal dist)?

## Code Tasks

1. **Basic Get/Set**:
   - Assume model = DeepseekV3Model(config) (stub if needed).
   - old_embeds = model.get_input_embeddings()
   - Print old_embeds.embedding_dim == config.hidden_size
   - new_embeds = nn.Embedding(config.vocab_size + 100, config.hidden_size, padding_idx=config.pad_token_id)
   - model.set_input_embeddings(new_embeds)
   - Verify model.get_input_embeddings() is new_embeds

   ```python
   # Pseudo-code; use simple nn.Module for model if no full DeepseekV3Model
   class SimpleModel(nn.Module):
       def __init__(self, config):
           super().__init__()
           self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
       def get_input_embeddings(self): return self.embed_tokens
       def set_input_embeddings(self, value): self.embed_tokens = value

   class Config: vocab_size=100; hidden_size=64  # Small for test
   config = Config()
   model = SimpleModel(config)
   # Your code here
   ```

2. **Resize Simulation**:
   - Start with V=100, create model.
   - Get old_embeds, create new with V=150 (copy old weights, random for new).
   - new_weight = torch.cat([old_embeds.weight, torch.randn(50, config.hidden_size)], dim=0)
   - new_embeds = nn.Embedding(150, config.hidden_size, _weight=new_weight)
   - Set and test forward with input_ids up to 149.

3. **Tied Weights Effect**:
   - In CausalLM model stub: self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False); self.lm_head.weight = self.embed_tokens.weight  # Tie
   - Set new_embeds (same V), verify lm_head.weight is now new_embeds.weight (shared).
   - Forward: embeds = model.embed_tokens(input_ids); logits = model.lm_head(embeds); print(logits.shape)

4. **Post-Init Check**:
   - Before set: Save norm of embed_tokens.weight.mean()
   - Set new random embeds.
   - model.post_init()  # Triggers _init_weights: normal_(0, 0.02)
   - Verify weights changed (norm ~0 after mean=0).

## Quiz

1. What does get_input_embeddings return?  
   a) weight tensor b) nn.Embedding module c) config dict

2. True/False: set_input_embeddings auto-resizes lm_head if tied.

3. When to call post_init() after set? (Short: To re-initialize weights)

## Advanced Task

Implement resize_token_embeddings method: Get old, create new with new_vocab_size, copy old weights, random init new rows/cols if needed, set, post_init. Test with V=100 to 120, input_ids with new token 110. Discuss handling tied LM head (resize separately if vocab changes).

Submit code, outputs, and answers.
