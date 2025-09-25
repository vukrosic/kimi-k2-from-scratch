# Lesson 1.2: Overview of Kimi-K2 Architecture

## Theory

Kimi-K2 is an advanced LLM based on the DeepSeek V3 architecture, a decoder-only transformer optimized for efficiency and performance. It builds on GPT-like models but incorporates innovations like Mixture-of-Experts (MoE), Rotary Position Embeddings (RoPE), and compressed Grouped-Query Attention (GQA).

### High-Level Components
1. **Input Embeddings**: Token IDs → dense vectors (vocab_size ~100k, hidden_size 4096+).
2. **Transformer Layers** (stacked, e.g., 30+ layers):
   - **RMSNorm Pre-Norm**: Stabilizes training (vs LayerNorm).
   - **Self-Attention**: Causal GQA with RoPE for positions. Uses LoRA-like compression for KV.
   - **Feed-Forward**: Alternates dense MLP and MoE layers for sparse activation.
   - **RMSNorm Post-Norm**.
3. **Final RMSNorm**: Normalizes output hidden states.
4. **LM Head**: Linear projection to vocab logits for next-token prediction.

### Key Innovations
- **RoPE**: Relative positional encoding via rotations; supports long contexts with scaling (linear, dynamic NTK, YaRN).
- **GQA with Compression**: Reduces KV heads via low-rank adaptation, saving memory.
- **MoE**: Routes tokens to top-k experts per layer; efficient scaling (activate subset of params).
- **Flash Attention**: Optimized attention for speed/memory (optional backend).

### Architecture Diagram (ASCII)
```
Input Tokens --> Embeddings (nn.Embedding)
                |
                v
[Transformer Decoder Layers] x N
  |--> RMSNorm (input)
  |     |
  |     v
  |--> Self-Attention (GQA + RoPE)
  |     |
  |     v
  |--> Residual + 
  |     |
  |     v
  |--> RMSNorm (post-attn)
  |     |
  |     v
  |--> MoE/MLP (routed experts or dense FFN)
  |     |
  |     v
  |--> Residual
  |
  v
Final RMSNorm
  |
  v
LM Head (Linear) --> Logits --> Softmax --> Next Token
```

### Mathematical Overview
- Attention: \( \text{Attention}(Q, K, V) = \text{softmax}(QK^T / \sqrt{d}) V \), causal mask.
- MoE Routing: Score tokens to experts, select top-k, weighted sum.
- Loss: Cross-entropy on shifted logits.

This structure enables Kimi-K2 to handle long contexts (128k+ tokens) efficiently.

## Code Walkthrough

High-level model skeleton:

```python
import torch.nn as nn

class SimpleKimiK2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids):
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        return logits

# Config example
class Config:
    vocab_size = 100000
    hidden_size = 4096
    num_layers = 30
    # ... other params

model = SimpleKimiK2(Config())
```

- Layers alternate MoE/MLP based on config.
- Full impl in later sections.

## Why This Matters
Grasping the overview helps navigate granular lessons. Kimi-K2's MoE makes it sparse yet powerful.

Next section: Embeddings in detail.
