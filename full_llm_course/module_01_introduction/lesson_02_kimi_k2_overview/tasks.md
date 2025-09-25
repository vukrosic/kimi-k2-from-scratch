# Lesson 1.2: Tasks - Kimi-K2 Architecture Overview

## Theory Exercises

1. **Component Breakdown**: List and explain the role of each major component in Kimi-K2 (embeddings, attention, MoE/MLP, norms, LM head). Why use RMSNorm instead of LayerNorm? (Hint: Training stability, compute.)

2. **Innovations Deep Dive**: 
   - Explain RoPE: How does rotation encode relative positions? Contrast with absolute positional embeddings.
   - Describe GQA: Why compress KV with low-rank? Benefits for memory/inference.
   - MoE vs Dense: How does routing to top-k experts reduce compute? What is load balancing in MoE?

3. **Mathematical Task**: Write the attention formula with scaling and causal mask. For a sequence of length 4, sketch the mask matrix (1s below diagonal, -inf above).

## Code Tasks

1. **Model Skeleton Implementation**:
   - Extend the `SimpleKimiK2` class from teaching.md. Add a placeholder `DecoderLayer` class with pre-norm, attention stub, MLP stub, residuals.
   - Forward pass: Embed → loop layers → norm → lm_head.
   - Test: `input_ids = torch.tensor([[1,2,3]])`, print logits.shape (should be (1,3,vocab_size)).

   ```python
   # Starter code from teaching
   class RMSNorm(nn.Module):  # Stub
       def __init__(self, dim): super().__init__(); self.scale = nn.Parameter(torch.ones(dim))
       def forward(self, x): return x * self.scale  # Simplified

   class DecoderLayer(nn.Module):  # Your impl
       def __init__(self, config): ...
       def forward(self, hidden): ...

   # Complete SimpleKimiK2 and test
   ```

2. **Diagram Reproduction**:
   - In code or drawing tool, recreate the ASCII diagram. Add labels for residual connections and where RoPE is applied.

3. **Config Customization**:
   - Create a Config class with params from DeepSeek V3 (e.g., hidden_size=4096, num_layers=30, num_heads=32, num_kv_heads=8 for GQA).
   - Instantiate model and compute param count: `sum(p.numel() for p in model.parameters())`.

## Quiz

1. In the forward flow, what happens after embeddings?  
   a) LM Head b) Decoder layers c) Final norm

2. True/False: MoE activates all experts for every token.

3. What is the purpose of the softmax scale in attention? (Short answer: \( 1/\sqrt{d_k} \))

## Advanced Task

Research DeepSeek V3 paper: Summarize one unique feature (e.g., MLA - Multi-head Latent Attention) and how it differs from standard GQA. Propose a simple code stub for it.

Submit code, explanations, and quiz answers.
