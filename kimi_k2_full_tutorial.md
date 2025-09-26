# Step-by-Step Tutorial: Modeling Kimi K2 from Scratch (Based on DeepSeek V3)

This comprehensive tutorial guides you through building the Kimi K2 model line-by-line, inspired by the DeepSeek V3 architecture from Hugging Face Transformers. Kimi K2 is a large language model using a Mixture of Experts (MoE) decoder with Rotary Position Embeddings (RoPE), RMSNorm, and efficient attention. We'll break it into multiple lessons, each focusing on a component with code explanations, math insights, and examples.

**Prerequisites:** Python, PyTorch, basic linear algebra. Install: `pip install torch transformers`. All code is from `kimi_k2_modeling.py`—we'll dissect it progressively.

**Note:** This is educational; full training requires massive data/GPUs. Focus on understanding the forward pass and architecture.

## Lesson 1: Introduction to Kimi K2 Architecture

Kimi K2 (based on DeepSeek V3) is a causal decoder-only Transformer with MoE for efficiency. Key features:
- **Embeddings:** Token embeddings + RoPE for positions.
- **Layers:** Stacked decoder layers with RMSNorm pre-attn/pre-MLP, self-attention (GQA/MQA support via projections), and MoE/MLP.
- **MoE:** Routed experts (top-k selection) + shared experts for sparsity.
- **Attention:** RoPE with scaling (linear/dynamic/YARN), optional Flash Attention 2.
- **Output:** RMSNorm + LM head for next-token prediction.
- **Config:** `DeepseekV3Config` (vocab_size, hidden_size, num_layers, etc.).

### Imports and Setup
The file starts with essentials:

```python
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
# ... (more imports for masks, outputs, utils)
from .configuration_deepseek import DeepseekV3Config
import torch.distributed as dist
import numpy as np

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    # ...
```

**Line-by-Line:**
- Standard libs: `math`, `warnings`, `typing` for type hints.
- Torch: Core (`nn`, `F`), checkpointing for memory, losses for training.
- Transformers: Activations (`ACT2FN`), cache for generation, masks/outputs/utils for HF integration.
- Config: Loads `DeepseekV3Config` (e.g., `hidden_size=4096`, `num_layers=30`, `n_routed_experts=64`).
- Distributed: For multi-GPU MoE.
- Optional Flash Attention: For faster attn if installed.

### Utility Functions
Early utils like `_get_unpad_data` for Flash Attn padding.

```python
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen_in_batch
```

**Breakdown:** Computes cumulative seq lens and indices for unpadded inputs in variable-length batches. Used in Flash Attn to handle padding efficiently.

**Example:** For mask `[[1,1,0], [1,0,0]]`, seqlens=[2,1], cu_seqlens=[0,2,3], indices=[0,1,3].

This sets up for efficient batching. Next: Normalization.

## Lesson 2: Normalization - DeepseekV3RMSNorm

RMSNorm normalizes activations for stable training, used pre-attn and pre-MLP.

### Code
```python
class DeepseekV3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
```

**Line-by-Line Breakdown:**
- `__init__`: `weight` learnable scale (ones init), `eps` for stability.
- `forward`:
  - Save dtype, cast to fp32 for precision.
  - `variance = ... .pow(2).mean(-1, keepdim=True)`: RMS² along features (dim=-1), shape (B, S, 1).
  - `rsqrt(variance + eps)`: 1/√RMS for unit norm.
  - Scale by `weight`, restore dtype.

**Math:** For vector x, RMSNorm(x) = x / √(mean(x²) + ε) * w. No mean subtract (faster than LayerNorm).

**Example Usage:**
```python
hidden_size = 512
x = torch.randn(2, 10, hidden_size)
norm = DeepseekV3RMSNorm(hidden_size)
y = norm(x)
print(torch.sqrt((y ** 2).mean(-1)).mean())  # ~1.0 (normalized)
```

RMS after ≈1, weights learn adjustments. In model: `self.input_layernorm`, `self.post_attention_layernorm`.

## Lesson 3: Positional Embeddings - Rotary Embeddings

RoPE encodes positions via rotations, relative and extrapolation-friendly.

### Base RotaryEmbedding
```python
class DeepseekV3RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype())
        self.max_seq_len_cached = None

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(t.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return self.cos_cached[:seq_len].to(dtype=x.dtype), self.sin_cached[:seq_len].to(dtype=x.dtype)
```

**Breakdown:**
- `__init__`: `inv_freq` = 1 / base^(i/dim) for pairs (even dims).
- `_set_cos_sin_cache`: Precompute cos/sin for positions 0 to seq_len-1. `freqs = t outer inv_freq`, cat for sin/cos pairs.
- `forward`: Cache/return sliced cos/sin for seq_len.

**Math:** For position m, dim 2i/2i+1: Rotate by θ_m,i = m / base^(i/dim). Q/K rotated: q' = q cos - q_sin sin, etc.

### Variants
- **LinearScaling:** Scales t by factor for longer contexts.
  ```python
  t = t / self.scaling_factor  # In _set_cos_sin_cache
  ```
- **DynamicNTK:** Adjusts base for NTK extrapolation.
  ```python
  if seq_len > max_pos: base = ... ** (dim/(dim-2))
  ```
- **YARN:** Hybrid freqs with ramp mask for low/high freqs, mscale.
  Complex: `inv_freq_mask` blends inter/extra freqs, mscale = log(scale).

**Apply Function:**
```python
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    # Reshape q/k to pairs, rotate: (q * cos) + (rotate_half(q) * sin)
    # rotate_half: cat(-x2, x1) for half dims
    return q_embed, k_embed
```

**Example:** For q [bs, heads, seq, head_dim], apply rotation per position.

In attn: `cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)`

## Lesson 4: Standard MLP - DeepseekV3MLP

Swiglu-style FFN: gate/up/down projections.

```python
class DeepseekV3MLP(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]  # e.g., silu

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
```

**Breakdown:**
- Projs: Gate/up to intermediate (e.g., 11008), down to hidden (4096). No bias for efficiency.
- `forward`: SwiGLU: act(gate(x)) * up(x), then down. Parallel gates for non-linearity.

**Math:** FFN(x) = down( Swi( gate(x) ) * up(x) ), where Swi = silu(x) = x * sigmoid(x).

**Usage:** In layers without MoE.

## Lesson 5: MoE Gating - MoEGate

Selects top-k experts per token using sigmoid scores.

```python
class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok  # e.g., 6
        self.n_routed_experts = config.n_routed_experts  # e.g., 64
        # ... other params: routed_scaling_factor, scoring_func='sigmoid', topk_method='noaux_tc'
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        if self.topk_method == "noaux_tc":
            self.e_score_correction_bias = nn.Parameter(torch.empty((self.n_routed_experts)))
        self.reset_parameters()

    def reset_parameters(self):
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32), None)
        if self.scoring_func == "sigmoid":
            scores = logits.sigmoid()
        # ... (top-k selection logic)
        # For 'noaux_tc': Add bias, group topk, mask, select top_k indices/weights
        # Norm if top_k>1 and norm_topk_prob
        topk_weight = topk_weight * self.routed_scaling_factor
        return topk_idx, topk_weight
```

**Breakdown:**
- `__init__`: Weight (experts x hidden) for logits, optional bias for correction.
- `reset`: Kaiming init for stability.
- `forward`:
  - Flatten to (B*S, H).
  - Logits = linear (scores pre-sigmoid).
  - Scores = sigmoid(logits) [0,1].
  - Top-k: For 'noaux_tc' (inference): Group experts, select top groups, mask scores, topk on masked.
  - Weights: Gather scores for selected, optional norm to sum=1, scale.

**Math:** Score_e = sigmoid( W_e · x ), select argtopk_k( scores ), weights = softmax(scores_selected) * factor.

Handles load balancing via groups/topk_group.

## Lesson 6: Mixture of Experts Layer - DeepseekV3MoE

Routes to experts + shared MLP.

```python
class DeepseekV3MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ... num_experts_per_tok
        if hasattr(config, "ep_size") and config.ep_size > 1:  # Distributed
            # Experts per rank, only create local experts
            self.experts = nn.ModuleList([DeepseekV3MLP(...) if local else None for i in range(n_experts)])
        else:
            self.experts = nn.ModuleList([DeepseekV3MLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.n_routed_experts)])
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            self.shared_experts = DeepseekV3MLP(config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts)

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if not self.training:
            y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
        # ... (training: all_reduce or similar, omitted)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        # Count tokens per expert, sort/permute x
        # If distributed: all_to_all to gather tokens to owning ranks
        # Compute expert outputs on local tokens
        # all_to_all back, gather, weighted sum
        return final_out
```

**Breakdown:**
- `__init__`: Create experts (MLPs) locally if distributed (ep_size=world_size). Gate for routing. Optional shared (dense) experts.
- `forward`: Get topk from gate, flatten, infer (no_grad for speed), add shared if any.
- `moe_infer`: 
  - Counts: Scatter-add 1s to expert bins.
  - Permute x by expert order.
  - Distributed: all_to_all tokens to ranks, compute local, all_to_all outputs back.
  - Weighted sum: y = sum( w_e * expert_e(x) ) for selected e.

**Efficiency:** Only top-k experts active per token (~8-16% params active).

**Example:** For 64 experts, top_k=6, routes ~6/64=9% compute.

## Lesson 7: Attention Mechanism - DeepseekV3Attention

Multi-head self-attention with RoPE and LoRA-like projections.

```python
class DeepseekV3Attention(nn.Module):
    def __init__(self, config: DeepseekV3Config, layer_idx: Optional[int] = None):
        super().__init__()
        # ... config params: num_heads, q_lora_rank (None or low-rank), qk_rope_head_dim, etc.
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim  # e.g., 128 total
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.q_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(self.hidden_size, config.q_lora_rank, bias=config.attention_bias)
            self.q_a_layernorm = DeepseekV3RMSNorm(config.q_lora_rank)
            self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)
        # KV: Compressed via LoRA + rope dim
        self.kv_a_proj_with_mqa = nn.Linear(self.hidden_size, config.kv_lora_rank + config.qk_rope_head_dim, bias=...)
        self.kv_a_layernorm = DeepseekV3RMSNorm(config.kv_lora_rank)
        self.kv_b_proj = nn.Linear(config.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=...)
        self._init_rope()  # Sets rotary_emb based on rope_scaling
        self.softmax_scale = self.q_head_dim ** (-0.5)  # 1/sqrt(d)

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = DeepseekV3RotaryEmbedding(self.qk_rope_head_dim, ...)
        else:
            # linear/dynamic/yarn based on type
            pass

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, ...):
        bsz, q_len, _ = hidden_states.size()
        # Q proj (LoRA if rank>0: low-rank approx)
        q = self.q_proj(hidden_states) if no LoRA else self.q_b_proj(norm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        # KV: compressed_kv = a_proj (low + rope), split, b_proj to k_nope + v
        # RoPE: cos/sin from rotary_emb(kv_seq_len), apply to q_pe, k_pe
        # Concat: query_states = [q_nope, q_pe_rot], key_states = [k_nope, k_pe_rot]
        # Cache update if past
        # Scores: matmul(q, k.T) * scale
        # Mask + softmax (fp32) + dropout
        # Output: matmul(attn, v).transpose -> o_proj
        return attn_output, attn_weights, past_key_value
```

**Breakdown:**
- Projs: Q full or LoRA (A: hidden->rank, norm, B: rank->full). KV compressed (low-rank + rope dim direct).
- RoPE: Apply only to rope dims, concat with non-rotated.
- Scores: Causal mask via attention_mask (4D).
- Scale: Adjusted for rope mscale if YARN.

**Math:** Attn(Q,K,V) = softmax( (Q K^T)/√d + mask ) V

**Shape:** Q/K/V (B, H, S, D_h), output (B, S, hidden).

## Lesson 8: Flash Attention Variant - DeepseekV3FlashAttention2

Optimized attn using Flash Attn 2 for speed/memory.

Inherits from Attention, overrides forward:

- Same QKV proj/RoPE.
- Transpose to (B, S, H, D) for Flash.
- If mask: Unpad inputs, call `flash_attn_varlen_func`, pad output.
- Else: `flash_attn_func` with causal.
- Pad V if q_head_dim != v_head_dim.
- Rest same: o_proj.

**Breakdown:** `_flash_attention_forward`: Handles padding via unpad/pad, causal flag. Top-left mask fix for old Flash.

**Benefits:** O(S) memory vs O(S²), faster on long seq.

**ATTENTION_CLASSES:** {'eager': Attention, 'flash_attention_2': FlashAttention2}

## Lesson 9: Decoder Layer - DeepseekV3DecoderLayer

Assembles attn + FFN (MoE or MLP) + norms.

```python
class DeepseekV3DecoderLayer(nn.Module):
    def __init__(self, config: DeepseekV3Config, layer_idx: int):
        super().__init__()
        self.self_attn = ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)
        self.mlp = DeepseekV3MoE(config) if (layer_idx >= first_k_dense_replace and layer_idx % moe_layer_freq == 0) else DeepseekV3MLP(config)
        self.input_layernorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attn
        hidden_states, self_attn_weights, present_key_value = self.self_attn(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, **kwargs)
        hidden_states = residual + hidden_states
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions: outputs += (self_attn_weights,)
        if use_cache: outputs += (present_key_value,)
        return outputs
```

**Breakdown:**
- Init: Attn (eager/flash), MLP/MoE based on layer (MoE every freq after initial dense).
- Forward: Pre-norm (RMS) + attn + residual, pre-norm + FFN + residual. Gated residuals.

**MoE Placement:** Sparse in later layers for capacity.

## Lesson 10: Full Model Construction - DeepseekV3Model and CausalLM

### Base Model
```python
class DeepseekV3Model(DeepseekV3PreTrainedModel):
    def __init__(self, config: DeepseekV3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([DeepseekV3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()  # Weights init

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        # Handle inputs: embed if ids, pos_ids if None (arange + past_len)
        # Mask: 4D causal if not flash, 2D if flash
        hidden_states = inputs_embeds or self.embed_tokens(input_ids)
        # Loop layers: normed attn/FFN residuals, collect hidden/attns/cache
        hidden_states = self.norm(hidden_states)
        # Return BaseModelOutputWithPast
```

**Breakdown:**
- Embed: Lookup (B,S) -> (B,S,H).
- Pos: Cumulative from mask or arange.
- Mask: Causal + padding.
- Layers: Stack N=30, each decoder_layer.
- Final RMSNorm.

### Causal LM
```python
class DeepseekV3ForCausalLM(DeepseekV3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = DeepseekV3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(self, input_ids=None, ..., labels=None, ...):
        outputs = self.model(input_ids, attention_mask, ..., return_dict=True)
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states).float()
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = CrossEntropyLoss()(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values, ...)
```

**Breakdown:**
- Model: Gets hidden.
- Head: Linear to vocab, tied weights possible.
- Loss: Shifted CE (predict next).
- Generation: prepare_inputs_for_generation handles cache, pos.

**Init Weights:** Normal(0, init_range=0.02) for linears/embed, zero bias.

## Lesson 11: Utilities and Inference

- **Caching:** DynamicCache for KV, RoPE sin/cos.
- **Scaling:** Rope_scaling in config (factor, type).
- **Distributed MoE:** all_to_all for expert parallelism.
- **Generation:** prepare_for_generation crops inputs for cache, reorders for beam.

**Full Example:**
```python
from transformers import AutoTokenizer, DeepseekV3ForCausalLM
model = DeepseekV3ForCausalLM.from_pretrained("path/to/kimi_k2")
tokenizer = AutoTokenizer.from_pretrained("path")
inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

This completes the line-by-line build. Experiment: Modify RoPE base, add print in forward. Next: Fine-tuning or eval.

**References:** DeepSeek V3 paper, HF Transformers docs.
