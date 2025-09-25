# Tutorial 6: Attention Mechanism in DeepSeek-V3

## What is the Attention Mechanism?

DeepSeek-V3's attention (`DeepseekV3Attention`) is a multi-head self-attention with Rotary Embeddings, but uniquely uses LoRA-style low-rank adaptation for Q/KV projections (q_lora_rank, kv_lora_rank) and splits head dims into rope and non-rope parts (qk_rope_head_dim, qk_nope_head_dim). It supports MQA (shared KV heads) and Flash Attention variant. Less known is the compressed KV projection and custom softmax scaling with rope factors.

This optimizes for efficiency in large MoE models.

### Key Benefits:
- Low-rank projections reduce params.
- Rope integration for positions.
- Flash Attention for speed on long seqs.

## Code Implementation

Core Attention class (key parts; full includes Flash variant):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import ...

class DeepseekV3Attention(nn.Module):
    def __init__(self, config: DeepseekV3Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank  # e.g., low-rank for Q
        self.qk_rope_head_dim = config.qk_rope_head_dim  # Dim for RoPE
        self.kv_lora_rank = config.kv_lora_rank  # Low-rank for KV
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim  # Non-RoPE dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.is_causal = True

        # Q projection: LoRA if rank >0
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.q_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(self.hidden_size, config.q_lora_rank, bias=config.attention_bias)
            self.q_a_layernorm = DeepseekV3RMSNorm(config.q_lora_rank)
            self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)

        # KV projection: Compressed with MQA
        self.kv_a_proj_with_mqa = nn.Linear(self.hidden_size, config.kv_lora_rank + config.qk_rope_head_dim, bias=config.attention_bias)
        self.kv_a_layernorm = DeepseekV3RMSNorm(config.kv_lora_rank)
        self.kv_b_proj = nn.Linear(config.kv_lora_rank, self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim), bias=False)

        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=config.attention_bias)
        self._init_rope()  # Sets rotary_emb
        self.softmax_scale = self.q_head_dim ** (-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)  # Assume defined
                self.softmax_scale = self.softmax_scale * mscale * mscale

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = DeepseekV3RotaryEmbedding(self.qk_rope_head_dim, max_position_embeddings=self.max_position_embeddings, base=self.rope_theta)
        else:
            # Variants: linear, dynamic, yarn
            scaling_type = self.config.rope_scaling["type"]
            if scaling_type == "yarn":
                # ... kwargs for YaRN
                self.rotary_emb = DeepseekV3YarnRotaryEmbedding(self.qk_rope_head_dim, ..., **kwargs)
            # Similar for others

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.v_head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, **kwargs):
        bsz, q_len, _ = hidden_states.size()
        # Q projection
        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # KV projection (compressed)
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)  # MQA: 1 KV head
        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv)).view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        kv_seq_len = value_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        # Combine rope and non-rope
        query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        query_states[:, :, :, :self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_rope_head_dim:] = q_pe
        key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        key_states[:, :, :, :self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_rope_head_dim:] = k_pe
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, {"sin": sin, "cos": cos})

        # Attention computation
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
        if attention_mask is not None:
            attn_weights += attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights if output_attentions else None, past_key_value
```

### Line-by-Line Breakdown

#### Initialization (`__init__` method):
- Stores config: heads, dims, ranks for LoRA.
- Q Proj: If `q_lora_rank` None, direct linear; else, low-rank A (hidden->rank), norm, B (rank->q_dim*heads). Reduces params.
- KV Proj: `kv_a_proj_with_mqa`: Compresses to low-rank + rope_dim. `kv_a_layernorm` on low-rank part. `kv_b_proj` expands to k_nope + v_dim * heads. MQA: KV shared across heads (1 KV head expanded).
- `o_proj`: Output linear.
- `_init_rope()`: Sets rotary embedding, possibly scaled.
- `softmax_scale`: 1/sqrt(q_head_dim), adjusted for rope scaling (mscale).

This uses low-rank for efficiency, splits for partial RoPE.

#### Rope Init (`_init_rope`):
- Basic or scaled (YaRN etc.) rotary emb for rope_dim only.

#### Forward (`forward` method):
- Q: Project (LoRA if applicable), view/transpose to (bs, heads, q_len, q_head_dim), split nope/pe.
- KV: `compressed_kv = kv_a_proj_with_mqa(hidden)`: Low + rope.
- `k_pe.view(..., 1, ...)`: MQA - single KV head.
- `kv = kv_b_proj(norm(compressed_kv))`: Expand low-rank to full k_nope + v.
- Split k_nope, value.
- Get cos/sin for kv_seq_len (incl cache).
- Apply RoPE only to pe parts: `q_pe, k_pe = apply_rotary_pos_emb(...)`.
- Combine: query_states = [q_nope | q_pe], key_states = [k_nope | k_pe].
- Cache update if past.
- `attn_weights = matmul(q, k.T) * scale`: Raw scores.
- Add mask, softmax (fp32), dropout.
- `attn_output = matmul(attn_weights, value)`: Weighted values.
- Reshape, o_proj to hidden.

Flash variant uses flash_attn_func instead of manual matmul.

## Step-by-Step Example Walkthrough

Assume hidden=512, heads=8, q_head_dim=64 (32 nope +32 rope), q_lora_rank=32, kv_lora_rank=32, v_dim=64, input (1,10,512).

1. Q: a_proj(512->32), norm, b_proj(32->8*64=512), view/transpose (1,8,10,64), split (1,8,10,32) nope/pe.
2. KV: kv_a(512->32+32=64), split compressed(32), k_pe(32). k_pe view (1,1,10,32) transpose (1,10,1,32)->(1,1,10,32).
3. kv_b(norm(compressed)->8*(32+64)=8*96=768), view/transpose (1,8,10,96), split k_nope(32), value(64).
4. cos/sin (10,32) for pe.
5. Apply RoPE to q_pe, k_pe.
6. query (1,8,10,64) = nope + pe, key similar (but heads=1 for KV expanded? Wait, code expands kv_b to num_heads).
7. matmul: (1,8,10,64) @ (1,8,10,64).T -> (1,8,10,10), *scale.
8. Softmax, matmul value (1,8,10,64) -> (1,8,10,64).
9. Transpose/reshape (1,10,512), o_proj to (1,10,512).

Low-rank saves params: Q ~ hidden*rank + rank*q_dim vs hidden*q_dim.

## Why Use This Attention in DeepSeek-V3?

DeepSeek-V3's attention optimizes for scale: LoRA ranks cut params (e.g., 32<<512), MQA reduces KV cache, partial RoPE saves compute, scaling adjusts softmax for long contexts. Crucial for 128K+ tokens in MoE.

These complete the 3 additional tutorials.
