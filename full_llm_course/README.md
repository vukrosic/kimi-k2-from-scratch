# Full LLM Course: Building Kimi-K2 from Scratch

This course provides an exhaustive, granular breakdown of implementing the Kimi-K2 model (based on DeepSeek V3 architecture) using PyTorch. Every concept, class, method, and even individual lines of code are dissected into separate lessons. The focus is on **theory ** (mathematical foundations, design choices) and **code ** (PyTorch functions, implementations).

## Course Structure

The course is organized into deeply nested folders:
- Major sections (e.g., 01_embeddings/)
- Subsections for classes/modules (e.g., 01_embeddings/01_embed_tokens/)
- Granular lessons (e.g., 01_embeddings/01_embed_tokens/01_init_method/) with:
  - `teaching.md`: Detailed explanation, theory, code walkthrough.
  - `tasks.md`: Exercises, coding tasks, quizzes.

### Folder Tree Overview

```
full_llm_course/
├── 00_introduction/
│   ├── 01_llm_basics/
│   │   ├── teaching.md
│   │   └── tasks.md
│   └── 02_kimi_k2_overview/
│       ├── teaching.md
│       └── tasks.md
├── 01_embeddings/
│   ├── 01_embed_tokens/
│   │   ├── 01_init/
│   │   │   ├── teaching.md
│   │   │   └── tasks.md
│   │   └── 02_forward/
│   │       ├── teaching.md
│   │       └── tasks.md
│   └── ... (more granular)
├── 02_rms_norm/
│   ├── 01_deepseekv3RMSNorm/
│   │   ├── 01_init/
│   │   │   ├── teaching.md
│   │   │   └── tasks.md
│   │   └── 02_forward/
│   │       ├── teaching.md
│   │       └── tasks.md
│   └── ... (variance calc, normalization steps)
├── 03_rope/
│   ├── 01_standard_rotary_embedding/
│   │   ├── 01_init/
│   │   ├── 02_set_cos_sin_cache/
│   │   └── 03_forward/
│   ├── 02_linear_scaling_rotary_embedding/
│   │   └── ... (differences from standard)
│   ├── 03_dynamic_ntk_scaling_rotary_embedding/
│   │   └── ...
│   └── 04_yarn_rotary_embedding/
│       └── ... (beta_fast/slow, mscale, etc.)
├── 04_attention/
│   ├── 01_deepseekv3_attention/
│   │   ├── 01_init_projections/ (q_proj, kv_proj, o_proj)
│   │   ├── 02_init_rope/
│   │   ├── 03_shape_method/
│   │   ├── 04_forward_qkv/
│   │   ├── 05_apply_rotary_pos_emb/
│   │   ├── 06_attention_computation/
│   │   └── ... (line-by-line for matmul, softmax, etc.)
│   └── 02_flash_attention2/
│       └── ... (unpad, flash_attn_func, etc.)
├── 05_mlp_moe/
│   ├── 01_deepseekv3_mlp/
│   │   ├── 01_init_gate_up_down/
│   │   └── 02_forward/
│   ├── 02_moe_gate/
│   │   ├── 01_init_weight/
│   │   ├── 02_reset_parameters/
│   │   ├── 03_forward_scoring/
│   │   └── 04_topk_selection/
│   └── 03_deepseekv3_moe/
│       ├── 01_init_experts/
│       ├── 02_forward_gate/
│       └── 03_moe_infer/ (distributed, all_to_all)
├── 06_decoder_layer/
│   ├── 01_init_self_attn_mlp/
│   ├── 02_forward_input_layernorm/
│   ├── 03_self_attention/
│   └── 04_post_attention_layernorm_mlp/
├── 07_full_model/
│   ├── 01_deepseekv3_model_init/
│   ├── 02_get_set_embeddings/
│   ├── 03_forward_embed_tokens/
│   ├── 04_decoder_layers_loop/
│   └── 05_final_norm/
├── 08_causal_lm/
│   ├── 01_deepseekv3_for_causallm_init/
│   ├── 02_forward_lm_head/
│   ├── 03_prepare_inputs_for_generation/
│   └── 04_reorder_cache/
└── 09_utilities/
    ├── 01_rotate_half/
    ├── 02_apply_rotary_pos_emb/
    ├── 03_repeat_kv/
    └── ... (all helper functions)
```

## Learning Path
- Start with 00_introduction for prerequisites.
- Progress section-by-section, implementing code snippets in tasks.
- Each lesson builds on the previous, culminating in a full Kimi-K2 implementation.
- Theory: Math derivations, why choices (e.g., RoPE vs absolute pos).
- Code: PyTorch nn.Module, torch functions (e.g., torch.matmul, F.softmax).

## Prerequisites
- Python, PyTorch basics.
- Linear algebra, probability for theory.

## Usage
Run lessons in order. Use code_snippets.py where provided for examples.

This course will expand with hundreds of lessons for ultimate depth.
