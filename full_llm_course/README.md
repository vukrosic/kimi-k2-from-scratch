# Full LLM Course: Building Kimi-K2 from Scratch

This course provides an exhaustive, granular breakdown of implementing the Kimi-K2 model (based on DeepSeek V3 architecture) using PyTorch. Every concept, class, method, and even individual lines of code are dissected into separate lessons. The focus is on **theory ** (mathematical foundations, design choices) and **code ** (PyTorch functions, implementations).

## Course Structure

The course is organized into modules with individual lessons:
- **Modules**: Major topic areas (e.g., module_01_introduction/)
- **Lessons**: Individual learning units (e.g., lesson_01_llm_basics/) with:
  - `teaching.md`: Detailed explanation, theory, code walkthrough.
  - `tasks.md`: Exercises, coding tasks, quizzes.

### Folder Tree Overview

```
full_llm_course/
├── module_01_introduction/
│   ├── lesson_01_llm_basics/
│   │   ├── teaching.md
│   │   └── tasks.md
│   └── lesson_02_kimi_k2_overview/
│       ├── teaching.md
│       └── tasks.md
├── module_02_embeddings/
│   ├── lesson_03_embed_tokens_init/
│   │   ├── teaching.md
│   │   └── tasks.md
│   ├── lesson_04_embed_tokens_forward/
│   │   ├── teaching.md
│   │   └── tasks.md
│   ├── lesson_05_get_set_embeddings/
│   │   ├── teaching.md
│   │   └── tasks.md
│   └── lesson_06_init_weights_for_embeds/
│       ├── teaching.md
│       └── tasks.md
├── module_03_rms_norm/
│   ├── lesson_07_deepseekv3_rms_norm_init/
│   │   ├── teaching.md
│   │   └── tasks.md
│   └── lesson_08_deepseekv3_rms_norm_forward/
│       ├── teaching.md
│       └── tasks.md
├── module_04_rotary_embedding/
│   ├── lesson_09_standard_rotary_init/
│   ├── lesson_10_standard_rotary_cache/
│   ├── lesson_11_standard_rotary_forward/
│   ├── lesson_12_linear_scaling_init/
│   ├── lesson_13_linear_scaling_cache/
│   ├── lesson_14_linear_scaling_forward/
│   ├── lesson_15_dynamic_ntk_init/
│   ├── lesson_16_dynamic_ntk_cache/
│   ├── lesson_17_dynamic_ntk_forward/
│   └── lesson_18_rotary_embedding_class/
└── ... (more modules to be added)
```

## Learning Path
- Start with module_01_introduction for prerequisites.
- Progress module-by-module, implementing code snippets in tasks.
- Each lesson builds on the previous, culminating in a full Kimi-K2 implementation.
- Theory: Math derivations, why choices (e.g., RoPE vs absolute pos).
- Code: PyTorch nn.Module, torch functions (e.g., torch.matmul, F.softmax).

## Prerequisites
- Python, PyTorch basics.
- Linear algebra, probability for theory.

## Usage
Run lessons in order. Use code_snippets.py where provided for examples.

This course will expand with hundreds of lessons for ultimate depth.
