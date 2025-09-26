Must be best course on AI research and LLMs ever created.
- Docs in chinese (and serbian?) and subtitles in chinese (and serbian?).
Clear roadmap for viewers.
- self study
1 research papers with at least 20 pages.

Specific experiments to implement:
- Learning Rate Search: Test lr=[1e-4, 5e-4, 1e-3, 2e-3] on same config
- DeepseekV3RMSNorm vs LayerNorm: Compare lines 103-108 vs standard LayerNorm
- MoE routing: Test topk_method=["noaux_tc", "aux_tc"] in MoEGate class (lines 392-415)
- Attention variants: Compare DeepseekV3Attention (line 627) vs DeepseekV3FlashAttention2 (line 860)
- RoPE scaling: Test rope_theta=[10000, 50000, 100000] in DeepseekV3Attention (line 646)
- Expert capacity: Test n_routed_experts=[4, 8, 16, 32] in DeepseekV3MoE (line 475)
- Sequence length: Test max_position_embeddings=[512, 1024, 2048] in config
- Load balancing: Compare routed_scaling_factor=[0.1, 0.5, 1.0] in MoEGate (line 398)