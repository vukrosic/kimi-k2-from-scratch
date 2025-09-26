# Experiment Design: Specific Experiments with Code References

## 🎯 Experiment Overview

This document provides specific, actionable experiments with exact code references from both `llm.py` and `kimi_k2_modeling.py`.

## 📋 Experiment 1: Attention Mechanism Comparison

### Objective
Compare standard attention vs Flash Attention 2 performance and efficiency.

### Code References
- **Baseline**: `MultiHeadAttention` in `llm.py` (lines 200-236)
- **Variant**: `DeepseekV3FlashAttention2` in `kimi_k2_modeling.py` (lines 860-946)

### Implementation Steps
1. **Extract Attention Classes**:
   ```python
   # From llm.py lines 200-236
   class MultiHeadAttention(nn.Module):
       def __init__(self, d_model, n_heads, dropout=0.1):
           # ... existing implementation
   
   # From kimi_k2_modeling.py lines 860-946
   class DeepseekV3FlashAttention2(DeepseekV3Attention):
       def forward(self, hidden_states, attention_mask=None, position_ids=None):
           # ... Flash Attention 2 implementation
   ```

2. **Create Experiment Config**:
   ```python
   attention_experiments = {
       'baseline': {'attention_type': 'standard', 'class': 'MultiHeadAttention'},
       'flash_attn': {'attention_type': 'flash', 'class': 'DeepseekV3FlashAttention2'}
   }
   ```

3. **Test Parameters**:
   - Sequence lengths: [512, 1024, 2048]
   - Batch sizes: [8, 16, 32]
   - Training steps: 1000 (for quick comparison)

### Metrics to Measure
- **Performance**: Perplexity on validation set
- **Efficiency**: Training time per step
- **Memory**: Peak GPU memory usage
- **Throughput**: Tokens processed per second

### Expected Results
- Flash Attention 2 should be faster for longer sequences
- Memory usage should be lower with Flash Attention 2
- Performance should be equivalent or better

## 📋 Experiment 2: Normalization Technique Study

### Objective
Compare RMSNorm vs LayerNorm for training stability and performance.

### Code References
- **Baseline**: Standard `nn.LayerNorm`
- **Variant**: `DeepseekV3RMSNorm` in `kimi_k2_modeling.py` (lines 94-108)

### Implementation Steps
1. **Extract Normalization Classes**:
   ```python
   # From kimi_k2_modeling.py lines 94-108
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

2. **Create Experiment Config**:
   ```python
   normalization_experiments = {
       'baseline': {'norm_type': 'layernorm', 'class': 'nn.LayerNorm'},
       'rmsnorm': {'norm_type': 'rmsnorm', 'class': 'DeepseekV3RMSNorm'}
   }
   ```

3. **Test Parameters**:
   - Learning rates: [1e-4, 5e-4, 1e-3, 2e-3]
   - Training steps: 2000
   - Epsilon values: [1e-6, 1e-8] (for RMSNorm)

### Metrics to Measure
- **Stability**: Training loss variance
- **Convergence**: Steps to reach target loss
- **Performance**: Final validation perplexity
- **Gradient**: Gradient norm statistics

### Expected Results
- RMSNorm should provide more stable training
- Faster convergence with RMSNorm
- Better performance on validation set

## 📋 Experiment 3: MoE Routing Strategy Study

### Objective
Compare different expert selection strategies and their impact on performance.

### Code References
- **Baseline**: Current MoE implementation in `llm.py`
- **Variant**: `MoEGate` in `kimi_k2_modeling.py` (lines 392-415)

### Implementation Steps
1. **Extract MoE Classes**:
   ```python
   # From kimi_k2_modeling.py lines 392-415
   class MoEGate(nn.Module):
       def __init__(self, config):
           super().__init__()
           self.config = config
           self.top_k = config.num_experts_per_tok
           self.n_routed_experts = config.n_routed_experts
           self.routed_scaling_factor = config.routed_scaling_factor
           self.scoring_func = config.scoring_func
           self.seq_aux = config.seq_aux
           self.topk_method = config.topk_method
           # ... rest of implementation
   ```

2. **Create Experiment Config**:
   ```python
   moe_experiments = {
       'baseline': {'topk_method': 'standard', 'experts': 8},
       'noaux_tc': {'topk_method': 'noaux_tc', 'experts': 8},
       'aux_tc': {'topk_method': 'aux_tc', 'experts': 8},
       'scaling_0.1': {'routed_scaling_factor': 0.1, 'experts': 8},
       'scaling_0.5': {'routed_scaling_factor': 0.5, 'experts': 8},
       'scaling_1.0': {'routed_scaling_factor': 1.0, 'experts': 8}
   }
   ```

3. **Test Parameters**:
   - Expert counts: [4, 8, 16, 32]
   - Top-k values: [1, 2, 4]
   - Scaling factors: [0.1, 0.5, 1.0]

### Metrics to Measure
- **Expert Utilization**: Percentage of experts used
- **Load Balancing**: Variance in expert usage
- **Performance**: Model perplexity
- **Efficiency**: Training time

### Expected Results
- Advanced routing should improve expert utilization
- Better load balancing with auxiliary losses
- Performance improvement with optimal scaling

## 📋 Experiment 4: Optimization Algorithm Study

### Objective
Compare Muon optimizer vs Adam/AdamW for training efficiency.

### Code References
- **Baseline**: Standard `torch.optim.Adam`
- **Variant**: `Muon` optimizer in `llm.py` (lines 95-150)

### Implementation Steps
1. **Extract Optimizer Classes**:
   ```python
   # From llm.py lines 95-150
   class Muon(torch.optim.Optimizer):
       def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
           defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
           super().__init__(params, defaults)
       # ... rest of implementation
   ```

2. **Create Experiment Config**:
   ```python
   optimizer_experiments = {
       'adam': {'optimizer': 'Adam', 'lr': 1e-3, 'weight_decay': 0.01},
       'adamw': {'optimizer': 'AdamW', 'lr': 1e-3, 'weight_decay': 0.01},
       'muon': {'optimizer': 'Muon', 'lr': 0.02, 'momentum': 0.95, 'ns_steps': 5}
   }
   ```

3. **Test Parameters**:
   - Learning rates: [1e-4, 5e-4, 1e-3, 2e-3]
   - Training steps: 3000
   - Momentum values: [0.9, 0.95, 0.99] (for Muon)

### Metrics to Measure
- **Convergence**: Steps to reach target loss
- **Final Performance**: Validation perplexity
- **Training Stability**: Loss variance
- **Memory**: Peak memory usage

### Expected Results
- Muon should converge faster
- Better final performance with Muon
- More stable training with Muon

## 📋 Experiment 5: RoPE Scaling Study

### Objective
Test different rotary position embedding scales for better long sequence handling.

### Code References
- **Baseline**: Standard RoPE implementation
- **Variant**: `DeepseekV3Attention` in `kimi_k2_modeling.py` (line 646)

### Implementation Steps
1. **Extract RoPE Configuration**:
   ```python
   # From kimi_k2_modeling.py line 646
   self.rope_theta = config.rope_theta
   ```

2. **Create Experiment Config**:
   ```python
   rope_experiments = {
       'baseline': {'rope_theta': 10000},
       'theta_50000': {'rope_theta': 50000},
       'theta_100000': {'rope_theta': 100000},
       'theta_200000': {'rope_theta': 200000}
   }
   ```

3. **Test Parameters**:
   - Sequence lengths: [512, 1024, 2048, 4096]
   - Theta values: [10000, 50000, 100000, 200000]
   - Training steps: 2000

### Metrics to Measure
- **Long Sequence Performance**: Perplexity on long sequences
- **Training Stability**: Loss convergence
- **Memory Usage**: Peak memory for different lengths

### Expected Results
- Higher theta values should help with longer sequences
- Optimal theta depends on sequence length
- Performance improvement on long sequences

## 📋 Experiment 6: Combined Architecture Study

### Objective
Test combinations of all improvements to find optimal architecture.

### Implementation Steps
1. **Create Combined Configs**:
   ```python
   combined_experiments = {
       'baseline': {
           'attention': 'standard',
           'normalization': 'layernorm',
           'moe': 'standard',
           'optimizer': 'adam'
       },
       'attention_only': {
           'attention': 'flash',
           'normalization': 'layernorm',
           'moe': 'standard',
           'optimizer': 'adam'
       },
       'normalization_only': {
           'attention': 'standard',
           'normalization': 'rmsnorm',
           'moe': 'standard',
           'optimizer': 'adam'
       },
       'moe_only': {
           'attention': 'standard',
           'normalization': 'layernorm',
           'moe': 'advanced',
           'optimizer': 'adam'
       },
       'optimizer_only': {
           'attention': 'standard',
           'normalization': 'layernorm',
           'moe': 'standard',
           'optimizer': 'muon'
       },
       'all_improvements': {
           'attention': 'flash',
           'normalization': 'rmsnorm',
           'moe': 'advanced',
           'optimizer': 'muon'
       }
   }
   ```

2. **Ablation Study**:
   - Start with all improvements
   - Remove one component at a time
   - Measure performance degradation

### Metrics to Measure
- **Overall Performance**: Validation perplexity
- **Training Efficiency**: Time to convergence
- **Component Contribution**: Individual component impact
- **Synergy Effects**: Combined vs individual improvements

### Expected Results
- Combined improvements should provide best performance
- Some components may have synergistic effects
- Clear ranking of component importance

## 📊 Statistical Analysis Plan

### Significance Testing
- **t-tests**: Compare performance between variants
- **Effect Size**: Cohen's d for practical significance
- **Confidence Intervals**: 95% CI for all metrics

### Multiple Comparisons
- **Bonferroni Correction**: Adjust p-values for multiple tests
- **False Discovery Rate**: Control for multiple comparisons

### Power Analysis
- **Sample Size**: 3 runs per experiment (minimum)
- **Effect Size**: Detect 5% improvement with 80% power
- **Alpha Level**: 0.05 for significance

## 🎯 Success Criteria

### Performance Targets
- **Perplexity**: >5% improvement over baseline
- **Training Speed**: >10% improvement in time to convergence
- **Memory Efficiency**: >15% reduction in peak memory usage
- **Stability**: <50% reduction in training loss variance

### Statistical Requirements
- **Significance**: p < 0.05 for all improvements
- **Effect Size**: Cohen's d > 0.5 (medium effect)
- **Reproducibility**: Consistent results across 3 runs

## 📈 Implementation Timeline

### Week 1-2: Setup
- Extract and modularize code components
- Set up experiment framework
- Establish baseline

### Week 3-6: Individual Experiments
- Run Experiments 1-5
- Collect and analyze data
- Document results

### Week 7-8: Combined Experiments
- Run Experiment 6
- Perform ablation studies
- Analyze synergy effects

### Week 9-10: Analysis
- Statistical analysis
- Create visualizations
- Write results section

---

**Next Steps**: Review `implementation_guide.md` for detailed code integration steps
