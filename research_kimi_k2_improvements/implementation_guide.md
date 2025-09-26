# Implementation Guide: Code Integration and Experimentation

## 🎯 Overview

This guide provides step-by-step instructions for integrating components from `kimi_k2_modeling.py` into `llm.py` and running the planned experiments.

## 📁 Project Structure Setup

### 1. Create Modular Code Structure

```
research_kimi_k2_improvements/
├── code_components/
│   ├── __init__.py
│   ├── attention_variants.py
│   ├── normalization_layers.py
│   ├── moe_implementations.py
│   ├── optimizers.py
│   └── configs/
│       ├── __init__.py
│       ├── baseline_config.py
│       ├── attention_config.py
│       ├── normalization_config.py
│       ├── moe_config.py
│       └── optimizer_config.py
├── experiments/
│   ├── __init__.py
│   ├── experiment_runner.py
│   ├── metrics.py
│   └── utils.py
└── results/
    ├── logs/
    ├── checkpoints/
    └── plots/
```

### 2. Extract and Modularize Components

#### Step 1: Extract Attention Components

Create `code_components/attention_variants.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class StandardMultiHeadAttention(nn.Module):
    """Standard multi-head attention from llm.py"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        
        # Linear transformations
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return self.w_o(context)

class FlashAttention2(nn.Module):
    """Flash Attention 2 implementation from kimi_k2_modeling.py"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        # Extract from DeepseekV3FlashAttention2
        # ... (copy relevant parts from kimi_k2_modeling.py lines 860-946)
        
    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        # ... (copy forward method from kimi_k2_modeling.py)
        pass
```

#### Step 2: Extract Normalization Components

Create `code_components/normalization_layers.py`:

```python
import torch
import torch.nn as nn

class StandardLayerNorm(nn.Module):
    """Standard LayerNorm"""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias

class RMSNorm(nn.Module):
    """RMSNorm from kimi_k2_modeling.py"""
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

#### Step 3: Extract MoE Components

Create `code_components/moe_implementations.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class StandardMoE(nn.Module):
    """Standard MoE implementation from llm.py"""
    def __init__(self, d_model, d_ff, num_experts, top_k):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Linear(d_model, num_experts)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        x_flat = x.view(-1, d_model)
        
        # Gating
        gate_scores = self.gate(x_flat)
        gate_probs = F.softmax(gate_scores, dim=-1)
        
        # Top-k selection
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Expert computation
        output = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]
            expert_output = self.experts[expert_idx](x_flat)
            output += top_k_probs[:, i:i+1] * expert_output
            
        return output.view(batch_size, seq_len, d_model)

class AdvancedMoE(nn.Module):
    """Advanced MoE from kimi_k2_modeling.py"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Extract from DeepseekV3MoE (lines 475-522)
        # ... (copy relevant parts)
        
    def forward(self, hidden_states):
        # ... (copy forward method from kimi_k2_modeling.py)
        pass
```

#### Step 4: Extract Optimizer Components

Create `code_components/optimizers.py`:

```python
import torch
import torch.optim as optim
from typing import List, Dict, Any

class MuonOptimizer(torch.optim.Optimizer):
    """Muon optimizer from llm.py"""
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)
        
    def step(self, closure=None):
        # ... (copy step method from llm.py lines 95-150)
        pass

def get_optimizer(model, config):
    """Get optimizer based on config"""
    if config.optimizer == 'adam':
        return optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == 'adamw':
        return optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == 'muon':
        return MuonOptimizer(model.parameters(), lr=config.lr, momentum=config.momentum)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
```

### 3. Create Configuration System

#### Step 1: Base Configuration

Create `code_components/configs/base_config.py`:

```python
from dataclasses import dataclass
from typing import Optional, List, Tuple

@dataclass
class BaseConfig:
    # Model architecture
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1536
    vocab_size: int = 50257
    
    # Training parameters
    batch_size: int = 24
    max_steps: int = 1000
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    
    # Data parameters
    max_seq_len: int = 512
    num_documents: int = 2000
    max_tokens: int = 500000
    
    # Evaluation
    eval_every: int = 500
    eval_steps: int = 100
    
    # Regularization
    dropout: float = 0.1
    grad_clip: float = 1.0
    
    # Technical
    use_amp: bool = True
    seed: int = 42
    
    # MoE specific
    num_experts: int = 8
    expert_top_k: int = 2
    load_balancing_weight: float = 0.01
```

#### Step 2: Experiment-Specific Configurations

Create `code_components/configs/experiment_configs.py`:

```python
from .base_config import BaseConfig

class AttentionExperimentConfig(BaseConfig):
    """Configuration for attention experiments"""
    attention_type: str = 'standard'  # 'standard' or 'flash'
    sequence_lengths: List[int] = [512, 1024, 2048]
    batch_sizes: List[int] = [8, 16, 32]

class NormalizationExperimentConfig(BaseConfig):
    """Configuration for normalization experiments"""
    normalization_type: str = 'layernorm'  # 'layernorm' or 'rmsnorm'
    learning_rates: List[float] = [1e-4, 5e-4, 1e-3, 2e-3]
    epsilon_values: List[float] = [1e-6, 1e-8]

class MoEExperimentConfig(BaseConfig):
    """Configuration for MoE experiments"""
    moe_type: str = 'standard'  # 'standard' or 'advanced'
    expert_counts: List[int] = [4, 8, 16, 32]
    top_k_values: List[int] = [1, 2, 4]
    scaling_factors: List[float] = [0.1, 0.5, 1.0]

class OptimizerExperimentConfig(BaseConfig):
    """Configuration for optimizer experiments"""
    optimizer_type: str = 'adam'  # 'adam', 'adamw', or 'muon'
    learning_rates: List[float] = [1e-4, 5e-4, 1e-3, 2e-3]
    momentum_values: List[float] = [0.9, 0.95, 0.99]
```

### 4. Create Experiment Runner

Create `experiments/experiment_runner.py`:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import json
import os
from typing import Dict, Any, List
from tqdm import tqdm

from code_components.attention_variants import StandardMultiHeadAttention, FlashAttention2
from code_components.normalization_layers import StandardLayerNorm, RMSNorm
from code_components.moe_implementations import StandardMoE, AdvancedMoE
from code_components.optimizers import get_optimizer
from code_components.configs.experiment_configs import *

class ExperimentRunner:
    def __init__(self, config, experiment_name):
        self.config = config
        self.experiment_name = experiment_name
        self.results = {}
        self.setup_experiment()
        
    def setup_experiment(self):
        """Setup experiment based on configuration"""
        # Set random seed
        torch.manual_seed(self.config.seed)
        
        # Create model based on config
        self.model = self.create_model()
        
        # Create optimizer
        self.optimizer = get_optimizer(self.model, self.config)
        
        # Create data loader
        self.data_loader = self.create_data_loader()
        
        # Setup logging
        self.setup_logging()
        
    def create_model(self):
        """Create model based on configuration"""
        # This would integrate components from both llm.py and kimi_k2_modeling.py
        # based on the experiment configuration
        pass
        
    def create_data_loader(self):
        """Create data loader for training"""
        # Implementation depends on your data setup
        pass
        
    def setup_logging(self):
        """Setup logging for experiment"""
        self.log_dir = f"results/logs/{self.experiment_name}"
        os.makedirs(self.log_dir, exist_ok=True)
        
    def run_experiment(self):
        """Run the experiment"""
        print(f"Starting experiment: {self.experiment_name}")
        
        # Training loop
        start_time = time.time()
        for step in tqdm(range(self.config.max_steps)):
            # Training step
            loss = self.training_step()
            
            # Logging
            if step % 100 == 0:
                self.log_metrics(step, loss)
                
            # Evaluation
            if step % self.config.eval_every == 0:
                eval_metrics = self.evaluate()
                self.log_eval_metrics(step, eval_metrics)
                
        total_time = time.time() - start_time
        
        # Final evaluation
        final_metrics = self.evaluate()
        
        # Save results
        self.save_results(final_metrics, total_time)
        
        return final_metrics
        
    def training_step(self):
        """Single training step"""
        # Implementation depends on your training setup
        pass
        
    def evaluate(self):
        """Evaluate model performance"""
        # Implementation depends on your evaluation setup
        pass
        
    def log_metrics(self, step, loss):
        """Log training metrics"""
        # Implementation for logging
        pass
        
    def log_eval_metrics(self, step, metrics):
        """Log evaluation metrics"""
        # Implementation for logging
        pass
        
    def save_results(self, metrics, total_time):
        """Save experiment results"""
        results = {
            'experiment_name': self.experiment_name,
            'config': self.config.__dict__,
            'metrics': metrics,
            'total_time': total_time
        }
        
        with open(f"{self.log_dir}/results.json", 'w') as f:
            json.dump(results, f, indent=2)

def run_attention_experiment():
    """Run attention mechanism comparison"""
    config = AttentionExperimentConfig()
    
    # Test different attention types
    for attention_type in ['standard', 'flash']:
        config.attention_type = attention_type
        experiment_name = f"attention_{attention_type}"
        
        runner = ExperimentRunner(config, experiment_name)
        results = runner.run_experiment()
        
        print(f"Results for {attention_type} attention: {results}")

def run_normalization_experiment():
    """Run normalization comparison"""
    config = NormalizationExperimentConfig()
    
    # Test different normalization types
    for norm_type in ['layernorm', 'rmsnorm']:
        config.normalization_type = norm_type
        experiment_name = f"normalization_{norm_type}"
        
        runner = ExperimentRunner(config, experiment_name)
        results = runner.run_experiment()
        
        print(f"Results for {norm_type}: {results}")

def run_moe_experiment():
    """Run MoE comparison"""
    config = MoEExperimentConfig()
    
    # Test different MoE types
    for moe_type in ['standard', 'advanced']:
        config.moe_type = moe_type
        experiment_name = f"moe_{moe_type}"
        
        runner = ExperimentRunner(config, experiment_name)
        results = runner.run_experiment()
        
        print(f"Results for {moe_type} MoE: {results}")

def run_optimizer_experiment():
    """Run optimizer comparison"""
    config = OptimizerExperimentConfig()
    
    # Test different optimizers
    for optimizer_type in ['adam', 'adamw', 'muon']:
        config.optimizer_type = optimizer_type
        experiment_name = f"optimizer_{optimizer_type}"
        
        runner = ExperimentRunner(config, experiment_name)
        results = runner.run_experiment()
        
        print(f"Results for {optimizer_type}: {results}")

if __name__ == "__main__":
    # Run all experiments
    run_attention_experiment()
    run_normalization_experiment()
    run_moe_experiment()
    run_optimizer_experiment()
```

### 5. Create Metrics and Evaluation

Create `experiments/metrics.py`:

```python
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List
import time

class MetricsCalculator:
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset all metrics"""
        self.losses = []
        self.perplexities = []
        self.training_times = []
        self.memory_usage = []
        
    def calculate_perplexity(self, logits, targets):
        """Calculate perplexity from logits and targets"""
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        perplexity = torch.exp(loss)
        return perplexity.item()
        
    def calculate_accuracy(self, logits, targets):
        """Calculate accuracy from logits and targets"""
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == targets).float()
        accuracy = correct.mean().item()
        return accuracy
        
    def measure_memory_usage(self):
        """Measure current GPU memory usage"""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**3  # GB
        return 0
        
    def update_metrics(self, loss, logits, targets, training_time):
        """Update all metrics"""
        self.losses.append(loss)
        self.perplexities.append(self.calculate_perplexity(logits, targets))
        self.training_times.append(training_time)
        self.memory_usage.append(self.measure_memory_usage())
        
    def get_summary(self):
        """Get summary of all metrics"""
        return {
            'mean_loss': np.mean(self.losses),
            'std_loss': np.std(self.losses),
            'mean_perplexity': np.mean(self.perplexities),
            'std_perplexity': np.std(self.perplexities),
            'mean_training_time': np.mean(self.training_times),
            'total_training_time': np.sum(self.training_times),
            'mean_memory_usage': np.mean(self.memory_usage),
            'max_memory_usage': np.max(self.memory_usage)
        }
```

### 6. Usage Instructions

#### Step 1: Setup Environment

```bash
# Create virtual environment
python -m venv research_env
source research_env/bin/activate  # On Windows: research_env\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio
pip install transformers datasets tqdm
pip install matplotlib seaborn pandas
pip install wandb  # For experiment tracking (optional)
```

#### Step 2: Run Individual Experiments

```python
# Run attention experiment
from experiments.experiment_runner import run_attention_experiment
run_attention_experiment()

# Run normalization experiment
from experiments.experiment_runner import run_normalization_experiment
run_normalization_experiment()

# Run MoE experiment
from experiments.experiment_runner import run_moe_experiment
run_moe_experiment()

# Run optimizer experiment
from experiments.experiment_runner import run_optimizer_experiment
run_optimizer_experiment()
```

#### Step 3: Run Combined Experiments

```python
# Run all experiments
from experiments.experiment_runner import *
run_attention_experiment()
run_normalization_experiment()
run_moe_experiment()
run_optimizer_experiment()
```

#### Step 4: Analyze Results

```python
import json
import pandas as pd
import matplotlib.pyplot as plt

# Load results
def load_results(experiment_name):
    with open(f"results/logs/{experiment_name}/results.json", 'r') as f:
        return json.load(f)

# Compare results
attention_standard = load_results("attention_standard")
attention_flash = load_results("attention_flash")

print(f"Standard Attention Perplexity: {attention_standard['metrics']['mean_perplexity']}")
print(f"Flash Attention Perplexity: {attention_flash['metrics']['mean_perplexity']}")
```

## 🎯 Key Integration Points

### 1. Model Architecture Integration

The key is to create a flexible model class that can swap components:

```python
class FlexibleModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Choose attention mechanism
        if config.attention_type == 'standard':
            self.attention = StandardMultiHeadAttention(config.d_model, config.n_heads)
        elif config.attention_type == 'flash':
            self.attention = FlashAttention2(config)
            
        # Choose normalization
        if config.normalization_type == 'layernorm':
            self.norm = StandardLayerNorm(config.d_model)
        elif config.normalization_type == 'rmsnorm':
            self.norm = RMSNorm(config.d_model)
            
        # Choose MoE
        if config.moe_type == 'standard':
            self.moe = StandardMoE(config.d_model, config.d_ff, config.num_experts, config.expert_top_k)
        elif config.moe_type == 'advanced':
            self.moe = AdvancedMoE(config)
```

### 2. Configuration Management

Use dataclasses for easy configuration management:

```python
@dataclass
class ExperimentConfig:
    # Base parameters
    d_model: int = 384
    n_heads: int = 8
    
    # Experiment-specific parameters
    attention_type: str = 'standard'
    normalization_type: str = 'layernorm'
    moe_type: str = 'standard'
    optimizer_type: str = 'adam'
    
    # Training parameters
    learning_rate: float = 1e-3
    batch_size: int = 24
    max_steps: int = 1000
```

### 3. Experiment Tracking

Use Weights & Biases or similar for experiment tracking:

```python
import wandb

def setup_wandb(config, experiment_name):
    wandb.init(
        project="kimi-k2-research",
        name=experiment_name,
        config=config.__dict__
    )

def log_metrics(step, metrics):
    wandb.log({
        "step": step,
        "loss": metrics["loss"],
        "perplexity": metrics["perplexity"],
        "accuracy": metrics["accuracy"]
    })
```

## 🚀 Next Steps

1. **Extract Components**: Copy relevant code from both files
2. **Create Modular Structure**: Organize code into separate modules
3. **Setup Configuration**: Create flexible configuration system
4. **Run Baseline**: Establish baseline performance
5. **Run Experiments**: Execute planned experiments
6. **Analyze Results**: Compare and analyze results
7. **Write Paper**: Document findings and conclusions

---

**Next Steps**: Review `evaluation_framework.md` for detailed evaluation methodology
