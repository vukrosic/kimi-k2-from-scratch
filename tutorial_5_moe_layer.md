# Tutorial 5: Mixture of Experts (MoE) Layer in DeepSeek-V3

## What is the MoE Layer?

The MoE layer in DeepSeek-V3 (`DeepseekV3MoE`) combines multiple expert MLPs, routed via the gate. It supports distributed experts across ranks (ep_size >1), with optional shared experts for common knowledge. Less known is the `moe_infer` method using all-to-all communication for efficient parallel computation without gradients.

This replaces dense MLPs in certain layers for sparse scaling.

### Key Benefits:
- Activates only selected experts per token.
- Distributed for large expert counts.
- Shared experts reduce redundancy.

## Code Implementation

Core MoE class (focus on key parts; full is longer):

```python
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np

class DeepseekV3MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        if hasattr(config, "ep_size") and config.ep_size > 1:
            assert config.ep_size == dist.get_world_size()
            self.ep_size = config.ep_size
            self.experts_per_rank = config.n_routed_experts // config.ep_size
            self.ep_rank = dist.get_rank()
            self.experts = nn.ModuleList([
                DeepseekV3MLP(config, intermediate_size=config.moe_intermediate_size)
                if i >= self.ep_rank * self.experts_per_rank and i < (self.ep_rank + 1) * self.experts_per_rank
                else None
                for i in range(config.n_routed_experts)
            ])
        else:
            self.ep_size = 1
            self.experts_per_rank = config.n_routed_experts
            self.ep_rank = 0
            self.experts = nn.ModuleList([
                DeepseekV3MLP(config, intermediate_size=config.moe_intermediate_size)
                for i in range(config.n_routed_experts)
            ])
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV3MLP(config=config, intermediate_size=intermediate_size)

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if not self.training:
            y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
        else:
            # Training forward (simplified; full uses dispatch)
            y = torch.zeros_like(hidden_states)
            # ... (dispatch to experts, weighted sum)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        sorted_tokens_shape = sorted_tokens.shape
        if self.ep_size > 1:
            tokens_per_ep_rank = tokens_per_expert.view(self.ep_size, -1).sum(dim=1)
            tokens_per_expert_group = tokens_per_expert.new_empty(tokens_per_expert.shape[0])
            dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert)
            output_splits = tokens_per_expert_group.view(self.ep_size, -1).sum(1).cpu().numpy().tolist()
            gathered_tokens = sorted_tokens.new_empty(tokens_per_expert_group.sum(dim=0).cpu().item(), sorted_tokens.shape[1])
            input_split_sizes = tokens_per_ep_rank.cpu().numpy().tolist()
            dist.all_to_all(list(gathered_tokens.split(output_splits)), list(sorted_tokens.split(input_split_sizes)))
            tokens_per_expert_post_gather = tokens_per_expert_group.view(self.ep_size, self.experts_per_rank).sum(dim=0)
            gatherd_idxs = np.zeros(shape=(gathered_tokens.shape[0],), dtype=np.int32)
            s = 0
            for i, k in enumerate(tokens_per_expert_group.cpu().numpy()):
                gatherd_idxs[s : s + k] = i % self.experts_per_rank
                s += k
            gatherd_idxs = gatherd_idxs.argsort()
            sorted_tokens = gathered_tokens[gatherd_idxs]
            tokens_per_expert = tokens_per_expert_post_gather
        tokens_per_expert = tokens_per_expert.cpu().numpy()
        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx
        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
        if self.ep_size > 1:
            new_x = torch.empty_like(outs)
            new_x[gatherd_idxs] = outs
            gathered_tokens = new_x.new_empty(*sorted_tokens_shape)
            dist.all_to_all(list(gathered_tokens.split(input_split_sizes)), list(new_x.split(output_splits)))
            outs = gathered_tokens
        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = new_x.view(*topk_ids.shape, -1).type(topk_weight.dtype).mul_(topk_weight.unsqueeze(dim=-1)).sum(dim=1).type(new_x.dtype)
        return final_out
```

### Line-by-Line Breakdown

#### Initialization (`__init__` method):
- `super().__init__()`: nn.Module setup.
- If `ep_size > 1` (distributed): 
  - `self.experts_per_rank = n_routed_experts // ep_size`: Experts per GPU.
  - `self.ep_rank = dist.get_rank()`: Current rank.
  - `self.experts = nn.ModuleList([...])`: Only instantiates local experts (None for others).
- Else: All experts on single device.
- `self.gate = MoEGate(config)`: Router.
- If `n_shared_experts`: Creates a wider MLP for shared computation.

This shards experts across devices for scalability.

#### Forward (`forward` method):
- `identity = hidden_states`: For residual if shared.
- `orig_shape = hidden_states.shape`: To reshape output.
- `topk_idx, topk_weight = self.gate(hidden_states)`: Gets routing.
- `hidden_states.view(-1, ...)`: Flattens for processing.
- If not training: `y = self.moe_infer(...)`: Efficient inference.
- If shared: `y += self.shared_experts(identity)`: Adds shared output.
- Returns reshaped y.

Training uses a dispatch mechanism (not shown for brevity).

#### Inference (`moe_infer` method, @no_grad):
- `cnts.scatter_(1, topk_ids, 1)`: Counts tokens per expert.
- `tokens_per_expert = cnts.sum(0)`: Total per expert.
- `idxs = topk_ids.view(-1).argsort()`: Sorts tokens by expert order for batching.
- `sorted_tokens = x[idxs // ...]`: Reorders input.
- If distributed (`ep_size >1`):
  - `tokens_per_ep_rank = ... .sum(1)`: Tokens per rank.
  - `dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert)`: Shares counts.
  - `output_splits = ...`: Splits for output.
  - `gathered_tokens = ...`: Allocates for gathered input.
  - `dist.all_to_all(..., list(sorted_tokens.split(...)))`: Distributes tokens to owning ranks.
  - Adjusts indices and sorts for local processing.
- Loop over experts: If tokens >0, run local expert on batch, collect outputs.
- `outs = torch.cat(outputs)`: Concat local outputs.
- If distributed: Reverse all-to-all to gather results back.
- `new_x[idxs] = outs`: Unsorts to original token order.
- `final_out = ... .mul_(topk_weight.unsqueeze(-1)).sum(1)`: Weighted sum over experts.

This efficiently computes only needed experts with comms.

## Step-by-Step Example Walkthrough

Assume single device, 4 experts, top_k=2, input (1,4,512) -> 4 tokens.

1. Gate: Returns topk_idx (4,2) e.g., [[0,1],[1,2],[0,3],[2,3]], weights (4,2).
2. Flatten hidden: (4,512).
3. cnts: (4,4), e.g., expert0:2 tokens, expert1:2, etc.
4. tokens_per_expert: [2,2,1,1].
5. idxs: Sort order, sorted_tokens reordered.
6. Loop: For expert0: Run MLP on its 2 tokens -> output0.
7. Cat outs: (6,512) (sum tokens).
8. new_x[idxs] = outs: Back to original order (4,512).
9. View (4,2,512), mul weights.unsqueeze(-1): (4,2,1,512), sum dim=1: Weighted avg (4,512).

In distributed, all-to-all moves tokens to expert owners.

## Why Use MoE Layer in DeepSeek-V3?

DeepSeek-V3's MoE layer enables 236B params with only 21B active, using distributed all-to-all for efficiency. Shared experts handle universal tasks, while routed specialize, key for its hybrid dense-MoE design.

Next Tutorial: Attention Mechanism.
