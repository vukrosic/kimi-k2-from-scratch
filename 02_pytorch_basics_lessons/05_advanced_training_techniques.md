# Advanced Training Techniques

## Learning Objectives
- Master advanced optimization techniques
- Understand regularization and overfitting prevention
- Learn about model ensembling and knowledge distillation
- Practice with distributed training and mixed precision

## Advanced Optimization

### Learning Rate Scheduling
```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Different learning rate schedulers
def get_schedulers(optimizer, total_epochs):
    schedulers = {
        'StepLR': optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1),
        'ExponentialLR': optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95),
        'CosineAnnealingLR': optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs),
        'ReduceLROnPlateau': optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10),
        'OneCycleLR': optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, total_steps=total_epochs),
        'CosineAnnealingWarmRestarts': optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    }
    return schedulers

# Custom learning rate scheduler
class CustomScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, max_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.current_epoch = 0
    
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # Warmup phase
            lr = self.base_lr + (self.max_lr - self.base_lr) * self.current_epoch / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.base_lr + (self.max_lr - self.base_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1

# Visualize learning rate schedules
def plot_lr_schedules():
    model = nn.Linear(10, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    total_epochs = 100
    
    schedulers = get_schedulers(optimizer, total_epochs)
    
    plt.figure(figsize=(15, 10))
    for i, (name, scheduler) in enumerate(schedulers.items()):
        plt.subplot(2, 3, i+1)
        lrs = []
        
        for epoch in range(total_epochs):
            lrs.append(optimizer.param_groups[0]['lr'])
            if name == 'ReduceLROnPlateau':
                # Simulate validation loss
                val_loss = 1.0 - epoch * 0.01 + np.random.normal(0, 0.1)
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        plt.plot(lrs)
        plt.title(f'{name}')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Run visualization
plot_lr_schedules()
```

### Advanced Optimizers
```python
# Custom optimizer implementations
class RAdam(optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(RAdam, self).__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Compute the length of the approximated SMA
                sma_inf = 2.0 / (1 - beta2) - 1
                sma_t = sma_inf - 2 * state['step'] * (1 - beta2 ** state['step']) / bias_correction2
                
                # Update parameters
                if sma_t >= 5:
                    # Rectified update
                    r_t = ((sma_t - 4) * (sma_t - 2) * sma_inf) / ((sma_inf - 4) * (sma_inf - 2) * sma_t)
                    denom = (exp_avg_sq.sqrt() / bias_correction2 ** 0.5).add_(group['eps'])
                    p.data.addcdiv_(exp_avg, denom, value=-group['lr'] * r_t / bias_correction1)
                else:
                    # Unrectified update
                    p.data.add_(exp_avg, alpha=-group['lr'] / bias_correction1)
                
                # Weight decay
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])
        
        return loss

# Lookahead optimizer wrapper
class Lookahead(optim.Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = {}
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0
    
    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)
    
    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)
    
    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update_lookahead()
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss
    
    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }
    
    def load_state_dict(self, state_dict):
        slow_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(fast_state_dict)
        self.optimizer.load_state_dict(slow_state_dict)

# Usage example
model = nn.Linear(10, 1)
base_optimizer = optim.Adam(model.parameters(), lr=0.001)
lookahead_optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)
```

## Regularization Techniques

### Advanced Regularization
```python
# Dropout variants
class DropBlock2d(nn.Module):
    def __init__(self, drop_rate=0.1, block_size=7):
        super(DropBlock2d, self).__init__()
        self.drop_rate = drop_rate
        self.block_size = block_size
    
    def forward(self, x):
        if not self.training or self.drop_rate == 0:
            return x
        
        # Get input dimensions
        N, C, H, W = x.shape
        
        # Calculate gamma
        total_size = W * H
        clipped_block_size = min(self.block_size, min(W, H))
        gamma = self.drop_rate * total_size / (clipped_block_size ** 2) / ((W - self.block_size + 1) * (H - self.block_size + 1))
        
        # Sample mask
        mask = torch.bernoulli(torch.full((N, C, H - self.block_size + 1, W - self.block_size + 1), gamma))
        
        # Expand mask
        mask = F.pad(mask, (0, self.block_size - 1, 0, self.block_size - 1))
        mask = F.max_pool2d(mask, kernel_size=self.block_size, stride=1, padding=self.block_size // 2)
        mask = 1 - mask
        
        # Apply mask
        x = x * mask * mask.numel() / mask.sum()
        
        return x

# Label smoothing
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

# Mixup augmentation
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# CutMix augmentation
def cutmix_data(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    
    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    
    # Uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2
```

### Gradient Clipping and Normalization
```python
# Gradient clipping
def clip_gradients(model, max_norm=1.0):
    """Clip gradients to prevent exploding gradients"""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

# Gradient accumulation
class GradientAccumulator:
    def __init__(self, accumulation_steps=4):
        self.accumulation_steps = accumulation_steps
        self.step_count = 0
    
    def should_update(self):
        self.step_count += 1
        return self.step_count % self.accumulation_steps == 0
    
    def scale_loss(self, loss):
        return loss / self.accumulation_steps

# Usage in training loop
def train_with_gradient_accumulation(model, train_loader, optimizer, criterion, accumulation_steps=4):
    model.train()
    accumulator = GradientAccumulator(accumulation_steps)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        output = model(data)
        loss = criterion(output, target)
        
        # Scale loss for gradient accumulation
        loss = accumulator.scale_loss(loss)
        loss.backward()
        
        if accumulator.should_update():
            # Clip gradients
            clip_gradients(model, max_norm=1.0)
            
            # Update parameters
            optimizer.step()
            optimizer.zero_grad()
```

## Model Ensembling

### Ensemble Methods
```python
class ModelEnsemble(nn.Module):
    def __init__(self, models, weights=None):
        super(ModelEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            self.weights = weights
    
    def forward(self, x):
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # Weighted average
        ensemble_pred = sum(w * pred for w, pred in zip(self.weights, predictions))
        return ensemble_pred

# Snapshot ensemble
class SnapshotEnsemble:
    def __init__(self, model, optimizer, snapshot_freq=10):
        self.model = model
        self.optimizer = optimizer
        self.snapshot_freq = snapshot_freq
        self.snapshots = []
    
    def save_snapshot(self, epoch):
        if epoch % self.snapshot_freq == 0:
            snapshot = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': epoch
            }
            self.snapshots.append(snapshot)
    
    def load_snapshot(self, snapshot_idx):
        snapshot = self.snapshots[snapshot_idx]
        self.model.load_state_dict(snapshot['model_state_dict'])
        self.optimizer.load_state_dict(snapshot['optimizer_state_dict'])
    
    def ensemble_predict(self, x):
        predictions = []
        for snapshot in self.snapshots:
            self.model.load_state_dict(snapshot['model_state_dict'])
            pred = self.model(x)
            predictions.append(pred)
        
        return torch.stack(predictions).mean(dim=0)

# Knowledge distillation
class KnowledgeDistillation:
    def __init__(self, teacher_model, student_model, temperature=3.0, alpha=0.7):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def distillation_loss(self, student_outputs, teacher_outputs, targets):
        # Soft targets from teacher
        soft_targets = self.softmax(teacher_outputs / self.temperature)
        soft_prob = self.log_softmax(student_outputs / self.temperature)
        
        # Hard targets
        hard_loss = F.cross_entropy(student_outputs, targets)
        
        # Soft loss
        soft_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (self.temperature ** 2)
        
        # Combined loss
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return total_loss
```

## Distributed Training

### Data Parallel Training
```python
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()

def train_distributed(rank, world_size, model, train_loader, num_epochs):
    """Distributed training function"""
    setup_distributed(rank, world_size)
    
    # Move model to GPU
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(rank), target.to(rank)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Rank {rank}, Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    cleanup_distributed()

# Launch distributed training
def launch_distributed_training(model, train_loader, num_epochs, world_size=2):
    mp.spawn(train_distributed, args=(world_size, model, train_loader, num_epochs), nprocs=world_size, join=True)
```

### Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

def train_with_mixed_precision(model, train_loader, optimizer, criterion, num_epochs):
    """Training with mixed precision"""
    scaler = GradScaler()
    
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            
            optimizer.zero_grad()
            
            # Use autocast for forward pass
            with autocast():
                output = model(data)
                loss = criterion(output, target)
            
            # Scale loss and backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

# Gradient checkpointing for memory efficiency
def train_with_gradient_checkpointing(model, train_loader, optimizer, criterion, num_epochs):
    """Training with gradient checkpointing"""
    model.gradient_checkpointing_enable()
    
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
```

## Practice Exercises

### Exercise 1: Advanced Optimization
Implement a training pipeline that:
- Uses advanced optimizers (RAdam, Lookahead)
- Implements learning rate scheduling
- Includes gradient clipping and accumulation
- Achieves better convergence than standard Adam

### Exercise 2: Regularization Techniques
Build a model that:
- Uses multiple regularization techniques
- Implements data augmentation (Mixup, CutMix)
- Applies label smoothing
- Prevents overfitting effectively

### Exercise 3: Model Ensembling
Create an ensemble system that:
- Combines multiple models
- Uses snapshot ensemble
- Implements knowledge distillation
- Improves prediction accuracy

### Exercise 4: Distributed Training
Set up distributed training that:
- Uses multiple GPUs
- Implements mixed precision
- Includes gradient checkpointing
- Scales efficiently

## AI Learning Prompt

**Copy this prompt into ChatGPT or any AI chatbot:**

"I'm learning advanced training techniques in PyTorch. I understand basic training but I'm struggling with:

1. Advanced optimization techniques and learning rate scheduling
2. Regularization methods to prevent overfitting
3. Model ensembling and knowledge distillation
4. Distributed training and mixed precision
5. Memory optimization techniques
6. Best practices for large-scale training

Please:
- Explain each technique with practical examples
- Show me how to implement advanced optimizers
- Help me understand when to use different regularization methods
- Walk me through setting up distributed training
- Give me exercises to practice with real models
- Explain performance optimization strategies

I want to train large, efficient models for production use. Please provide hands-on examples and help me understand the implementation details."

## Key Takeaways
- Advanced optimizers can improve convergence and final performance
- Proper regularization is crucial for preventing overfitting
- Model ensembling can significantly improve prediction accuracy
- Distributed training enables training of large models
- Mixed precision training reduces memory usage and speeds up training
- Gradient checkpointing trades computation for memory efficiency

## Next Steps
Master these advanced techniques and you'll be ready for:
- Training large language models
- Computer vision at scale
- Production model deployment
- Research in deep learning
- Optimizing model performance
- Building efficient training pipelines
