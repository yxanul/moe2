import wandb
import torch
import numpy as np
from typing import Dict, List, Optional, Any
from collections import defaultdict


class WandbLogger:
    """Wandb logger for MoE training metrics."""
    
    def __init__(
        self,
        project: str = "moe-training",
        name: Optional[str] = None,
        config: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        mode: str = "online",
    ):
        self.project = project
        self.name = name
        self.config = config or {}
        self.tags = tags or []
        self.notes = notes
        self.mode = mode
        
        # Initialize wandb
        self.run = wandb.init(
            project=project,
            name=name,
            config=config,
            tags=tags,
            notes=notes,
            mode=mode,
        )
        
        # Metrics accumulator
        self.metrics_accumulator = defaultdict(list)
        
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to wandb."""
        wandb.log(metrics, step=step)
    
    def log_moe_metrics(
        self,
        loss: float,
        aux_loss: float,
        routing_stats: List[Dict],
        step: int,
        learning_rate: Optional[float] = None,
    ):
        """Log MoE-specific metrics."""
        
        # Basic metrics
        metrics = {
            'train/loss': loss,
            'train/aux_loss': aux_loss,
            'train/perplexity': np.exp(loss) if loss < 50 else float('inf'),
        }
        
        if learning_rate is not None:
            metrics['train/learning_rate'] = learning_rate
        
        # Process routing statistics per layer
        if routing_stats:
            # Calculate entropy and load balancing metrics
            layer_metrics = self._process_routing_stats(routing_stats)
            metrics.update(layer_metrics)
        
        # Log to wandb
        self.log(metrics, step=step)
    
    def _process_routing_stats(self, routing_stats: List[Dict]) -> Dict:
        """Process routing statistics from all layers."""
        metrics = {}
        
        for layer_idx, layer_stats in enumerate(routing_stats):
            # Extract auxiliary loss per layer
            if 'aux_loss' in layer_stats:
                metrics[f'routing/layer_{layer_idx}/aux_loss'] = layer_stats['aux_loss']
            
            # Extract load balancing loss
            if 'load_balancing_loss' in layer_stats:
                metrics[f'routing/layer_{layer_idx}/load_balance_loss'] = layer_stats['load_balancing_loss']
        
        # Calculate average metrics across layers
        if routing_stats:
            avg_aux_loss = np.mean([
                stats.get('aux_loss', 0) for stats in routing_stats
            ])
            metrics['routing/avg_aux_loss'] = avg_aux_loss
        
        return metrics
    
    def log_expert_utilization(
        self,
        expert_counts: torch.Tensor,
        layer_idx: int,
        step: int,
    ):
        """Log expert utilization statistics."""
        # Calculate utilization metrics
        total_tokens = expert_counts.sum().item()
        num_experts = len(expert_counts)
        
        if total_tokens > 0:
            # Calculate entropy of expert distribution
            probs = expert_counts.float() / total_tokens
            probs = probs[probs > 0]  # Remove zeros for entropy calculation
            entropy = -torch.sum(probs * torch.log(probs)).item()
            
            # Calculate coefficient of variation (load balance metric)
            mean_load = expert_counts.float().mean().item()
            std_load = expert_counts.float().std().item()
            cv = std_load / (mean_load + 1e-6)
            
            metrics = {
                f'experts/layer_{layer_idx}/entropy': entropy,
                f'experts/layer_{layer_idx}/load_cv': cv,
                f'experts/layer_{layer_idx}/total_tokens': total_tokens,
                f'experts/layer_{layer_idx}/mean_tokens_per_expert': mean_load,
            }
            
            # Log top-k expert usage
            top_k = 5
            top_experts, top_counts = torch.topk(expert_counts, min(top_k, num_experts))
            for i, (expert_id, count) in enumerate(zip(top_experts, top_counts)):
                metrics[f'experts/layer_{layer_idx}/top_{i}_expert_id'] = expert_id.item()
                metrics[f'experts/layer_{layer_idx}/top_{i}_token_count'] = count.item()
            
            self.log(metrics, step=step)
    
    def log_gradient_stats(self, model: torch.nn.Module, step: int):
        """Log gradient statistics."""
        total_norm = 0.0
        param_norms = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                
                # Log individual parameter group norms
                if 'expert' in name:
                    param_group = 'expert'
                elif 'gate' in name or 'router' in name:
                    param_group = 'gate'
                elif 'attention' in name:
                    param_group = 'attention'
                elif 'embed' in name or 'lm_head' in name:
                    param_group = 'embedding'
                else:
                    param_group = 'other'
                
                if param_group not in param_norms:
                    param_norms[param_group] = 0.0
                param_norms[param_group] += param_norm ** 2
        
        total_norm = total_norm ** 0.5
        
        metrics = {'gradients/total_norm': total_norm}
        for group, norm_sq in param_norms.items():
            metrics[f'gradients/{group}_norm'] = norm_sq ** 0.5
        
        self.log(metrics, step=step)
    
    def log_memory_stats(self, step: int):
        """Log GPU memory statistics."""
        if torch.cuda.is_available():
            metrics = {
                'system/gpu_memory_allocated_gb': torch.cuda.memory_allocated() / 1e9,
                'system/gpu_memory_reserved_gb': torch.cuda.memory_reserved() / 1e9,
            }
            
            # Get memory stats for each GPU if multiple GPUs
            if torch.cuda.device_count() > 1:
                for i in range(torch.cuda.device_count()):
                    metrics[f'system/gpu_{i}_memory_allocated_gb'] = (
                        torch.cuda.memory_allocated(i) / 1e9
                    )
            
            self.log(metrics, step=step)
    
    def log_training_progress(
        self,
        epoch: int,
        batch_idx: int,
        total_batches: int,
        tokens_processed: int,
        elapsed_time: float,
        step: int,
    ):
        """Log training progress metrics."""
        progress = (batch_idx + 1) / total_batches
        tokens_per_second = tokens_processed / elapsed_time if elapsed_time > 0 else 0
        
        metrics = {
            'progress/epoch': epoch,
            'progress/batch': batch_idx,
            'progress/epoch_progress': progress * 100,
            'progress/total_tokens': tokens_processed,
            'performance/tokens_per_second': tokens_per_second,
            'performance/elapsed_time_hours': elapsed_time / 3600,
        }
        
        self.log(metrics, step=step)
    
    def finish(self):
        """Finish wandb run."""
        wandb.finish()


class MetricsTracker:
    """Simple metrics tracker for offline logging."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.steps = []
    
    def add(self, metrics: Dict[str, float], step: int):
        """Add metrics for a step."""
        self.steps.append(step)
        for key, value in metrics.items():
            self.metrics[key].append(value)
    
    def get_average(self, key: str, last_n: int = 100) -> float:
        """Get average of last n values for a metric."""
        if key not in self.metrics:
            return 0.0
        values = self.metrics[key][-last_n:]
        return np.mean(values) if values else 0.0
    
    def get_latest(self, key: str) -> float:
        """Get latest value for a metric."""
        if key not in self.metrics or not self.metrics[key]:
            return 0.0
        return self.metrics[key][-1]
    
    def summary(self) -> Dict[str, float]:
        """Get summary statistics for all metrics."""
        summary = {}
        for key, values in self.metrics.items():
            if values:
                summary[f'{key}_mean'] = np.mean(values)
                summary[f'{key}_std'] = np.std(values)
                summary[f'{key}_min'] = np.min(values)
                summary[f'{key}_max'] = np.max(values)
                summary[f'{key}_latest'] = values[-1]
        return summary
