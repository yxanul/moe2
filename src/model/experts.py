import torch
import torch.nn as nn
import torch.nn.functional as F
from tutel import moe as tutel_moe
from tutel import system


class SwiGLUExpert(nn.Module):
    """SwiGLU expert implementation for MoE."""
    
    def __init__(self, d_model, intermediate_dim, num_local_experts):
        super().__init__()
        self.num_local_experts = num_local_experts
        self.d_model = d_model
        self.intermediate_dim = intermediate_dim
        
        # Batched weights for all local experts
        # Gate and up projections
        self.w_gate = nn.Parameter(
            torch.randn(num_local_experts, d_model, intermediate_dim) * 0.02
        )
        self.w_up = nn.Parameter(
            torch.randn(num_local_experts, d_model, intermediate_dim) * 0.02
        )
        # Down projection
        self.w_down = nn.Parameter(
            torch.randn(num_local_experts, intermediate_dim, d_model) * 0.02
        )
        
        # Mark parameters to skip allreduce (local to each GPU)
        for param in self.parameters():
            setattr(param, 'skip_allreduce', True)
    
    def forward(self, x, ctx=None):
        """
        x: [num_local_experts, capacity, d_model]
        """
        # Gate and up projections
        gate = torch.bmm(x, self.w_gate)  # [E, C, intermediate_dim]
        up = torch.bmm(x, self.w_up)      # [E, C, intermediate_dim]
        
        # SwiGLU activation
        hidden = F.silu(gate) * up
        
        # Down projection
        output = torch.bmm(hidden, self.w_down)  # [E, C, d_model]
        
        return output


def build_swiglu_expert(
    model_dim,
    num_experts_per_device=None,
    num_local_experts=None,
    hidden_size_per_expert=None,
    intermediate_dim=None,
    **kwargs,
):
    """Factory for Tutel 'custom' experts."""
    nle = num_local_experts if num_local_experts is not None else num_experts_per_device
    inter = intermediate_dim if intermediate_dim is not None else hidden_size_per_expert
    return SwiGLUExpert(model_dim, inter, nle)


class MoELayer(nn.Module):
    """Mixture of Experts layer using Tutel."""
    
    def __init__(
        self,
        d_model,
        num_experts=64,
        expert_intermediate_dim=512,
        top_k=2,
        capacity_factor=1.0,
        aux_loss_weight=0.001,
        dropout=0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.aux_loss_weight = aux_loss_weight
        
        # Initialize parallel environment
        if not hasattr(system, '_parallel_env_initialized'):
            self.parallel_env = system.init_data_model_parallel()
            system._parallel_env_initialized = True
        else:
            self.parallel_env = system.get_local_session()
        
        self.dist_rank = self.parallel_env.global_rank
        self.dist_world_size = self.parallel_env.global_size
        
        # Calculate local experts per GPU
        self.num_local_experts = num_experts // self.dist_world_size
        if num_experts % self.dist_world_size != 0:
            raise ValueError(f"num_experts ({num_experts}) must be divisible by world_size ({self.dist_world_size})")
        
        # Create MoE layer using Tutel
        self._moe = tutel_moe.moe_layer(
            gate_type={'type': 'top', 'k': top_k, 'capacity_factor': capacity_factor, 'fp32_gate': True},
            experts={
                'type': 'custom',
                'module': build_swiglu_expert,
                'num_experts_per_device': self.num_local_experts,
                'hidden_size_per_expert': expert_intermediate_dim,
            },
            model_dim=d_model,
            scan_expert_func=lambda name, param: setattr(param, 'skip_allreduce', True),
            seeds=(1, self.dist_rank + 1, 1),
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass through MoE layer.
        Returns: (output, aux_loss, routing_stats)
        """
        # x shape: [batch_size, seq_len, d_model]
        output = self._moe(x)
        
        # Get auxiliary loss for load balancing
        aux_val = self._moe.l_aux if hasattr(self._moe, 'l_aux') else 0.0
        if isinstance(aux_val, torch.Tensor):
            try:
                aux_loss = float(aux_val.detach().float().cpu().item())
            except Exception:
                aux_loss = float(aux_val.item())
        else:
            aux_loss = float(aux_val)
        
        # Apply dropout
        output = self.dropout(output)
        
        # Collect routing statistics
        routing_stats = {
            'aux_loss': float(aux_loss),
            'load_balancing_loss': float(aux_loss * self.aux_loss_weight),
        }
        
        return output, aux_loss, routing_stats
