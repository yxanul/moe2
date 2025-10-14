import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMSNorm implementation for layer normalization."""
    
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


def next_multiple_of_n(value, n=128):
    """Round up to the next multiple of n for better compute efficiency."""
    return ((value + n - 1) // n) * n


class SwiGLU(nn.Module):
    """SwiGLU activation function as used in LLaMA."""
    
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.w_gate = nn.Linear(dim_in, dim_out, bias=False)
        self.w_up = nn.Linear(dim_in, dim_out, bias=False)
        
    def forward(self, x):
        return F.silu(self.w_gate(x)) * self.w_up(x)
