import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from .layers import RMSNorm


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention with QK-Norm and tied QKV projections."""
    
    def __init__(
        self,
        d_model,
        n_heads=8,
        n_kv_heads=None,
        head_dim=None,
        qk_norm_eps=1e-5,
        dropout=0.0,
        max_seq_len=2048,
        use_gating=True,
        gating_position='g1',  # 'g1', 'g2', 'g3', 'g4', 'g5', or None
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.head_dim = head_dim if head_dim is not None else d_model // n_heads
        
        # Validate head divisibility
        assert self.n_heads % self.n_kv_heads == 0, \
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        
        self.n_rep = self.n_heads // self.n_kv_heads
        self.dropout = dropout
        
        # Single GEMM for QKV projection (tied)
        self.qkv_proj = nn.Linear(
            d_model,
            (self.n_heads + 2 * self.n_kv_heads) * self.head_dim,
            bias=False
        )
        
        # QK-Norm for stability in BF16 training
        self.q_norm = RMSNorm(self.head_dim, eps=qk_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=qk_norm_eps)
        
        # Output projection
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim,
            d_model,
            bias=False
        )
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Precompute scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Gated Attention configuration
        self.use_gating = use_gating
        self.gating_position = gating_position.lower() if gating_position else None
        
        if self.use_gating and self.gating_position:
            if self.gating_position == 'g1':
                # G1: Gate after SDPA (most effective according to paper)
                self.gate_proj = nn.Linear(d_model, self.n_heads, bias=True)
            elif self.gating_position == 'g2':
                # G2: Gate on V
                self.gate_proj = nn.Linear(d_model, self.n_kv_heads, bias=True)
            elif self.gating_position in ['g3', 'g4']:
                # G3: Gate on K, G4: Gate on Q
                gate_heads = self.n_kv_heads if self.gating_position == 'g3' else self.n_heads
                self.gate_proj = nn.Linear(d_model, gate_heads, bias=True)
            elif self.gating_position == 'g5':
                # G5: Gate after output projection
                self.gate_proj = nn.Linear(d_model, d_model, bias=True)
            else:
                raise ValueError(f"Unknown gating position: {self.gating_position}")
            
            # Initialize gate projection for stability
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.ones_(self.gate_proj.bias)  # Start with gates mostly open
        
        # Rotary Position Embeddings (RoPE)
        self._init_rope(max_seq_len)
    
    def _init_rope(self, max_seq_len, base=10000):
        """Initialize Rotary Position Embeddings."""
        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        
        # Build position indices
        t = torch.arange(max_seq_len, dtype=torch.float32)
        
        # Compute frequencies [max_seq_len, head_dim/2]
        freqs = torch.outer(t, inv_freq)
        
        # Cache cos and sin values with shape [1, 1, max_seq_len, head_dim/2]
        cos_cached = torch.cos(freqs).unsqueeze(0).unsqueeze(0)
        sin_cached = torch.sin(freqs).unsqueeze(0).unsqueeze(0)
        
        # Register as buffers (not persistent to avoid saving in checkpoints)
        self.register_buffer('cos_cached', cos_cached, persistent=False)
        self.register_buffer('sin_cached', sin_cached, persistent=False)
    
    @staticmethod
    def apply_rotary_emb(x, cos, sin):
        """Apply rotary embeddings to input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, n_heads, seq_len, head_dim]
            cos: Cosine cache of shape [1, 1, seq_len, head_dim/2]
            sin: Sine cache of shape [1, 1, seq_len, head_dim/2]
        """
        # Split the last dimension into two halves for rotation
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        
        # Apply rotation using complex number multiplication formula
        # (a + ib) * (cos θ + i sin θ) = (a cos θ - b sin θ) + i(a sin θ + b cos θ)
        rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
        
        return rotated
        
    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Store original input for gating computation (pre-norm input)
        x_for_gate = x
        
        # Single GEMM for QKV
        qkv = self.qkv_proj(x)
        
        # Split into Q, K, V
        q_size = self.n_heads * self.head_dim
        k_size = self.n_kv_heads * self.head_dim
        v_size = self.n_kv_heads * self.head_dim
        
        q, k, v = torch.split(qkv, [q_size, k_size, v_size], dim=-1)
        
        # Reshape and transpose
        q = rearrange(q, 'b s (h d) -> b h s d', h=self.n_heads)
        k = rearrange(k, 'b s (h d) -> b h s d', h=self.n_kv_heads)
        v = rearrange(v, 'b s (h d) -> b h s d', h=self.n_kv_heads)
        
        # Apply QK-Norm for stability
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Apply gating at G4 (Query) or G3 (Key) positions if configured
        if self.use_gating and self.gating_position == 'g4':
            gate = torch.sigmoid(self.gate_proj(x_for_gate))  # [B, S, n_heads]
            gate = rearrange(gate, 'b s h -> b h s 1')
            q = q * gate
        elif self.use_gating and self.gating_position == 'g3':
            gate = torch.sigmoid(self.gate_proj(x_for_gate))  # [B, S, n_kv_heads]
            gate = rearrange(gate, 'b s h -> b h s 1')
            k = k * gate
        elif self.use_gating and self.gating_position == 'g2':
            gate = torch.sigmoid(self.gate_proj(x_for_gate))  # [B, S, n_kv_heads]
            gate = rearrange(gate, 'b s h -> b h s 1')
            v = v * gate
        
        # Apply Rotary Position Embeddings
        # Slice caches to current sequence length
        cos = self.cos_cached[..., :seq_len, :].to(q.dtype)
        sin = self.sin_cached[..., :seq_len, :].to(q.dtype)
        
        # Apply RoPE to q and k (but not v)
        q = self.apply_rotary_emb(q, cos, sin)
        k = self.apply_rotary_emb(k, cos, sin)
        
        # Repeat k and v for grouped query attention
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        
        # Use scaled_dot_product_attention for efficient computation
        # Always apply causal mask for autoregressive modeling
        # attention_mask (if provided) handles padding
        if attention_mask is not None and attention_mask.dim() == 4:
            # Combine causal mask with provided attention mask (for padding)
            # Create causal mask
            causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).triu(diagonal=1)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
            causal_mask = causal_mask.expand(batch_size, self.n_heads, -1, -1)
            
            # Convert causal mask to additive format
            causal_additive = causal_mask.to(x.dtype) * torch.finfo(x.dtype).min
            
            # Combine with attention mask (which already handles padding)
            combined_mask = attention_mask + causal_additive
            
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=combined_mask,
                dropout_p=self.dropout if self.training else 0.0,
                scale=self.scale,
            )
        else:
            # No attention mask provided, just use causal
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,  # Let SDPA handle causal mask efficiently
                scale=self.scale,
            )
        
        # Apply gated attention at G1 position (after SDPA, before output projection)
        if self.use_gating and self.gating_position == 'g1':
            # Compute gate from pre-norm input
            gate = torch.sigmoid(self.gate_proj(x_for_gate))  # [B, S, n_heads]
            gate = rearrange(gate, 'b s h -> b h s 1')  # [B, n_heads, S, 1]
            # Apply gate to attention output
            attn_output = attn_output * gate  # [B, n_heads, S, head_dim]
        
        # Reshape and project output
        attn_output = rearrange(attn_output, 'b h s d -> b s (h d)')
        output = self.o_proj(attn_output)
        
        # Apply gating at G5 position (after output projection) if configured
        if self.use_gating and self.gating_position == 'g5':
            gate = torch.sigmoid(self.gate_proj(x_for_gate))  # [B, S, d_model]
            output = output * gate
        
        output = self.resid_dropout(output)
        
        return output
