import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from .layers import RMSNorm, next_multiple_of_n
from .attention import GroupedQueryAttention
from .experts import MoELayer


class TransformerBlock(nn.Module):
    """Transformer block with GQA attention and MoE FFN."""
    
    def __init__(
        self,
        d_model,
        n_heads,
        n_kv_heads,
        num_experts,
        expert_intermediate_dim,
        top_k,
        capacity_factor,
        aux_loss_weight,
        qk_norm_eps=1e-5,
        rms_norm_eps=1e-6,
        dropout=0.0,
        max_seq_len=2048,
    ):
        super().__init__()
        
        # Pre-normalization layers
        self.attention_norm = RMSNorm(d_model, eps=rms_norm_eps)
        self.ffn_norm = RMSNorm(d_model, eps=rms_norm_eps)
        
        # Attention layer with gated attention (G1 position - most effective per paper)
        self.attention = GroupedQueryAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            qk_norm_eps=qk_norm_eps,
            dropout=dropout,
            max_seq_len=max_seq_len,
            use_gating=True,
            gating_position='g1',  # After SDPA - reduces attention sink and improves stability
        )
        
        # MoE FFN layer
        self.ffn = MoELayer(
            d_model=d_model,
            num_experts=num_experts,
            expert_intermediate_dim=expert_intermediate_dim,
            top_k=top_k,
            capacity_factor=capacity_factor,
            aux_loss_weight=aux_loss_weight,
            dropout=dropout,
        )
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        # Attention with residual connection
        residual = x
        x = self.attention_norm(x)
        x = self.attention(x, attention_mask)
        x = residual + x
        
        # MoE FFN with residual connection
        residual = x
        x = self.ffn_norm(x)
        ffn_out, aux_loss, routing_stats = self.ffn(x)
        x = residual + ffn_out
        
        return x, {'aux_loss': aux_loss, 'routing_stats': routing_stats}


class MoETransformer(nn.Module):
    """Full MoE Transformer model with 16 layers."""
    
    def __init__(
        self,
        vocab_size=50257,
        d_model=1024,
        n_layers=16,
        n_heads=16,
        n_kv_heads=8,
        num_experts=64,
        expert_intermediate_dim=512,
        top_k=2,
        capacity_factor=1.0,
        aux_loss_weight=0.001,
        qk_norm_eps=1e-5,
        rms_norm_eps=1e-6,
        dropout=0.0,
        max_seq_len=2048,
        tie_word_embeddings=True,
    ):
        super().__init__()
        
        # Pad vocab size to next multiple of 128 for better compute efficiency
        self.vocab_size = next_multiple_of_n(vocab_size, n=128)  # 50304 for GPT-2
        self.d_model = d_model
        self.n_layers = n_layers
        self.tie_word_embeddings = tie_word_embeddings
        self.aux_loss_weight = aux_loss_weight  # Store aux_loss_weight
        
        # Token embeddings
        self.embed = nn.Embedding(self.vocab_size, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                num_experts=num_experts,
                expert_intermediate_dim=expert_intermediate_dim,
                top_k=top_k,
                capacity_factor=capacity_factor,
                aux_loss_weight=aux_loss_weight,
                qk_norm_eps=qk_norm_eps,
                rms_norm_eps=rms_norm_eps,
                dropout=dropout,
                max_seq_len=max_seq_len,
            )
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.ln_f = RMSNorm(d_model, eps=rms_norm_eps)
        
        # Language modeling head
        if tie_word_embeddings:
            self.lm_head = nn.Linear(d_model, self.vocab_size, bias=False)
            # Weight tying
            self.lm_head.weight = self.embed.weight
        else:
            self.lm_head = nn.Linear(d_model, self.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token indices [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Target token indices for language modeling [batch_size, seq_len]
            
        Returns:
            Dictionary with loss, logits, and auxiliary information
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token embeddings
        token_embeds = self.embed(input_ids)
        
        # Apply dropout to embeddings
        x = self.dropout(token_embeds)
        
        # Process attention mask for padding only
        # SDPA will handle causal masking via is_causal parameter
        if attention_mask is not None:
            # Build boolean padding mask for SDPA [B, 1, S, S]; True indicates masked positions
            pad_mask = (attention_mask == 0)
            pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
            pad_mask = pad_mask.expand(-1, -1, seq_len, -1)  # [B, 1, S, S]
            attention_mask = pad_mask
        
        # Pass through transformer layers
        total_aux_loss = 0.0
        all_routing_stats = []
        
        for layer in self.layers:
            x, layer_stats = layer(x, attention_mask)
            total_aux_loss += layer_stats['aux_loss']
            all_routing_stats.append(layer_stats['routing_stats'])
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Language modeling head
        logits = self.lm_head(x)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for loss calculation
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
            
            # Add auxiliary loss for load balancing
            loss = loss + self.aux_loss_weight * total_aux_loss
        
        return {
            'loss': loss,
            'logits': logits,
            'aux_loss': total_aux_loss,
            'routing_stats': all_routing_stats,
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.95,
    ) -> torch.Tensor:
        """Simple generation method for inference."""
        self.eval()
        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                outputs = self.forward(input_ids)
                logits = outputs['logits'][:, -1, :] / temperature
                
                # Top-p sampling
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                # Fix: Create a new tensor for scatter operation
                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
                indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
        return input_ids
