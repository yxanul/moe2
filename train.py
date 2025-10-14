#!/usr/bin/env python3

"""
Main training script for MoE Transformer with advanced features.
"""

import os
import sys
import argparse
import time
import yaml
from pathlib import Path
from typing import Dict, Optional
import zlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
import bitsandbytes as bnb

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data.dataset import create_dataloader
from src.utils.logging import WandbLogger, MetricsTracker

# Tutel imports for distributed training
from tutel import system
from tutel import net
from src.model.moe_transformer import MoETransformer


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_distributed():
    """Setup distributed training environment."""
    if torch.cuda.is_available():
        parallel_env = system.init_data_model_parallel(backend='nccl')
    else:
        parallel_env = system.init_data_model_parallel(backend='gloo')
    
    return parallel_env


def create_optimizer(model: nn.Module, config: Dict):
    """Create AdamW 8-bit optimizer from bitsandbytes."""
    optimizer_config = config['optimizer']
    training_config = config['training']
    
    # Separate parameters that should not be decayed
    no_decay = ["bias", "LayerNorm.weight", "layer_norm", "ln", "norm"]
    wd = float(training_config['weight_decay'])
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": wd,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]
    
    # Create 8-bit AdamW optimizer
    lr = float(training_config['learning_rate'])
    betas = tuple(float(x) for x in optimizer_config['betas'])
    eps = float(optimizer_config['eps'])

    optimizer = bnb.optim.AdamW8bit(
        optimizer_grouped_parameters,
        lr=lr,
        betas=betas,
        eps=eps,
    )
    
    return optimizer


def get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1
):
    """Create a schedule with a learning rate that decreases linearly after warmup."""
    
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) /
            float(max(1, num_training_steps - num_warmup_steps))
        )
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def train_step(
    model: nn.Module,
    batch: Dict,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    config: Dict,
    parallel_env,
) -> Dict:
    """Perform a single training step."""
    
    model.train()
    
    # Move batch to device
    device = parallel_env.local_device
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    
    # Forward pass with mixed precision
    with torch.amp.autocast('cuda', dtype=torch.bfloat16 if config['training']['bf16'] else torch.float16):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        loss = outputs['loss']
        
        # Scale loss for gradient accumulation
        loss = loss / config['training']['gradient_accumulation_steps']
    
    # Backward pass
    scaler.scale(loss).backward()
    
    # Collect metrics
    metrics = {
        'loss': loss.item() * config['training']['gradient_accumulation_steps'],
        'aux_loss': outputs['aux_loss'] if isinstance(outputs['aux_loss'], float) else outputs['aux_loss'].item(),
    }
    
    return metrics, outputs.get('routing_stats', [])


def main():
    """Main training function."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train MoE Transformer')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/train_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)

    # Enable TF32 matmul if requested
    if config.get('training', {}).get('tf32', False):
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass
    
    # Set random seed
    torch.manual_seed(config['training']['seed'])
    
    # Setup distributed training
    parallel_env = setup_distributed()
    dist_rank = parallel_env.global_rank
    dist_world_size = parallel_env.global_size
    device = parallel_env.local_device
    
    # Only main process handles logging
    is_main_process = dist_rank == 0
    
    if is_main_process:
        print(f"Starting training with {dist_world_size} GPUs...")
        print(f"Configuration: {config}")
    
    # Initialize wandb logger (only on main process)
    if is_main_process:
        logger = WandbLogger(
            project=config['wandb']['project'],
            name=config['wandb']['name'],
            config=config,
            tags=config['wandb']['tags'],
            mode=config['wandb']['mode'],
        )
    else:
        logger = None
    
    # Create model
    model = MoETransformer(
        vocab_size=config['model']['vocab_size'],
        d_model=config['model']['d_model'],
        n_layers=config['model']['n_layers'],
        n_heads=config['model']['n_heads'],
        n_kv_heads=config['model']['n_kv_heads'],
        num_experts=config['model']['num_experts'],
        expert_intermediate_dim=config['model']['expert_intermediate_dim'],
        top_k=config['model']['top_k'],
        capacity_factor=config['model']['capacity_factor'],
        aux_loss_weight=config['model']['aux_loss_weight'],
        qk_norm_eps=config['model']['qk_norm_eps'],
        rms_norm_eps=config['model']['rms_norm_eps'],
        dropout=config['model']['dropout'],
        max_seq_len=config['model']['max_seq_len'],
        tie_word_embeddings=config['model']['tie_word_embeddings'],
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expert_params = sum(
        p.numel() for n, p in model.named_parameters()
        if 'expert' in n.lower() or 'ffn' in n.lower()
    )
    
    if is_main_process:
        print(f"Model initialized with {total_params / 1e6:.2f}M parameters")
        print(f"  Trainable: {trainable_params / 1e6:.2f}M")
        print(f"  Expert params: {expert_params / 1e6:.2f}M")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['training']['warmup_steps'],
        num_training_steps=config['training']['max_steps'],
    )
    
    # Create gradient scaler for mixed precision
    scaler = torch.amp.GradScaler('cuda', enabled=config['training']['fp16'])
    
    # Create dataloader
    if is_main_process:
        print("Creating dataloader...")
    
    dataloader, tokenizer = create_dataloader(
        batch_size=config['training']['batch_size'],
        max_seq_len=config['dataset']['max_seq_len'],
        num_workers=config['dataset']['num_workers'],
        prefetch_factor=config['dataset']['prefetch_factor'],
    )
    
    # Training loop
    if is_main_process:
        print("Starting training loop...")
    
    global_step = 0
    accumulation_steps = config['training']['gradient_accumulation_steps']
    start_time = time.time()
    tokens_processed = 0
    
    # Metrics tracker
    metrics_tracker = MetricsTracker()
    
    # Get parameters that need all-reduce
    params_for_all_reduce = [
        p for p in model.parameters()
        if not hasattr(p, 'skip_allreduce') and p.requires_grad
    ]
    
    for epoch in range(100):  # Large number, will break on max_steps
        # Inform streaming dataset about current epoch for per-epoch reshuffle
        try:
            if hasattr(dataloader, 'dataset') and hasattr(dataloader.dataset, 'set_epoch'):
                dataloader.dataset.set_epoch(epoch)
        except Exception:
            pass

        for batch_idx, batch in enumerate(dataloader):
            # Training step
            step_metrics, routing_stats = train_step(
                model, batch, optimizer, scheduler, scaler, config, parallel_env
            )
            
            # Track metrics
            metrics_tracker.add(step_metrics, global_step)
            
            # Accumulate gradients
            if (batch_idx + 1) % accumulation_steps == 0:
                # Clip gradients
                if config['training']['grad_clip'] > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config['training']['grad_clip']
                    )
                
                # All-reduce gradients across GPUs
                if dist_world_size > 1:
                    for p in params_for_all_reduce:
                        if p.grad is not None:
                            p.grad = net.simple_all_reduce(p.grad) / dist_world_size
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                tokens_processed += (
                    config['training']['batch_size'] *
                    config['dataset']['max_seq_len'] *
                    accumulation_steps *
                    dist_world_size
                )
                
                # Logging
                if global_step % config['training']['logging_steps'] == 0 and is_main_process:
                    elapsed_time = time.time() - start_time
                    tokens_per_second = tokens_processed / elapsed_time
                    
                    # Get average metrics
                    avg_loss = metrics_tracker.get_average('loss', last_n=100)
                    avg_aux_loss = metrics_tracker.get_average('aux_loss', last_n=100)
                    
                    # Log to wandb
                    if logger:
                        logger.log_moe_metrics(
                            loss=avg_loss,
                            aux_loss=avg_aux_loss,
                            routing_stats=routing_stats,
                            step=global_step,
                            learning_rate=scheduler.get_last_lr()[0],
                        )
                        
                        logger.log_training_progress(
                            epoch=epoch,
                            batch_idx=batch_idx,
                            total_batches=100000,  # Placeholder
                            tokens_processed=tokens_processed,
                            elapsed_time=elapsed_time,
                            step=global_step,
                        )
                        
                        # Log memory stats
                        if global_step % 100 == 0:
                            logger.log_memory_stats(step=global_step)
                            logger.log_gradient_stats(model, step=global_step)
                    
                    # Print progress
                    # Lightweight batch fingerprint to detect repetition
                    try:
                        batch_crc32 = zlib.crc32(batch['input_ids'].detach().cpu().numpy().tobytes())
                    except Exception:
                        batch_crc32 = -1

                    print(
                        f"Step {global_step} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Aux Loss: {avg_aux_loss:.6f} | "
                        f"LR: {scheduler.get_last_lr()[0]:.6f} | "
                        f"Tokens/s: {tokens_per_second:.0f} | "
                        f"batch_crc32: {batch_crc32}"
                    )
                
                # Save checkpoint
                if global_step % config['training']['save_steps'] == 0 and is_main_process:
                    checkpoint_dir = Path(config['checkpoint']['save_dir'])
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    
                    checkpoint_path = checkpoint_dir / f"checkpoint-{global_step}.pt"
                    torch.save({
                        'step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'config': config,
                    }, checkpoint_path)
                    
                    print(f"Saved checkpoint to {checkpoint_path}")
                
                # Check if training is complete
                if global_step >= config['training']['max_steps']:
                    if is_main_process:
                        print(f"Training complete! Reached {global_step} steps.")
                        if logger:
                            logger.finish()
                    return
    
    if is_main_process:
        print("Training complete!")
        if logger:
            logger.finish()


if __name__ == "__main__":
    main()
