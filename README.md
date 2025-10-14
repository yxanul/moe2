# MoE Transformer Training Project

A state-of-the-art Mixture of Experts (MoE) Transformer implementation with advanced features for efficient and stable training.

## Features

### Architecture
- **16-layer Transformer** with d_model=1024
- **64 experts** distributed across GPUs with top-k=2 routing
- **SwiGLU activation** in expert networks (intermediate_dim=512)
- **Grouped Query Attention (GQA)** with tied QKV projections
- **Gated Attention** at G1 position (reduces attention sink from 46.7% to 4.8%)
- **Rotary Position Embeddings (RoPE)** instead of learned position embeddings
- **QK-Normalization** for stable BF16 training (eps=1e-5)
- **RMSNorm** for layer normalization
- **Weight tying** between embeddings and language model head

### Training Features
- **8-bit AdamW optimizer** from bitsandbytes for memory efficiency
- **Mixed precision training** with BF16/FP16 support
- **Distributed training** support via Microsoft Tutel
- **Streaming dataset** loading from HuggingFace FineWeb-Edu (10BT sample)
- **GPT-2 tokenizer** with vocab padding to 50304 (next multiple of 128)
- **Comprehensive logging** with Weights & Biases (wandb)

### Monitoring & Metrics
- Expert utilization statistics (entropy, load balancing coefficient)
- Routing metrics per layer
- Auxiliary loss tracking for load balancing
- Token throughput and memory usage
- Gradient norms by parameter group

## Project Structure

```
moe_training/
├── configs/
│   └── train_config.yaml       # Training configuration
├── src/
│   ├── model/
│   │   ├── moe_transformer.py  # Main MoE transformer
│   │   ├── attention.py        # GQA with QK-Norm
│   │   ├── experts.py          # SwiGLU expert implementation
│   │   └── layers.py           # RMSNorm and utilities
│   ├── data/
│   │   └── dataset.py          # FineWeb-Edu dataset loader
│   └── utils/
│       └── logging.py          # Wandb logging utilities
├── train.py                    # Main training script
├── setup_tutel.sh             # Tutel installation script
├── run_training.sh            # Training launch script
└── pyproject.toml             # Project dependencies
```

## Installation

### Prerequisites
- Python 3.10+
- CUDA 11.8+ with development tools (required for Tutel)
- uv (Python package manager)

### Setup

1. Install dependencies with uv:
```bash
uv sync
```

2. **Required**: Install Microsoft Tutel for MoE:
```bash
# Set CUDA path (adjust if your CUDA is installed elsewhere)
export CUDA_HOME=/usr/local/cuda
./setup_tutel.sh
```

**Note**: Tutel is required for MoE functionality. There is no fallback implementation.

3. Alternatively, install all dependencies manually:
```bash
# Install Tutel (required)
export CUDA_HOME=/usr/local/cuda
uv pip install --no-build-isolation git+https://github.com/microsoft/tutel@main

# Dependencies are already in pyproject.toml and installed via uv sync
```

## Configuration

Edit `configs/train_config.yaml` to customize training parameters:

### Key Configuration Options

- **Model size**: Adjust `d_model`, `n_layers`, `n_heads`
- **Expert configuration**: Modify `num_experts`, `top_k`, `expert_intermediate_dim`
- **Training**: Set `batch_size`, `learning_rate`, `max_steps`
- **Logging**: Configure wandb project and logging frequency
- **Hardware**: Adjust for your GPU setup in `distributed` section

### Example Configuration Changes

For smaller scale testing:
```yaml
model:
  n_layers: 8          # Reduce layers
  num_experts: 16      # Fewer experts
  d_model: 512         # Smaller model dimension

training:
  batch_size: 4        # Smaller batch size
  max_steps: 10000     # Shorter training
```

## Training

### Single GPU Training
```bash
python train.py --config configs/train_config.yaml
```

### Multi-GPU Training
```bash
./run_training.sh
```

Or manually with torchrun:
```bash
torchrun --nproc_per_node=4 train.py --config configs/train_config.yaml
```

### Distributed Training Across Nodes
```bash
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.1" \
    --master_port=29500 \
    train.py --config configs/train_config.yaml
```

## Monitoring

Training metrics are logged to Weights & Biases. Key metrics include:

- **Training Loss & Perplexity**
- **Auxiliary Loss** (for load balancing)
- **Expert Utilization**:
  - Entropy per layer
  - Load coefficient of variation
  - Token distribution across experts
- **Performance Metrics**:
  - Tokens per second
  - GPU memory usage
  - Gradient norms

View your training runs at: https://wandb.ai/YOUR_USERNAME/moe-training

## Model Details

### Attention Mechanism
- **F.scaled_dot_product_attention**: Uses PyTorch's optimized SDPA for Flash Attention support
- **Gated Attention (G1)**: Modulates SDPA output to reduce attention sink and improve stability
- **Rotary Position Embeddings (RoPE)**: Encodes positions via rotation, no learned parameters
- **Grouped Query Attention (GQA)**: Reduces KV cache memory with 8 KV heads for 16 query heads
- **QK-Normalization**: Stabilizes attention in BF16 training with eps=1e-5
- **Tied QKV Projections**: Single GEMM for efficiency
- **Proper Causal Masking**: Ensures autoregressive training with combined causal and padding masks

### Expert Architecture
- **SwiGLU Activation**: `SwiGLU(x) = (x * W_gate) ⊙ SiLU(x * W_up)`
- **Expert Routing**: Top-2 selection with capacity factor 1.0
- **Load Balancing**: Auxiliary loss weight of 0.001

### Tokenization
- **GPT-2 Tokenizer**: 50,257 base vocabulary
- **Padded Vocabulary**: 50,304 (next multiple of 128 for compute efficiency)

## Performance Optimization

### Memory Optimization
- 8-bit AdamW optimizer reduces optimizer memory by 75%
- GQA reduces KV cache memory
- Streaming dataset prevents loading entire dataset into memory

### Compute Optimization
- Vocabulary padding to multiple of 128 improves GEMM efficiency
- TF32 enabled for Ampere GPUs
- Mixed precision training with BF16

### Distributed Training
- Microsoft Tutel for efficient all-to-all communication
- Expert parallelism across GPUs
- Gradient accumulation for larger effective batch sizes

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in config
- Increase `gradient_accumulation_steps`
- Reduce `d_model` or `n_layers`
- Use more GPUs for distributed training

### Slow Training
- Ensure TF32 is enabled for Ampere GPUs
- Check `num_workers` for data loading
- Verify NCCL is being used for multi-GPU
- Consider reducing logging frequency

### Poor Load Balancing
- Increase `aux_loss_weight` (try 0.01)
- Adjust `capacity_factor` (try 1.25)
- Monitor expert utilization in wandb

## Citation

This implementation uses:
- [Microsoft Tutel](https://github.com/microsoft/tutel) for efficient MoE
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) dataset
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) for 8-bit optimization

## License

This project is for research and educational purposes.
