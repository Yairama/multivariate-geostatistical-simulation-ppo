# Memory Optimization Guide for MineRL-NPV

## Problem Description

The original error:
```
❌ Training failed with error: Unable to allocate 105. GiB for an array with shape (2048, 4, 17, 49, 71, 58) and data type float32
```

This occurs because the default configuration requires:
- **Memory**: ~80-230 GB RAM
- **Batch size**: 2048 steps × 4 environments × 17 channels × 49×71×58 spatial dimensions
- **Model complexity**: Large 3D CNN with 128+ channels

## Available Memory-Optimized Configurations

### 1. 🔧 Memory Optimized (4.3 GB)
**Recommended for**: Mid-range GPUs with 8-16GB VRAM

```bash
python3 train_model.py --config mine_rl_npv/configs/train_memory_optimized.yaml --data mine_rl_npv/data/sample_model.csv --visualization
```

**Optimizations**:
- Reduced batch size: 512 steps (4x smaller)
- Fewer environments: 2 parallel envs (2x smaller)
- Smaller CNN: 16→32→64 channels instead of 32→64→128
- Smaller networks: 128×128 instead of 256×256
- Smaller grid: 50×50×30 instead of 100×100×50

### 2. 🚀 Ultra Light (0.02 GB)
**Recommended for**: CPU training or GPUs with <8GB VRAM

```bash
python3 train_model.py --config mine_rl_npv/configs/train_ultra_light.yaml --data mine_rl_npv/data/sample_model.csv --visualization
```

**Optimizations**:
- Minimal batch size: 128 steps (16x smaller)
- Single environment: 1 env only
- Tiny CNN: CNN3DTiny with 8→16 channels
- Small networks: 64×64
- Very small grid: 20×20×10

### 3. 🧪 Small (Testing - 0.01 GB)
**Recommended for**: Quick testing and development

```bash
python3 train_model.py --config examples/train_small.yaml --data mine_rl_npv/data/sample_model.csv --visualization
```

## Memory Helper Tool

Use the included memory helper to choose the right configuration:

```bash
# See all configurations and their memory requirements
python3 memory_helper.py --compare

# Get recommendation for your available memory
python3 memory_helper.py --memory 8    # For 8GB
python3 memory_helper.py --memory 16   # For 16GB
python3 memory_helper.py --memory 32   # For 32GB
```

## Configuration Details

### Memory Usage Comparison

| Configuration | Memory (GB) | Grid Size | Steps | Envs | Reduction |
|---------------|-------------|-----------|-------|------|-----------|
| Full          | 228.88      | 100×100×50| 2048  | 4    | 1x        |
| Memory Opt.   | 4.29        | 50×50×30  | 512   | 2    | 53x       |
| Ultra Light   | 0.02        | 20×20×10  | 128   | 1    | 4163x     |
| Small         | 0.01        | 10×10×5   | 128   | 2    | 6000x+    |

### Performance vs Memory Trade-offs

1. **Full Configuration**: Best learning performance, highest memory
2. **Memory Optimized**: Good balance of performance and memory usage
3. **Ultra Light**: Basic learning capability, minimal memory
4. **Small**: Testing only, synthetic data

## Technical Implementation

### Feature Extractor Variants

The memory optimizations use different CNN architectures:

- **CNN3DFeatureExtractor**: Full 3D CNN (32→64→128 channels)
- **CNN3DFeatureExtractorSmall**: Reduced CNN (16→32→64 channels) 
- **CNN3DFeatureExtractorTiny**: Minimal CNN (8→16 channels)

### Environment Grid Scaling

- **Full**: Uses real block model dimensions (49×71×58)
- **Memory Optimized**: Reduced grid (50×50×30) 
- **Ultra Light**: Small synthetic grid (20×20×10)
- **Small**: Minimal synthetic grid (10×10×5)

## Troubleshooting

### If you still get memory errors:

1. **Check your system memory**:
   ```bash
   free -h  # Linux
   ```

2. **Use even smaller configuration**:
   ```bash
   python3 train_model.py --config examples/train_small.yaml --data mine_rl_npv/data/sample_model.csv --timesteps 1000
   ```

3. **Force CPU usage**:
   ```bash
   python3 train_model.py --config mine_rl_npv/configs/train_ultra_light.yaml --data mine_rl_npv/data/sample_model.csv --device cpu
   ```

4. **Reduce timesteps for testing**:
   ```bash
   python3 train_model.py --config mine_rl_npv/configs/train_ultra_light.yaml --data mine_rl_npv/data/sample_model.csv --timesteps 100
   ```

### Performance Tips

1. **Use GPU when available**: Add `--device cuda` if you have NVIDIA GPU
2. **Monitor memory usage**: Use `htop` or `nvidia-smi` to monitor resource usage
3. **Start small**: Always test with ultra_light first, then scale up
4. **Use visualization mode only when needed**: Add `--visualization` only for debugging

## Command Examples

### Original failing command:
```bash
python3 train_model.py --config mine_rl_npv/configs/train.yaml --data mine_rl_npv/data/sample_model.csv --visualization
```

### Fixed commands (choose based on your memory):

```bash
# For 8GB+ systems
python3 train_model.py --config mine_rl_npv/configs/train_memory_optimized.yaml --data mine_rl_npv/data/sample_model.csv --visualization

# For 4GB systems or CPU-only
python3 train_model.py --config mine_rl_npv/configs/train_ultra_light.yaml --data mine_rl_npv/data/sample_model.csv --visualization

# For quick testing
python3 train_model.py --config examples/train_small.yaml --data mine_rl_npv/data/sample_model.csv --timesteps 1000
```

The memory optimizations reduce memory usage by **18-4000x** while maintaining training functionality.