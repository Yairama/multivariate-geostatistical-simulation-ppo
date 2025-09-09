# MineRL-NPV: Deep Reinforcement Learning for Mining Optimization

A reinforcement learning system for optimizing mining operations using PPO (Proximal Policy Optimization) on multivariate geostatistical data.

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Training a Model
```bash
# Basic training (headless mode)
python train_model.py --config mine_rl_npv/configs/train.yaml --data mine_rl_npv/data/sample_model.csv

# Training with 3D visualization
python train_model.py --config mine_rl_npv/configs/train.yaml --data mine_rl_npv/data/sample_model.csv --visualization

# Custom training with specific settings
python train_model.py --config mine_rl_npv/configs/train.yaml --data mine_rl_npv/data/sample_model.csv --timesteps 50000 --device cuda
```

### Evaluating a Trained Model
```bash
# Basic evaluation (headless mode)
python evaluate_model.py --model path/to/best_model.zip --data mine_rl_npv/data/sample_model.csv

# Evaluation with visualization and comparison
python evaluate_model.py --model path/to/best_model.zip --data mine_rl_npv/data/sample_model.csv --visualization --compare --plot
```

## 📁 Project Structure

```
multivariate-geostatistical-simulation-ppo/
├── train_model.py          # Main training script (headless/visualization modes)
├── evaluate_model.py       # Model evaluation script (headless/visualization modes)
├── requirements.txt        # Python dependencies
├── mine_rl_npv/           # Core package
│   ├── configs/           # Configuration files
│   ├── data/             # Sample datasets
│   ├── envs/             # Gymnasium environments
│   ├── rl/               # Reinforcement learning components
│   └── viz/              # Visualization tools
└── README.md             # This file
```

## 🎯 Features

- **PPO-based RL Agent**: Uses Proximal Policy Optimization for stable training
- **3D Mining Environment**: Gymnasium-compatible environment with 3D block models
- **Economic Optimization**: Optimizes Net Present Value (NPV) considering copper and molybdenum grades
- **Flexible Modes**: Supports both headless (server) and visualization (development) modes
- **GPU/CPU Compatible**: Automatic device detection with fallback to CPU
- **TensorBoard Support**: Built-in logging and monitoring capabilities

## 🔧 Configuration

Configuration files are located in `mine_rl_npv/configs/`. The main training configuration (`train.yaml`) includes:

- **Environment settings**: Grid dimensions, action spaces, reward functions
- **Model architecture**: CNN3D feature extractor, policy networks
- **Training parameters**: Learning rates, batch sizes, exploration settings
- **Hardware optimization**: Memory management for different GPU/RAM configurations

## 📊 Data Format

The system expects CSV files with block model data containing:
- **Spatial coordinates**: X, Y, Z positions
- **Geological features**: Rock types, structural data
- **Mineral grades**: Copper, molybdenum concentrations
- **Economic parameters**: Processing costs, recovery rates

## 🚀 Performance

- **Model size**: ~753K parameters CNN3D architecture
- **Training data**: 153K+ blocks across 49×71×58 grid
- **Memory requirements**: Optimized for 16GB VRAM systems
- **Action space**: ~3,479 possible actions with dynamic masking

## 📈 Monitoring

Training progress can be monitored using TensorBoard:
```bash
tensorboard --logdir experiments/runs
```

## 🛠️ Development

For development and debugging:
- Use `--verbose 2` for detailed logging
- Use `--visualization` for interactive 3D plots
- Monitor GPU memory usage with smaller timestep values for testing