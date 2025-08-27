# MineRL-NPV Training and Evaluation Scripts

This directory contains standalone scripts for training and evaluating MineRL-NPV models with both headless and visualization modes.

## Scripts Overview

### `train_model.py` - Training Script
Standalone script for training MineRL-NPV models using MaskablePPO.

### `evaluate_model.py` - Evaluation Script  
Standalone script for evaluating trained MineRL-NPV models.

## Modes

Both scripts support two operation modes:

### Headless Mode (Default)
- **Purpose**: Server/cluster environments without display
- **Features**: Console logging only, no 3D visualization
- **Usage**: `--headless` flag (default behavior)

### Visualization Mode
- **Purpose**: Interactive development and analysis
- **Features**: 3D visualizations, plots, interactive displays
- **Usage**: `--visualization` flag

## Training Script Usage

### Basic Training (Headless)
```bash
python train_model.py --config mine_rl_npv/configs/train.yaml --data mine_rl_npv/data/sample_model.csv
```

### Training with Visualization
```bash
python train_model.py --config mine_rl_npv/configs/train.yaml --data mine_rl_npv/data/sample_model.csv --visualization
```

### Advanced Training Examples
```bash
# Custom timesteps and device
python train_model.py --config mine_rl_npv/configs/train.yaml --data mine_rl_npv/data/sample_model.csv --timesteps 2000000 --device cuda

# Custom output directory
python train_model.py --config mine_rl_npv/configs/train.yaml --data mine_rl_npv/data/sample_model.csv --output-dir custom_experiments/

# With random seed for reproducibility
python train_model.py --config mine_rl_npv/configs/train.yaml --data mine_rl_npv/data/sample_model.csv --seed 42
```

### Training Arguments
- `--config`: Path to training configuration file (required)
- `--data`: Path to block model data file (required)
- `--headless`: Run in headless mode (default)
- `--visualization`: Run with visualization capabilities
- `--device`: Device to use (auto, cpu, cuda)
- `--timesteps`: Override total training timesteps
- `--output-dir`: Output directory for results
- `--verbose`: Verbosity level (0-2)
- `--seed`: Random seed for reproducibility

## Evaluation Script Usage

### Basic Evaluation (Headless)
```bash
python evaluate_model.py --model experiments/runs/run_20231201/models/best_model.zip --data mine_rl_npv/data/sample_model.csv
```

### Evaluation with Visualization
```bash
python evaluate_model.py --model experiments/runs/run_20231201/models/best_model.zip --data mine_rl_npv/data/sample_model.csv --visualization
```

### Advanced Evaluation Examples
```bash
# Comprehensive evaluation with plots and comparison
python evaluate_model.py --model best_model.zip --data mine_rl_npv/data/sample_model.csv --episodes 100 --compare --plot

# Custom output directory
python evaluate_model.py --model best_model.zip --data mine_rl_npv/data/sample_model.csv --output custom_evaluation/

# Evaluation with visualization and comparison
python evaluate_model.py --model best_model.zip --data mine_rl_npv/data/sample_model.csv --visualization --compare --plot --episodes 50
```

### Evaluation Arguments
- `--model`: Path to trained model file (.zip) (required)
- `--data`: Path to block model data file (required)
- `--headless`: Run in headless mode (default)
- `--visualization`: Run with visualization capabilities
- `--episodes`: Number of episodes to evaluate
- `--output`: Output directory for results
- `--config`: Training configuration (auto-detected if not provided)
- `--compare`: Compare with random policy baseline
- `--plot`: Generate evaluation plots
- `--deterministic`: Use deterministic policy evaluation
- `--verbose`: Verbosity level (0-2)
- `--seed`: Random seed for reproducibility

## Outputs

### Training Outputs
- **Headless Mode**:
  - Console logs with training progress
  - TensorBoard logs in output directory
  - Model checkpoints
  - Training configuration backup

- **Visualization Mode**:
  - All headless outputs plus:
  - Instructions for viewing TensorBoard
  - Visualization guidance

### Evaluation Outputs
- **Headless Mode**:
  - Console evaluation report
  - Evaluation statistics
  - Optional comparison with random policy
  - Optional evaluation plots

- **Visualization Mode**:
  - All headless outputs plus:
  - 3D grade visualizations
  - Mining state visualizations
  - Economic potential visualizations
  - Interactive plots (when possible)

## Environment Setup

### Headless Environment
The scripts automatically configure the environment for headless operation:
```python
os.environ['PYVISTA_OFF_SCREEN'] = 'true'
os.environ['DISPLAY'] = ''
```

### Visualization Environment
When visualization mode is enabled, the scripts attempt to use interactive backends when available.

## Error Handling

Both scripts include comprehensive error handling:
- Configuration file validation
- Data file validation
- Model file validation (evaluation)
- Import error detection
- Memory allocation warnings
- Graceful interruption handling

## Integration with Existing Code

The scripts integrate seamlessly with the existing codebase:
- Use existing `MiningTrainer` and `MiningEvaluator` classes
- Maintain compatibility with configuration files
- Preserve all existing functionality
- Add mode-specific enhancements

## Performance Considerations

### Memory Usage
- Large grid sizes may require significant memory
- Consider reducing grid dimensions for testing
- Use `--verbose 2` for detailed memory information

### Computational Requirements
- Training is computationally intensive
- Use `--device cuda` for GPU acceleration when available
- Reduce `--timesteps` for quick testing

## Troubleshooting

### Common Issues

1. **Memory allocation errors**:
   - Reduce grid size in environment configuration
   - Use smaller batch sizes in training configuration

2. **Display/visualization errors in headless mode**:
   - Scripts automatically handle this
   - Ensure headless mode is properly set

3. **Configuration file errors**:
   - Verify YAML syntax
   - Check file paths are correct
   - Ensure learning rates are numeric (not scientific notation strings)

4. **Data loading errors**:
   - Verify data file exists and is accessible
   - Check data file format (CSV expected)
   - Ensure sufficient disk space

### Debug Mode
Use `--verbose 2` for detailed debugging information including full stack traces.

## Examples

See the `examples/` directory for complete usage examples and sample configurations.