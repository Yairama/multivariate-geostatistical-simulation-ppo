#!/usr/bin/env python3
"""
Standalone training script for MineRL-NPV model.

This script provides both headless and visualization modes for training.
- Headless mode: Training without 3D visualization (console logs only)
- Visualization mode: Training with 3D visualization capabilities

Usage:
    # Headless mode (default)
    python train_model.py --config mine_rl_npv/configs/train.yaml --data data/sample

    # Visualization mode
    python train_model.py --config mine_rl_npv/configs/train.yaml --data data/sample --visualization

    # Custom settings
    python train_model.py --config mine_rl_npv/configs/train.yaml --data data/sample --headless --device cuda
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

# Add the mine_rl_npv package to the Python path
script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir / "mine_rl_npv"))

# Set environment for headless rendering if needed
def setup_headless_environment():
    """Setup environment variables for headless operation."""
    os.environ['PYVISTA_OFF_SCREEN'] = 'true'
    os.environ['DISPLAY'] = ''
    # Suppress Qt and visualization warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='pyvista')
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='vtk')


def main():
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser(
        description="Train MineRL-NPV model with headless/visualization modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training (headless mode)
  python train_model.py --config mine_rl_npv/configs/train.yaml --data data/sample

  # Training with visualization
  python train_model.py --config mine_rl_npv/configs/train.yaml --data data/sample --visualization

  # Custom training configuration
  python train_model.py --config mine_rl_npv/configs/train.yaml --data data/sample --device cuda --timesteps 2000000
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to training configuration file (e.g., mine_rl_npv/configs/train.yaml)"
    )
    
    parser.add_argument(
        "--data", 
        type=str, 
        required=True,
        help="Path to block model data directory or file"
    )
    
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--headless", 
        action="store_true", 
        default=True,
        help="Run in headless mode (no 3D visualization, console logs only) - DEFAULT"
    )
    mode_group.add_argument(
        "--visualization", 
        action="store_true", 
        default=False,
        help="Run with visualization mode (enable 3D visualization capabilities)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for training (default: auto)"
    )
    
    parser.add_argument(
        "--timesteps", 
        type=int, 
        default=None,
        help="Override total training timesteps"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="experiments/runs",
        help="Output directory for training results (default: experiments/runs)"
    )
    
    parser.add_argument(
        "--verbose", 
        type=int, 
        default=1,
        choices=[0, 1, 2],
        help="Verbosity level (0: silent, 1: info, 2: debug)"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Determine mode
    if args.visualization:
        headless_mode = False
    else:
        headless_mode = True
    
    # Setup environment based on mode
    if headless_mode:
        print("🤖 Running in HEADLESS mode (no 3D visualization)")
        setup_headless_environment()
    else:
        print("👁️  Running in VISUALIZATION mode (3D visualization enabled)")
    
    # Validate paths
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ Error: Data path not found: {data_path}")
        sys.exit(1)
    
    # Import training modules (after environment setup)
    try:
        from rl.train import MiningTrainer
        import torch
        import yaml
    except ImportError as e:
        print(f"❌ Error importing required modules: {e}")
        print("Make sure you're running from the repository root and all dependencies are installed.")
        sys.exit(1)
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"🔧 Using device: {device}")
    print(f"📊 Data path: {data_path}")
    print(f"⚙️  Config path: {config_path}")
    
    # Set random seed if provided
    if args.seed is not None:
        print(f"🎲 Setting random seed: {args.seed}")
        torch.manual_seed(args.seed)
        import numpy as np
        np.random.seed(args.seed)
    
    # Check for environment configuration
    env_config_path = config_path.parent / "env.yaml"
    if not env_config_path.exists():
        # Try to use examples/env_small.yaml as fallback
        if Path("examples/env_small.yaml").exists():
            print("ℹ️  env.yaml not found, using examples/env_small.yaml as fallback")
            # Create a temporary config directory with both files
            temp_dir = Path("/tmp/minerl_config")
            temp_dir.mkdir(exist_ok=True)
            
            # Copy train config
            temp_config_path = temp_dir / "train.yaml"
            import shutil
            shutil.copy2(config_path, temp_config_path)
            
            # Copy small env config
            temp_env_path = temp_dir / "env.yaml"
            shutil.copy2("examples/env_small.yaml", temp_env_path)
            
            config_path = temp_config_path
        else:
            print(f"❌ Error: env.yaml not found at {env_config_path}")
            print("❌ Error: examples/env_small.yaml also not found")
            print("Please ensure env.yaml exists in the same directory as the training config")
            sys.exit(1)
    
    # Override timesteps if provided
    if args.timesteps is not None:
        print(f"⏱️  Overriding timesteps: {args.timesteps:,}")
        # Load and modify config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        config['schedule']['total_timesteps'] = args.timesteps
        
        # Save modified config to temporary directory with env.yaml
        temp_dir = Path("/tmp/minerl_config")
        temp_dir.mkdir(exist_ok=True)
        
        temp_config_path = temp_dir / "train.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Copy env.yaml to temp directory (check if we already copied it)
        temp_env_path = temp_dir / "env.yaml"
        if not temp_env_path.exists():
            # Determine the appropriate env config based on the training config name
            original_config_name = Path(config_path.name if hasattr(config_path, 'name') else config_path).stem
            config_dir = Path(config_path).parent if not str(config_path).startswith('/tmp') else Path("mine_rl_npv/configs")
            
            # Try to match the environment config to the training config
            if "memory_optimized" in original_config_name:
                env_config_candidates = [
                    config_dir / "env_memory_optimized.yaml",
                    Path("mine_rl_npv/configs/env_memory_optimized.yaml")
                ]
            elif "ultra_light" in original_config_name:
                env_config_candidates = [
                    config_dir / "env_ultra_light.yaml",
                    Path("mine_rl_npv/configs/env_ultra_light.yaml")
                ]
            elif "small" in original_config_name:
                env_config_candidates = [
                    config_dir / "env_small.yaml",
                    Path("examples/env_small.yaml")
                ]
            else:
                env_config_candidates = [
                    config_dir / "env.yaml",
                    Path("mine_rl_npv/configs/env.yaml")
                ]
            
            # Find the first existing config
            env_config_path = None
            for candidate in env_config_candidates:
                if candidate.exists():
                    env_config_path = candidate
                    break
            
            if env_config_path is None:
                print(f"⚠️  Warning: No environment config found")
                print("⚠️  Please ensure env.yaml exists in the same directory as the training config")
                sys.exit(1)
            
            import shutil
            shutil.copy2(env_config_path, temp_env_path)
            print(f"ℹ️  Using environment config: {env_config_path}")
        
        config_path = temp_config_path
    
    try:
        # Create trainer
        print("🚀 Initializing trainer...")
        trainer = MiningTrainer(str(config_path), str(data_path))
        
        # Override output directory if provided
        if args.output_dir != "experiments/runs":
            trainer.run_dir = Path(args.output_dir) / trainer.run_dir.name
            trainer.log_dir = trainer.run_dir / "logs"
            trainer.checkpoint_dir = trainer.run_dir / "models"
            
            # Create directories
            trainer.run_dir.mkdir(parents=True, exist_ok=True)
            trainer.log_dir.mkdir(parents=True, exist_ok=True)
            trainer.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Output directory: {trainer.run_dir}")
        
        # Start training
        print("🎯 Starting training...")
        print("=" * 60)
        
        model = trainer.train()
        
        print("=" * 60)
        print("✅ Training completed successfully!")
        print(f"📄 Results saved to: {trainer.run_dir}")
        print(f"🏆 Best model: {trainer.checkpoint_dir}/best_model.zip")
        
        # Additional information based on mode
        if not headless_mode:
            print("\n📊 Visualization tips:")
            print(f"  - View TensorBoard: tensorboard --logdir {trainer.log_dir}")
            print(f"  - View 3D results: python evaluate_model.py --model {trainer.checkpoint_dir}/best_model.zip --data {data_path} --visualization")
        else:
            print("\n📊 Next steps:")
            print(f"  - View TensorBoard: tensorboard --logdir {trainer.log_dir}")
            print(f"  - Evaluate model: python evaluate_model.py --model {trainer.checkpoint_dir}/best_model.zip --data {data_path}")
        
        return model
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        if args.verbose >= 2:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()