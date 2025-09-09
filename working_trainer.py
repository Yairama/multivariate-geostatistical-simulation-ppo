#!/usr/bin/env python3
"""
Working trainer for MineRL-NPV - simplified version that actually works.
"""

import os
import sys
import argparse
from pathlib import Path
import yaml
from datetime import datetime

# Add the mine_rl_npv package to the Python path
script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir / "mine_rl_npv"))

from envs.mining_env import make_mining_env
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from rl.feature_extractor import CNN3DFeatureExtractorSmall
import torch.nn as nn

def create_experiment_dir():
    """Create experiment directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(f"experiments/runs/minerl_npv_working_{timestamp}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir

def train_model(config_path, env_config_path, data_path, timesteps, device="auto"):
    """Train the model with working configuration."""
    print("🚀 Starting MineRL-NPV Training (Working Version)")
    print("=" * 60)
    
    # Create experiment directory
    exp_dir = create_experiment_dir()
    print(f"📁 Experiment directory: {exp_dir}")
    
    # Auto-detect device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"🔧 Using device: {device}")
    print(f"📊 Data: {data_path}")
    print(f"⚙️  Environment config: {env_config_path}")
    print(f"🎯 Training steps: {timesteps:,}")
    
    # Create environment
    print("\n🌍 Creating training environment...")
    env = DummyVecEnv([lambda: make_mining_env(env_config_path, data_path)])
    print(f"✓ Environment created. Observation space: {env.observation_space}")
    
    # Load training configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract hyperparameters
    hyperparams = config['training']['hyperparameters']
    
    # Create policy kwargs
    policy_kwargs = {
        'features_extractor_class': CNN3DFeatureExtractorSmall,
        'features_extractor_kwargs': {
            'features_dim': config['training']['feature_extractor']['output_dim'],
            'dropout': config['training']['feature_extractor']['dropout']
        },
        'net_arch': config['training']['policy']['net_arch'],
        'activation_fn': getattr(nn, config['training']['policy']['activation_fn'].title())
    }
    
    # Create model
    print("\n🤖 Creating MaskablePPO model...")
    model = MaskablePPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=hyperparams['learning_rate'],
        n_steps=hyperparams['n_steps'],
        batch_size=hyperparams['batch_size'],
        n_epochs=hyperparams['n_epochs'],
        gamma=hyperparams['gamma'],
        gae_lambda=hyperparams['gae_lambda'],
        clip_range=hyperparams['clip_range'],
        ent_coef=hyperparams['ent_coef'],
        vf_coef=hyperparams['vf_coef'],
        max_grad_norm=hyperparams['max_grad_norm'],
        policy_kwargs=policy_kwargs,
        tensorboard_log=None,  # Disable TensorBoard for now
        verbose=1,
        device=device
    )
    
    print(f"✓ Model created. Parameters: {sum(p.numel() for p in model.policy.parameters()):,}")
    
    # Save configuration
    config_save_path = exp_dir / "config.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"✓ Configuration saved to: {config_save_path}")
    
    # Train
    print(f"\n🎯 Starting training for {timesteps:,} timesteps...")
    try:
        model.learn(total_timesteps=timesteps)
        print("✅ Training completed successfully!")
        
        # Save final model
        model_path = exp_dir / "final_model.zip"
        model.save(str(model_path))
        print(f"💾 Model saved to: {model_path}")
        
    except KeyboardInterrupt:
        print("⏹️  Training interrupted by user")
        interrupted_path = exp_dir / "interrupted_model.zip"
        model.save(str(interrupted_path))
        print(f"💾 Interrupted model saved to: {interrupted_path}")
    
    finally:
        env.close()
    
    print("\n🎉 Training session completed!")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Working MineRL-NPV Trainer")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="mine_rl_npv/configs/train_optimized.yaml",
        help="Training configuration file"
    )
    
    parser.add_argument(
        "--env-config", 
        type=str, 
        default="mine_rl_npv/configs/env_optimized.yaml",
        help="Environment configuration file"
    )
    
    parser.add_argument(
        "--data", 
        type=str, 
        required=True,
        help="Path to the mining data CSV file"
    )
    
    parser.add_argument(
        "--timesteps", 
        type=int, 
        default=10000,
        help="Number of training timesteps"
    )
    
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for training"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.config).exists():
        print(f"❌ Training config not found: {args.config}")
        return 1
        
    if not Path(args.env_config).exists():
        print(f"❌ Environment config not found: {args.env_config}")
        return 1
        
    if not Path(args.data).exists():
        print(f"❌ Data file not found: {args.data}")
        return 1
    
    # Start training
    train_model(args.config, args.env_config, args.data, args.timesteps, args.device)
    return 0

if __name__ == "__main__":
    exit(main())