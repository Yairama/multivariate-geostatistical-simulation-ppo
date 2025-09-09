#!/usr/bin/env python3
"""
Minimal working trainer to isolate the issue.
"""

import os
import sys
from pathlib import Path

# Add the mine_rl_npv package to the Python path
script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir / "mine_rl_npv"))

import yaml
from envs.mining_env import make_mining_env
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from rl.feature_extractor import CNN3DFeatureExtractorSmall
import torch.nn as nn

def minimal_train(config_path, data_path, timesteps=32):
    """Minimal training function that works."""
    print("Starting minimal training...")
    
    # Create single environment
    env = DummyVecEnv([lambda: make_mining_env(config_path, data_path)])
    print(f"Environment created. Obs space: {env.observation_space}")
    
    # Create policy kwargs with custom feature extractor
    policy_kwargs = {
        'features_extractor_class': CNN3DFeatureExtractorSmall,
        'features_extractor_kwargs': {
            'features_dim': 256,
            'dropout': 0.1
        },
        'net_arch': {'pi': [128, 128], 'vf': [128, 128]},
        'activation_fn': nn.Tanh
    }
    
    # Create model
    model = MaskablePPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=2,
        tensorboard_log=None,
        n_steps=16,  # Small for testing
        batch_size=16,
        policy_kwargs=policy_kwargs,
        device="cpu"
    )
    
    print(f"Model created. Parameters: {sum(p.numel() for p in model.policy.parameters()):,}")
    
    # Train
    print(f"Starting training for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps)
    
    print("Training completed successfully!")
    
    # Save model
    model_path = "experiments/minimal_model.zip"
    os.makedirs("experiments", exist_ok=True)
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    env.close()

if __name__ == "__main__":
    config_path = "mine_rl_npv/configs/env_16gb_gpu.yaml"
    data_path = "mine_rl_npv/data/sample_model.csv"
    
    minimal_train(config_path, data_path, timesteps=32)