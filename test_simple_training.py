#!/usr/bin/env python3
"""
Simple test to isolate where training hangs.
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

def test_env_creation():
    """Test environment creation."""
    print("Testing environment creation...")
    
    # Use the 16GB GPU config
    config_path = "mine_rl_npv/configs/env_16gb_gpu.yaml"
    data_path = "mine_rl_npv/data/sample_model.csv"
    
    try:
        # Create single environment
        env = make_mining_env(config_path, data_path)
        print("✓ Environment created successfully")
        
        # Test reset
        print("Testing environment reset...")
        obs, info = env.reset()
        print(f"✓ Environment reset successful, obs shape: {obs.shape}")
        
        # Test action mask
        print("Testing action mask...")
        action_mask = env.action_masks()
        print(f"✓ Action mask generated, shape: {action_mask.shape}, valid actions: {action_mask.sum()}")
        
        # Test single step
        print("Testing single step...")
        valid_actions = [i for i, mask in enumerate(action_mask) if mask]
        if valid_actions:
            action = valid_actions[0]
            obs, reward, done, truncated, info = env.step(action)
            print(f"✓ Step successful, reward: {reward}, done: {done}")
        
        env.close()
        print("✓ Environment test complete")
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_creation():
    """Test model creation."""
    print("Testing model creation...")
    
    try:
        # Create environment
        config_path = "mine_rl_npv/configs/env_16gb_gpu.yaml"
        data_path = "mine_rl_npv/data/sample_model.csv"
        
        env = DummyVecEnv([lambda: make_mining_env(config_path, data_path)])
        print("✓ Vectorized environment created")
        
        # Create model with minimal settings
        model = MaskablePPO(
            "CnnPolicy",
            env,
            verbose=1,
            tensorboard_log=None,  # Disable tensorboard
            n_steps=32,  # Very small for testing
            batch_size=32,
            device="cpu"
        )
        print("✓ Model created successfully")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_single_learn_step():
    """Test a very short learn call."""
    print("Testing single learn step...")
    
    try:
        # Create environment
        config_path = "mine_rl_npv/configs/env_16gb_gpu.yaml" 
        data_path = "mine_rl_npv/data/sample_model.csv"
        
        env = DummyVecEnv([lambda: make_mining_env(config_path, data_path)])
        
        # Create model with minimal settings and no callbacks
        model = MaskablePPO(
            "CnnPolicy",
            env,
            verbose=2,
            tensorboard_log=None,  # Disable tensorboard completely
            n_steps=16,  # Very small
            batch_size=16,
            device="cpu"
        )
        
        print("Starting learn with 16 timesteps...")
        # Try very short learning
        model.learn(total_timesteps=16, tb_log_name=None)
        print("✓ Learn completed successfully!")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Learn failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing MineRL-NPV Training Components")
    print("=" * 50)
    
    # Test step by step
    if test_env_creation():
        if test_model_creation():
            test_single_learn_step()
    
    print("Testing complete.")