#!/usr/bin/env python3
"""
Setup and test script for MineRL-NPV project.
"""

import os
import sys
import subprocess
from pathlib import Path
import yaml
import numpy as np


def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")
    
    required_packages = [
        'numpy', 'pandas', 'gymnasium', 'stable_baselines3', 
        'sb3_contrib', 'torch', 'tensorboard', 'matplotlib',
        'seaborn', 'pyyaml', 'tqdm', 'scikit_learn', 'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("All dependencies are installed!")
    return True


def test_data_loading():
    """Test data loading functionality."""
    print("\nTesting data loading...")
    
    try:
        sys.path.append('mine_rl_npv')
        from geo.loaders import BlockModelLoader
        
        # Test with real data
        if Path("mine_rl_npv/data/sample_model.csv").exists():
            loader = BlockModelLoader("mine_rl_npv/configs/env.yaml")
            data = loader.load_and_preprocess("mine_rl_npv/data/sample_model.csv")
            print(f"✓ Loaded real data: {list(data.keys())}")
        else:
            print("✗ Sample data not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False


def test_environment():
    """Test the mining environment."""
    print("\nTesting mining environment...")
    
    try:
        sys.path.append('mine_rl_npv')
        from envs.mining_env import make_mining_env
        
        # Create environment
        env = make_mining_env("mine_rl_npv/configs/env.yaml")
        
        # Test reset
        obs, info = env.reset()
        print(f"✓ Environment reset successful, obs shape: {obs.shape}")
        
        # Test action masking
        action_mask = env.action_mask()
        valid_actions = np.where(action_mask)[0]
        print(f"✓ Action masking works, {len(valid_actions)} valid actions")
        
        # Test a few steps
        for i in range(3):
            if len(valid_actions) > 0:
                action = np.random.choice(valid_actions)
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"✓ Step {i+1}: reward={reward:.2f}, terminated={terminated}")
                
                if terminated or truncated:
                    break
                
                action_mask = env.action_mask()
                valid_actions = np.where(action_mask)[0]
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False


def test_feature_extractor():
    """Test the 3D CNN feature extractor."""
    print("\nTesting feature extractor...")
    
    try:
        sys.path.append('mine_rl_npv')
        from rl.feature_extractor import CNN3DFeatureExtractorTiny
        import torch
        import gymnasium as gym
        
        # Create dummy observation space
        obs_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(15, 20, 20, 10),
            dtype=np.float32
        )
        
        # Create extractor
        extractor = CNN3DFeatureExtractorTiny(obs_space, features_dim=64)
        
        # Test forward pass
        dummy_input = torch.randn(2, 15, 20, 20, 10)
        output = extractor(dummy_input)
        
        print(f"✓ Feature extractor works, output shape: {output.shape}")
        return True
        
    except Exception as e:
        print(f"✗ Feature extractor test failed: {e}")
        return False


def test_synthetic_generation():
    """Test synthetic data generation."""
    print("\nTesting synthetic data generation...")
    
    try:
        sys.path.append('mine_rl_npv')
        from geo.synth_generator import SyntheticGenerator
        
        generator = SyntheticGenerator("mine_rl_npv/configs/env.yaml")
        data = generator.generate_porphyry_deposit(nx=10, ny=10, nz=5, seed=42)
        
        print(f"✓ Synthetic generation works, created {len(data)} arrays")
        
        # Save test file
        test_output = "mine_rl_npv/data/test_synthetic.csv"
        generator.save_as_csv(data, test_output)
        
        if Path(test_output).exists():
            print(f"✓ Synthetic data saved to {test_output}")
            return True
        else:
            print("✗ Failed to save synthetic data")
            return False
        
    except Exception as e:
        print(f"✗ Synthetic generation test failed: {e}")
        return False


def create_example_configs():
    """Create example configuration files."""
    print("\nCreating example configurations...")
    
    # Create a simplified training config for testing
    simple_train_config = {
        'training': {
            'algorithm': 'MaskablePPO',
            'hyperparameters': {
                'learning_rate': 3e-4,
                'n_steps': 512,
                'batch_size': 32,
                'n_epochs': 5,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5
            },
            'policy': {
                'net_arch': {'pi': [128, 128], 'vf': [128, 128]},
                'activation_fn': 'tanh'
            },
            'feature_extractor': {
                'type': 'CNN3D',
                'channels': [8, 16],
                'kernel_sizes': [3, 3],
                'strides': [1, 2],
                'pooling': 'adaptive',
                'dropout': 0.1,
                'output_dim': 256
            }
        },
        'schedule': {
            'total_timesteps': 10000,
            'eval_freq': 2000,
            'save_freq': 5000
        },
        'env_settings': {
            'n_envs': 1,
            'max_episode_steps': 100,
            'normalize_obs': True,
            'normalize_reward': False
        },
        'evaluation': {
            'n_eval_episodes': 3,
            'deterministic': True,
            'render': False
        },
        'logging': {
            'tensorboard': True,
            'tensorboard_log': './experiments/runs/',
            'verbose': 1,
            'custom_metrics': ['episode_npv', 'total_tonnage_mined', 'avg_cu_grade']
        },
        'experiment': {
            'name': 'minerl_npv_test',
            'tags': ['mining', 'rl', 'test'],
            'notes': 'Test run with simplified configuration'
        }
    }
    
    # Save simplified config
    test_config_path = "mine_rl_npv/configs/train_test.yaml"
    with open(test_config_path, 'w') as f:
        yaml.dump(simple_train_config, f, default_flow_style=False)
    
    print(f"✓ Created test training config: {test_config_path}")


def run_quick_training_test():
    """Run a very quick training test."""
    print("\nRunning quick training test...")
    
    try:
        # Generate small synthetic dataset
        sys.path.append('mine_rl_npv')
        from geo.synth_generator import SyntheticGenerator
        
        generator = SyntheticGenerator("mine_rl_npv/configs/env.yaml")
        data = generator.generate_porphyry_deposit(nx=8, ny=8, nz=4, seed=123)
        test_data_path = "mine_rl_npv/data/quick_test.csv"
        generator.save_as_csv(data, test_data_path)
        
        # Import training components
        from rl.train import MiningTrainer
        
        # Create trainer with test config
        trainer = MiningTrainer("mine_rl_npv/configs/train_test.yaml", test_data_path)
        
        print("✓ Quick training test setup successful")
        print("Note: Full training test would take too long for setup verification")
        
        return True
        
    except Exception as e:
        print(f"✗ Quick training test failed: {e}")
        return False


def main():
    """Main setup and test function."""
    print("MineRL-NPV Setup and Test")
    print("=" * 50)
    
    # Change to project directory
    os.chdir("mine_rl_npv")
    
    tests = [
        ("Dependencies", check_dependencies),
        ("Data Loading", test_data_loading),
        ("Environment", test_environment),
        ("Feature Extractor", test_feature_extractor),
        ("Synthetic Generation", test_synthetic_generation),
        ("Quick Training", run_quick_training_test)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:20}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! MineRL-NPV is ready to use.")
        print("\nQuick start:")
        print("1. Generate synthetic data: python geo/synth_generator.py")
        print("2. Train model: python rl/train.py --config configs/train_test.yaml")
        print("3. Visualize results: python viz/viewer.py")
    else:
        print(f"\n❌ {total - passed} tests failed. Check error messages above.")
        
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)