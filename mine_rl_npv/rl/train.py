"""
Training script for MineRL-NPV using MaskablePPO.
"""

import os
import yaml
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from sb3_contrib import MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from sb3_contrib.common.wrappers import ActionMasker

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# Fix imports for package structure
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from envs.mining_env import make_mining_env
from rl.feature_extractor import CNN3DFeatureExtractor, CNN3DFeatureExtractorSmall, CNN3DFeatureExtractorTiny
# Import visualization callback
from rl.visualization_callback import VisualizationCallback


class CustomMiningCallback(BaseCallback):
    """Minimal callback for basic logging."""
    
    def __init__(self, log_dir: str, config_path: str = None, verbose: int = 1):
        super().__init__(verbose)
        self.log_dir = log_dir
        try:
            self.writer = SummaryWriter(log_dir)
            self.logging_enabled = True
        except:
            self.logging_enabled = False
            if verbose > 0:
                print("Warning: TensorBoard logging disabled due to initialization error")
    
    def _on_step(self) -> bool:
        """Minimal step logging."""
        if not self.logging_enabled:
            return True
            
        # Only log basic reward statistics to prevent hangs
        if hasattr(self.locals, 'rewards') and self.locals['rewards'] is not None:
            try:
                rewards = self.locals['rewards']
                step = self.num_timesteps
                self.writer.add_scalar('reward/mean', np.mean(rewards), step)
            except:
                pass  # Ignore logging errors
        
        return True

    def _on_rollout_end(self) -> bool:
        """Minimal rollout logging."""
        return True  # Skip complex logging that causes hangs
    
    def _on_training_end(self) -> None:
        """Clean up."""
        if self.logging_enabled:
            try:
                self.writer.close()
            except:
                pass


class MiningTrainer:
    """Main trainer class for MineRL-NPV."""
    
    def __init__(self, config_path: str, data_path: str = None):
        """Initialize trainer with configuration."""
        # Load configurations
        with open(config_path, 'r') as f:
            self.train_config = yaml.safe_load(f)
        
        # Try to find corresponding environment configuration
        config_dir = Path(config_path).parent
        config_name = Path(config_path).stem
        
        # Try specific env config first
        if "optimized" in config_name:
            env_config_path = config_dir / "env_optimized.yaml"
        elif "16gb_gpu" in config_name:
            env_config_path = config_dir / "env_16gb_gpu.yaml"
        elif "memory_optimized" in config_name:
            env_config_path = config_dir / "env_memory_optimized.yaml"
        elif "ultra_light" in config_name:
            env_config_path = config_dir / "env_ultra_light.yaml"
        elif "small" in config_name:
            env_config_path = config_dir / "env_small.yaml"
        else:
            env_config_path = config_dir / "env.yaml"
        
        # Fall back to default env.yaml if specific config doesn't exist
        if not env_config_path.exists():
            env_config_path = config_dir / "env.yaml"
        
        self.env_config_path = str(env_config_path)
        with open(self.env_config_path, 'r') as f:
            self.env_config = yaml.safe_load(f)
        
        print(f"Using environment config: {self.env_config_path}")
        
        self.data_path = data_path
        self.setup_directories()
        
    def setup_directories(self):
        """Setup logging and checkpoint directories."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = self.train_config['experiment']['name']
        
        self.run_dir = Path(f"experiments/runs/{exp_name}_{timestamp}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.log_dir = self.run_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        print(f"Experiment directory: {self.run_dir}")
    
    def create_env(self, rank: int = 0):
        """Create a single environment instance."""
        def _init():
            env = make_mining_env(self.env_config_path, self.data_path)
            
            # Add monitoring
            monitor_path = str(self.log_dir / f"monitor_{rank}.csv")
            env = Monitor(env, monitor_path)
            
            return env
        
        set_random_seed(rank)
        return _init
    
    def create_vec_env(self):
        """Create vectorized environment."""
        n_envs = self.train_config['env_settings']['n_envs']
        
        if n_envs == 1:
            env_fns = [self.create_env(0)]
            return DummyVecEnv(env_fns)
        else:
            env_fns = [self.create_env(i) for i in range(n_envs)]
            return SubprocVecEnv(env_fns)
    
    def create_policy_kwargs(self, observation_space):
        """Create policy kwargs with custom feature extractor."""
        extractor_config = self.train_config['training']['feature_extractor']
        
        # Select feature extractor type
        extractor_type = extractor_config['type']
        if extractor_type == "CNN3D":
            feature_extractor_class = CNN3DFeatureExtractor
            feature_extractor_kwargs = {
                'features_dim': extractor_config['output_dim'],
                'channels': extractor_config['channels'],
                'kernel_sizes': extractor_config['kernel_sizes'],
                'strides': extractor_config['strides'],
                'dropout': extractor_config['dropout'],
                'pooling': extractor_config['pooling']
            }
        elif extractor_type == "CNN3DSmall":
            feature_extractor_class = CNN3DFeatureExtractorSmall
            feature_extractor_kwargs = {
                'features_dim': extractor_config['output_dim'],
                'dropout': extractor_config['dropout']
            }
        elif extractor_type == "CNN3DTiny":
            feature_extractor_class = CNN3DFeatureExtractorTiny
            feature_extractor_kwargs = {
                'features_dim': extractor_config['output_dim'],
                'dropout': extractor_config.get('dropout', 0.1)
            }
        else:
            # Backward compatibility: check channel configuration
            if extractor_config.get('channels') == [8, 16]:  # Small config
                feature_extractor_class = CNN3DFeatureExtractorSmall
                feature_extractor_kwargs = {
                    'features_dim': extractor_config['output_dim'],
                    'dropout': extractor_config['dropout']
                }
            else:
                feature_extractor_class = CNN3DFeatureExtractor
                feature_extractor_kwargs = {
                    'features_dim': extractor_config['output_dim'],
                    'channels': extractor_config['channels'],
                    'kernel_sizes': extractor_config['kernel_sizes'],
                    'strides': extractor_config['strides'],
                    'dropout': extractor_config['dropout'],
                    'pooling': extractor_config['pooling']
                }
        
        # Policy network architecture
        policy_config = self.train_config['training']['policy']
        
        policy_kwargs = {
            'features_extractor_class': feature_extractor_class,
            'features_extractor_kwargs': feature_extractor_kwargs,
            'net_arch': policy_config['net_arch'],
            'activation_fn': getattr(nn, policy_config['activation_fn'].title())
        }
        
        return policy_kwargs
    
    def create_model(self, env):
        """Create MaskablePPO model."""
        hyperparams = self.train_config['training']['hyperparameters']
        policy_kwargs = self.create_policy_kwargs(env.observation_space)
        
        model = MaskablePPO(
            policy="MultiInputPolicy",  # Use MultiInputPolicy for complex obs spaces
            env=env,
            learning_rate=hyperparams['learning_rate'],
            n_steps=hyperparams['n_steps'],
            batch_size=hyperparams['batch_size'],
            n_epochs=hyperparams['n_epochs'],
            gamma=hyperparams['gamma'],
            gae_lambda=hyperparams['gae_lambda'],
            clip_range=hyperparams['clip_range'],
            clip_range_vf=hyperparams.get('clip_range_vf'),
            ent_coef=hyperparams['ent_coef'],
            vf_coef=hyperparams['vf_coef'],
            max_grad_norm=hyperparams['max_grad_norm'],
            policy_kwargs=policy_kwargs,
            tensorboard_log=None,  # DISABLE TENSORBOARD FOR TESTING
            verbose=self.train_config['logging']['verbose']
        )
        
        return model
    
    def create_callbacks(self, eval_env, enable_visualization=False):
        """Create training callbacks."""
        callbacks = []
        
        # Custom mining metrics callback (simplified)
        mining_callback = CustomMiningCallback(
            log_dir=str(self.log_dir),
            verbose=self.train_config['logging']['verbose']
        )
        callbacks.append(mining_callback)
        
        # Add visualization callback if enabled
        if enable_visualization and self.train_config['logging'].get('enable_visualization', False):
            visualization_callback = VisualizationCallback(
                env_config_path=self.env_config_path,
                visualization_freq=self.train_config['logging'].get('visualization_freq', 1000),
                save_screenshots=self.train_config['logging'].get('save_screenshots', True),
                output_dir=str(self.run_dir / "visualizations"),
                verbose=self.train_config['logging']['verbose']
            )
            callbacks.append(visualization_callback)
            print("🎬 Visualization callback added to training!")
        
        # Evaluation callback (only if eval_env is provided)
        if eval_env is not None:
            eval_config = self.train_config['evaluation']
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=str(self.checkpoint_dir),
                log_path=str(self.log_dir),
                eval_freq=self.train_config['schedule']['eval_freq'],
                n_eval_episodes=eval_config['n_eval_episodes'],
                deterministic=eval_config['deterministic'],
                render=eval_config['render']
            )
            callbacks.append(eval_callback)
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.train_config['schedule']['save_freq'],
            save_path=str(self.checkpoint_dir),
            name_prefix="model_checkpoint"
        )
        callbacks.append(checkpoint_callback)
        
        return CallbackList(callbacks)
    
    def train(self, enable_visualization=False):
        """Main training loop."""
        print("Starting MineRL-NPV training...")
        
        # Create environments
        print("Creating training environment...")
        train_env = self.create_vec_env()
        
        print("Creating evaluation environment...")
        # SKIP EVAL ENV FOR TESTING
        eval_env = None
        
        # Create model
        print("Creating MaskablePPO model...")
        model = self.create_model(train_env)
        
        # Print model info
        print(f"Model parameters: {sum(p.numel() for p in model.policy.parameters()):,}")
        
        # Create callbacks with visualization support
        callbacks = self.create_callbacks(eval_env, enable_visualization=enable_visualization)
        
        # Save configurations
        config_save_path = self.run_dir / "config.yaml"
        with open(config_save_path, 'w') as f:
            yaml.dump({
                'training': self.train_config,
                'environment': self.env_config
            }, f, default_flow_style=False)
        
        print(f"Configuration saved to: {config_save_path}")
        
        # Start training
        total_timesteps = self.train_config['schedule']['total_timesteps']
        print(f"Training for {total_timesteps:,} timesteps...")
        
        if enable_visualization:
            print("🎬 Visualization enabled - screenshots will be captured during training!")
        
        try:
            model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                tb_log_name=None  # DISABLE TB LOG NAME
            )
            
            # Save final model
            final_model_path = self.run_dir / "final_model.zip"
            model.save(str(final_model_path))
            print(f"Final model saved to: {final_model_path}")
            
        except KeyboardInterrupt:
            print("Training interrupted by user")
            # Save current model
            interrupted_model_path = self.run_dir / "interrupted_model.zip"
            model.save(str(interrupted_model_path))
            print(f"Interrupted model saved to: {interrupted_model_path}")
        
        finally:
            train_env.close()
            # eval_env.close()  # Skip since eval_env is None
        
        print("Training completed!")
        return model


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train MineRL-NPV model")
    parser.add_argument("--config", type=str, default="configs/train.yaml",
                       help="Path to training configuration file")
    parser.add_argument("--data", type=str, default=None,
                       help="Path to block model data file")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (cpu, cuda, auto)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create trainer and start training
    trainer = MiningTrainer(args.config, args.data)
    model = trainer.train()
    
    return model


if __name__ == "__main__":
    main()