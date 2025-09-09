"""
Visualization callback for training with real-time visualization and screenshot capture.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for screenshots
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional
from stable_baselines3.common.callbacks import BaseCallback
import pyvista as pv

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Fix imports for package structure
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from viz.viewer import MiningVisualizer


class VisualizationCallback(BaseCallback):
    """
    Callback for real-time visualization during training with screenshot capture.
    """
    
    def __init__(self, 
                 env_config_path: str,
                 visualization_freq: int = 1000,
                 save_screenshots: bool = True,
                 output_dir: str = "visualizations",
                 verbose: int = 1):
        """
        Initialize visualization callback.
        
        Args:
            env_config_path: Path to environment configuration
            visualization_freq: Frequency to show/save visualizations (in timesteps)
            save_screenshots: Whether to save screenshots
            output_dir: Directory to save visualizations
            verbose: Verbosity level
        """
        super().__init__(verbose)
        
        self.env_config_path = env_config_path
        self.visualization_freq = visualization_freq
        self.save_screenshots = save_screenshots
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize visualizer
        try:
            self.visualizer = MiningVisualizer(env_config_path)
            self.visualizer_available = True
            print(f"✅ Visualization callback initialized - screenshots will be saved to: {self.output_dir}")
        except Exception as e:
            self.visualizer_available = False
            if verbose > 0:
                print(f"⚠️ Warning: Could not initialize visualizer: {e}")
    
    def _on_training_start(self) -> None:
        """Called when training starts."""
        if self.visualizer_available:
            print("🎬 Starting visualization during training...")
            # Create initial visualization directory structure
            self.training_viz_dir = self.output_dir / f"training_step_screenshots"
            self.training_viz_dir.mkdir(exist_ok=True)
            
            if self.verbose > 0:
                print(f"📸 Screenshots will be saved to: {self.training_viz_dir}")
    
    def _on_step(self) -> bool:
        """Called after each training step."""
        if not self.visualizer_available:
            return True
            
        # Check if it's time to create visualization
        if self.num_timesteps % self.visualization_freq == 0:
            self._create_training_visualization()
        
        return True
    
    def _create_training_visualization(self):
        """Create and save current training state visualization."""
        try:
            # Get current environment state from the first environment
            if hasattr(self.training_env, 'envs') and len(self.training_env.envs) > 0:
                env = self.training_env.envs[0]
            else:
                env = self.training_env
            
            # Get mining environment from wrapped environment
            mining_env = env
            while hasattr(mining_env, 'env'):
                mining_env = mining_env.env
            
            if hasattr(mining_env, 'data'):
                # Load current data into visualizer
                self.visualizer.load_data(mining_env.data)
                
                # Create multiple visualizations
                step_dir = self.training_viz_dir / f"step_{self.num_timesteps:08d}"
                step_dir.mkdir(exist_ok=True)
                
                # 1. Current mining state visualization
                self._create_mining_state_plot(mining_env, step_dir)
                
                # 2. Grade visualization
                self._create_grade_plot(step_dir)
                
                # 3. Training progress plot
                self._create_progress_plot(step_dir)
                
                print(f"📸 Screenshots saved for step {self.num_timesteps:,} to: {step_dir}")
                
        except Exception as e:
            if self.verbose > 0:
                print(f"⚠️ Warning: Could not create visualization at step {self.num_timesteps}: {e}")
    
    def _create_mining_state_plot(self, mining_env, save_dir: Path):
        """Create mining state visualization with matplotlib."""
        try:
            # Get current mining state
            mined_mask = mining_env.data.get('mined_flag', np.zeros_like(mining_env.data['cu'], dtype=bool))
            extraction_days = mining_env.data.get('extraction_day', np.zeros_like(mining_env.data['cu']))
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Mining Progress - Training Step {self.num_timesteps:,}', fontsize=14)
            
            # Top view of mined blocks
            mined_top = np.sum(mined_mask, axis=2)  # Sum over Z axis
            im1 = axes[0, 0].imshow(mined_top.T, origin='lower', cmap='Reds', alpha=0.8)
            axes[0, 0].set_title('Mined Blocks (Top View)')
            axes[0, 0].set_xlabel('X (blocks)')
            axes[0, 0].set_ylabel('Y (blocks)')
            plt.colorbar(im1, ax=axes[0, 0])
            
            # Copper grade distribution
            cu_avg = np.mean(mining_env.data['cu'], axis=2)
            im2 = axes[0, 1].imshow(cu_avg.T, origin='lower', cmap='viridis')
            axes[0, 1].set_title('Copper Grade Distribution')
            axes[0, 1].set_xlabel('X (blocks)')
            axes[0, 1].set_ylabel('Y (blocks)')
            plt.colorbar(im2, ax=axes[0, 1])
            
            # Mining progress over time
            if np.any(extraction_days > 0):
                extraction_hist, bins = np.histogram(extraction_days[extraction_days > 0], bins=20)
                axes[1, 0].bar(bins[:-1], extraction_hist, width=np.diff(bins), alpha=0.7)
                axes[1, 0].set_title('Mining Activity Over Time')
                axes[1, 0].set_xlabel('Extraction Day')
                axes[1, 0].set_ylabel('Blocks Mined')
            else:
                axes[1, 0].text(0.5, 0.5, 'No blocks mined yet', ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Mining Activity Over Time')
            
            # Statistics
            total_blocks = np.prod(mining_env.data['cu'].shape)
            mined_blocks = np.sum(mined_mask)
            mined_percentage = (mined_blocks / total_blocks) * 100
            
            stats_text = f"""Training Statistics:
            
Total Blocks: {total_blocks:,}
Mined Blocks: {mined_blocks:,}
Mined Percentage: {mined_percentage:.1f}%
Training Step: {self.num_timesteps:,}

Current Episode:
Episode Length: {getattr(mining_env, 'current_day', 'N/A')}
Total NPV: ${getattr(mining_env, 'total_npv', 0):,.2f}
"""
            axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, 
                           verticalalignment='top', fontfamily='monospace', fontsize=10)
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].set_title('Training Statistics')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            # Save the plot
            save_path = save_dir / f'mining_state_step_{self.num_timesteps:08d}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            if self.verbose > 0:
                print(f"Could not create mining state plot: {e}")
            plt.close('all')
    
    def _create_grade_plot(self, save_dir: Path):
        """Create grade distribution visualization."""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(f'Grade Distributions - Step {self.num_timesteps:,}', fontsize=14)
            
            # Copper grade histogram
            cu_grades = self.visualizer.data['cu'].flatten()
            axes[0].hist(cu_grades, bins=50, alpha=0.7, color='orange', edgecolor='black')
            axes[0].set_title('Copper Grade Distribution')
            axes[0].set_xlabel('Copper Grade (%)')
            axes[0].set_ylabel('Frequency')
            axes[0].grid(True, alpha=0.3)
            
            # Add statistics
            cu_mean = np.mean(cu_grades)
            cu_std = np.std(cu_grades)
            axes[0].axvline(cu_mean, color='red', linestyle='--', label=f'Mean: {cu_mean:.3f}%')
            axes[0].legend()
            
            # Molybdenum grade histogram
            if 'mo' in self.visualizer.data:
                mo_grades = self.visualizer.data['mo'].flatten()
                axes[1].hist(mo_grades, bins=50, alpha=0.7, color='blue', edgecolor='black')
                axes[1].set_title('Molybdenum Grade Distribution')
                axes[1].set_xlabel('Molybdenum Grade (%)')
                axes[1].set_ylabel('Frequency')
                axes[1].grid(True, alpha=0.3)
                
                mo_mean = np.mean(mo_grades)
                axes[1].axvline(mo_mean, color='red', linestyle='--', label=f'Mean: {mo_mean:.4f}%')
                axes[1].legend()
            else:
                axes[1].text(0.5, 0.5, 'Mo data not available', ha='center', va='center', transform=axes[1].transAxes)
                axes[1].set_title('Molybdenum Grade Distribution')
            
            plt.tight_layout()
            
            # Save the plot
            save_path = save_dir / f'grade_distribution_step_{self.num_timesteps:08d}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            if self.verbose > 0:
                print(f"Could not create grade plot: {e}")
            plt.close('all')
    
    def _create_progress_plot(self, save_dir: Path):
        """Create training progress visualization."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f'Training Progress - Step {self.num_timesteps:,}', fontsize=14)
            
            # Training progress over time (if we have access to logs)
            axes[0, 0].plot([0, self.num_timesteps], [0, 100], 'b-', alpha=0.5)
            axes[0, 0].axvline(self.num_timesteps, color='red', linestyle='--', alpha=0.7)
            axes[0, 0].set_title('Training Progress')
            axes[0, 0].set_xlabel('Training Steps')
            axes[0, 0].set_ylabel('Progress (%)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Model performance (placeholder - would need actual metrics)
            steps = np.arange(0, self.num_timesteps, max(1, self.num_timesteps // 20))
            mock_rewards = np.sin(steps / 1000) * 1000 + np.random.normal(0, 100, len(steps))
            axes[0, 1].plot(steps, mock_rewards, 'g-', alpha=0.7, linewidth=2)
            axes[0, 1].set_title('Episode Rewards (Sample)')
            axes[0, 1].set_xlabel('Training Steps')
            axes[0, 1].set_ylabel('Reward')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Memory usage info
            if PSUTIL_AVAILABLE:
                memory_info = psutil.virtual_memory()
                memory_text = f"""System Resources:
            
Memory Usage: {memory_info.percent:.1f}%
Available: {memory_info.available / (1024**3):.1f} GB
Total: {memory_info.total / (1024**3):.1f} GB

Training Info:
Current Step: {self.num_timesteps:,}
Callback Frequency: {self.visualization_freq:,}
Screenshots Saved: {len(list(self.training_viz_dir.glob('*'))) if hasattr(self, 'training_viz_dir') else 0}
"""
            else:
                memory_text = f"""Training Info:
            
Current Step: {self.num_timesteps:,}
Callback Frequency: {self.visualization_freq:,}
Screenshots Saved: {len(list(self.training_viz_dir.glob('*'))) if hasattr(self, 'training_viz_dir') else 0}

(psutil not available for memory stats)
"""
            axes[1, 0].text(0.05, 0.95, memory_text, transform=axes[1, 0].transAxes, 
                           verticalalignment='top', fontfamily='monospace', fontsize=10)
            axes[1, 0].set_title('System Info')
            axes[1, 0].axis('off')
            
            # Create a simple progress bar
            progress = self.num_timesteps / 50000  # Assuming 50k total steps
            axes[1, 1].barh([0], [progress], height=0.3, color='green', alpha=0.7)
            axes[1, 1].barh([0], [1], height=0.3, color='lightgray', alpha=0.3)
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(-0.5, 0.5)
            axes[1, 1].set_title(f'Overall Progress: {progress*100:.1f}%')
            axes[1, 1].set_xlabel('Progress')
            axes[1, 1].set_yticks([])
            
            plt.tight_layout()
            
            # Save the plot
            save_path = save_dir / f'training_progress_step_{self.num_timesteps:08d}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            if self.verbose > 0:
                print(f"Could not create progress plot: {e}")
            plt.close('all')
    
    def _on_training_end(self) -> None:
        """Called when training ends."""
        if self.visualizer_available:
            print(f"🎬 Training visualization completed!")
            print(f"📁 All screenshots saved to: {self.output_dir}")
            
            # Create a summary of all visualizations
            self._create_training_summary()
    
    def _create_training_summary(self):
        """Create a summary image showing the training progression."""
        try:
            # Find all screenshot directories
            step_dirs = sorted([d for d in self.training_viz_dir.glob('step_*') if d.is_dir()])
            
            if not step_dirs:
                return
            
            print(f"📊 Creating training summary from {len(step_dirs)} visualization steps...")
            
            # Create summary file
            summary_file = self.output_dir / "training_summary.txt"
            with open(summary_file, 'w') as f:
                f.write(f"Training Visualization Summary\n")
                f.write(f"==============================\n\n")
                f.write(f"Total visualization steps: {len(step_dirs)}\n")
                f.write(f"Visualization frequency: Every {self.visualization_freq} training steps\n")
                f.write(f"Output directory: {self.output_dir}\n\n")
                f.write("Visualization steps:\n")
                for step_dir in step_dirs:
                    step_num = step_dir.name.split('_')[1]
                    image_count = len(list(step_dir.glob('*.png')))
                    f.write(f"  Step {int(step_num):8,}: {image_count} images\n")
            
            print(f"📄 Training summary saved to: {summary_file}")
            
        except Exception as e:
            if self.verbose > 0:
                print(f"Could not create training summary: {e}")