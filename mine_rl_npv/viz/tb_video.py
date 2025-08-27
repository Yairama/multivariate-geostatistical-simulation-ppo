"""
TensorBoard video logging for mining visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import io
import cv2
from pathlib import Path
from typing import Dict, List, Optional
import torch
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append('..')

from viz.viewer import MiningVisualizer


class TensorBoardVideoLogger:
    """Logger for creating and uploading mining visualization videos to TensorBoard."""
    
    def __init__(self, log_dir: str, config_path: str):
        """Initialize video logger."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(str(self.log_dir))
        self.visualizer = MiningVisualizer(config_path)
        
        # Video settings
        self.fps = 2  # Frames per second for mining sequence
        self.frame_size = (800, 600)
    
    def log_mining_episode_video(self, episode_data: Dict, data_arrays: Dict[str, np.ndarray], 
                                 step: int, tag: str = "mining/episode_video"):
        """Create and log a video of the mining episode."""
        self.visualizer.load_data(data_arrays)
        
        # Create frames for the mining sequence
        frames = self._create_mining_frames(episode_data, data_arrays)
        
        if len(frames) > 0:
            # Convert frames to video tensor
            video_tensor = self._frames_to_tensor(frames)
            
            # Log to TensorBoard
            self.writer.add_video(tag, video_tensor, step, fps=self.fps)
            
            print(f"Logged mining episode video to TensorBoard: {tag}")
        else:
            print("No frames generated for mining episode video")
    
    def log_grade_visualization(self, data_arrays: Dict[str, np.ndarray], 
                               step: int, grade_type: str = 'cu'):
        """Log static grade visualization images."""
        self.visualizer.load_data(data_arrays)
        
        # Create grade cross-sections
        fig = self.visualizer.create_grade_cross_sections(grade_type, save_dir=None)
        
        # Convert to image and log
        image_tensor = self._fig_to_tensor(fig)
        self.writer.add_image(f'grades/{grade_type}_cross_sections', image_tensor, step)
        
        plt.close(fig)
        
        # Create economic visualization
        fig, _ = self.visualizer.create_economic_visualization(save_dir=None)
        image_tensor = self._fig_to_tensor(fig)
        self.writer.add_image('economics/block_values', image_tensor, step)
        
        plt.close(fig)
    
    def log_episode_metrics(self, episode_data: Dict, step: int):
        """Log episode metrics as scalars and histograms."""
        # Scalar metrics
        metrics = {
            'episode/total_npv': episode_data.get('total_npv', 0),
            'episode/total_tonnage': episode_data.get('total_tonnage', 0),
            'episode/avg_cu_grade': episode_data.get('avg_cu_grade', 0),
            'episode/avg_mo_grade': episode_data.get('avg_mo_grade', 0),
            'episode/waste_percentage': episode_data.get('waste_percentage', 0),
            'episode/steps': episode_data.get('steps', 0),
            'episode/avg_reward': np.mean(episode_data.get('rewards', [0]))
        }
        
        for tag, value in metrics.items():
            self.writer.add_scalar(tag, value, step)
        
        # Histograms
        if 'rewards' in episode_data and len(episode_data['rewards']) > 0:
            self.writer.add_histogram('episode/reward_distribution', 
                                    np.array(episode_data['rewards']), step)
        
        if 'daily_cu_grade' in episode_data and len(episode_data['daily_cu_grade']) > 0:
            self.writer.add_histogram('episode/daily_cu_grades', 
                                    np.array(episode_data['daily_cu_grade']), step)
        
        # Trend plots
        if len(episode_data.get('daily_npv', [])) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(episode_data['daily_npv'])
            ax.set_title('Cumulative NPV Over Time')
            ax.set_xlabel('Day')
            ax.set_ylabel('NPV ($)')
            
            image_tensor = self._fig_to_tensor(fig)
            self.writer.add_image('episode/npv_trend', image_tensor, step)
            plt.close(fig)
    
    def _create_mining_frames(self, episode_data: Dict, data_arrays: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """Create frames showing the mining sequence."""
        actions = episode_data.get('actions_taken', [])
        if not actions:
            return []
        
        ny = data_arrays['cu'].shape[1]
        nx, ny, nz = data_arrays['cu'].shape
        
        frames = []
        
        # Initialize state
        mined_state = np.zeros((nx, ny, nz), dtype=bool)
        extraction_days = np.zeros((nx, ny, nz), dtype=int)
        
        # Sample frames throughout the episode
        frame_interval = max(1, len(actions) // 20)  # Max 20 frames
        
        for day, action in enumerate(actions):
            # Convert action to coordinates
            x = action // ny
            y = action % ny
            
            # Find surface block and mine it
            for z in range(nz - 1, -1, -1):
                if not mined_state[x, y, z] and data_arrays['upl'][x, y, z]:
                    mined_state[x, y, z] = True
                    extraction_days[x, y, z] = day + 1
                    break
            
            # Create frame every few days
            if day % frame_interval == 0 or day == len(actions) - 1:
                frame = self._create_2d_frame(data_arrays, mined_state, extraction_days, day + 1)
                frames.append(frame)
        
        return frames
    
    def _create_2d_frame(self, data_arrays: Dict[str, np.ndarray], 
                        mined_state: np.ndarray, extraction_days: np.ndarray, 
                        current_day: int) -> np.ndarray:
        """Create a 2D frame showing the mining state."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Top view - mining state
        mined_2d = np.any(mined_state, axis=2)
        cu_grades_2d = np.mean(data_arrays['cu'], axis=2)
        
        # Show Cu grades for unmined blocks, mining day for mined blocks
        display_data = np.where(mined_2d, 
                               np.max(extraction_days, axis=2), 
                               cu_grades_2d)
        
        im1 = axes[0, 0].imshow(display_data.T, origin='lower', cmap='viridis')
        axes[0, 0].set_title(f'Mining Progress - Day {current_day}')
        axes[0, 0].set_xlabel('X (blocks)')
        axes[0, 0].set_ylabel('Y (blocks)')
        
        # Cu grade distribution (mined vs unmined)
        mined_cu = data_arrays['cu'][mined_state]
        unmined_cu = data_arrays['cu'][~mined_state & (data_arrays['upl'] > 0)]
        
        axes[0, 1].hist(unmined_cu, bins=30, alpha=0.7, label='Unmined', color='gray')
        if len(mined_cu) > 0:
            axes[0, 1].hist(mined_cu, bins=30, alpha=0.7, label='Mined', color='red')
        axes[0, 1].set_title('Cu Grade Distribution')
        axes[0, 1].set_xlabel('Cu Grade (%)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        
        # Cumulative tonnage
        if current_day > 1:
            daily_tonnage = []
            for day in range(1, current_day + 1):
                day_mask = extraction_days == day
                tonnage = np.sum(data_arrays['ton'][day_mask])
                daily_tonnage.append(tonnage)
            
            axes[1, 0].plot(range(1, current_day + 1), np.cumsum(daily_tonnage))
            axes[1, 0].set_title('Cumulative Tonnage Mined')
            axes[1, 0].set_xlabel('Day')
            axes[1, 0].set_ylabel('Total Tonnage (t)')
        
        # NPV progression (if available in episode data)
        if 'daily_npv' in self.__dict__ and hasattr(self, 'current_episode_data'):
            npv_data = self.current_episode_data.get('daily_npv', [])
            if len(npv_data) >= current_day:
                axes[1, 1].plot(range(1, min(current_day + 1, len(npv_data) + 1)), 
                               npv_data[:current_day])
                axes[1, 1].set_title('NPV Progression')
                axes[1, 1].set_xlabel('Day')
                axes[1, 1].set_ylabel('NPV ($)')
        
        plt.tight_layout()
        
        # Convert to numpy array
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return frame
    
    def _frames_to_tensor(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Convert list of frames to PyTorch tensor for TensorBoard video."""
        if not frames:
            return torch.zeros(1, 1, 3, 100, 100)  # Dummy tensor
        
        # Resize frames to consistent size
        resized_frames = []
        target_height, target_width = self.frame_size[1], self.frame_size[0]
        
        for frame in frames:
            if frame.shape[:2] != (target_height, target_width):
                frame_resized = cv2.resize(frame, self.frame_size)
            else:
                frame_resized = frame
            resized_frames.append(frame_resized)
        
        # Stack frames: (T, H, W, C) -> (1, T, C, H, W)
        video_array = np.stack(resized_frames)
        video_tensor = torch.from_numpy(video_array).permute(0, 3, 1, 2).unsqueeze(0)
        
        return video_tensor.float() / 255.0  # Normalize to [0, 1]
    
    def _fig_to_tensor(self, fig) -> torch.Tensor:
        """Convert matplotlib figure to PyTorch tensor."""
        fig.canvas.draw()
        
        # Get image as numpy array
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # Convert to tensor: (H, W, C) -> (C, H, W)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image_tensor
    
    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()


class EpisodeVideoCallback:
    """Callback for logging episode videos during training."""
    
    def __init__(self, log_dir: str, config_path: str, video_freq: int = 1000):
        """Initialize callback."""
        self.video_logger = TensorBoardVideoLogger(log_dir, config_path)
        self.video_freq = video_freq
        self.episode_count = 0
    
    def on_episode_end(self, episode_data: Dict, data_arrays: Dict[str, np.ndarray], 
                      timestep: int):
        """Called at the end of each episode."""
        self.episode_count += 1
        
        # Log metrics every episode
        self.video_logger.log_episode_metrics(episode_data, timestep)
        
        # Log video periodically
        if self.episode_count % self.video_freq == 0:
            self.video_logger.log_mining_episode_video(
                episode_data, data_arrays, timestep
            )
            self.video_logger.log_grade_visualization(data_arrays, timestep)
    
    def close(self):
        """Close the callback."""
        self.video_logger.close()


def create_demo_video(config_path: str, data_path: str, output_path: str = "demo_video.mp4"):
    """Create a demonstration video of the mining environment."""
    from envs.mining_env import make_mining_env
    
    # Create environment
    env = make_mining_env(config_path, data_path)
    
    # Create video logger
    video_logger = TensorBoardVideoLogger("demo_logs", config_path)
    
    # Run a random episode
    obs, info = env.reset()
    episode_data = {
        'actions_taken': [],
        'rewards': [],
        'daily_npv': [],
        'daily_cu_grade': [],
        'daily_tonnage': []
    }
    
    done = False
    while not done and len(episode_data['actions_taken']) < 100:  # Limit episode length
        # Get valid actions
        action_mask = env.action_mask()
        valid_actions = np.where(action_mask)[0]
        
        if len(valid_actions) == 0:
            break
        
        # Random action
        action = np.random.choice(valid_actions)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Record data
        episode_data['actions_taken'].append(action)
        episode_data['rewards'].append(reward)
        
        if info:
            episode_data['daily_npv'].append(info.get('episode_npv', 0))
            episode_data['daily_cu_grade'].append(info.get('avg_cu_grade', 0))
            episode_data['daily_tonnage'].append(info.get('total_mined_tonnage', 0))
    
    # Get environment data
    data_arrays = env.unwrapped.data
    
    # Create video
    video_logger.current_episode_data = episode_data  # Store for frame creation
    video_logger.log_mining_episode_video(episode_data, data_arrays, 0)
    
    video_logger.close()
    env.close()
    
    print(f"Demo video logged to TensorBoard in: demo_logs")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create demo video")
    parser.add_argument("--config", type=str, default="configs/env.yaml",
                       help="Environment configuration")
    parser.add_argument("--data", type=str, default=None,
                       help="Block model data path")
    parser.add_argument("--output", type=str, default="demo_video.mp4",
                       help="Output video path")
    
    args = parser.parse_args()
    
    create_demo_video(args.config, args.data, args.output)