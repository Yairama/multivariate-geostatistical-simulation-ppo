"""
Mining Environment for Reinforcement Learning with action masking.
Compatible with MaskablePPO from sb3-contrib.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any
import yaml
from pathlib import Path

# Fix imports for package structure
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from geo.loaders import BlockModelLoader, create_synthetic_data


class MiningEnv(gym.Env):
    """
    Gymnasium environment for mining planning with MaskablePPO.
    
    Action space: Discrete(nx * ny) - choose surface column to mine
    Observation space: Box - 3D tensor with geological and operational features
    Reward: Economic value (revenue - costs) of mined blocks
    """
    
    def __init__(self, config_path: str, data_path: Optional[str] = None):
        super().__init__()
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.env_config = self.config['environment']
        self.economics_config = self.config['economics']
        self.constraints_config = self.config['constraints']
        self.state_config = self.config['state']
        self.reward_config = self.config['reward']
        
        # Load data
        if data_path:
            loader = BlockModelLoader(config_path)
            self.data = loader.load_and_preprocess(data_path)
        else:
            # Create synthetic data for testing
            self.data = create_synthetic_data(config_path)
        
        # Get dimensions
        self.nx, self.ny, self.nz = self.data['cu'].shape
        
        # Define action space - choose column (x, y)
        self.action_space = spaces.Discrete(self.nx * self.ny)
        
        # Define observation space - 3D tensor with multiple channels
        self.feature_names = self._get_feature_names()
        self.n_features = len(self.feature_names)
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_features, self.nx, self.ny, self.nz),
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
        
    def _get_feature_names(self) -> list:
        """Get list of feature names for observation."""
        features = []
        features.extend(self.state_config['geological_features'])
        features.extend(self.state_config['mineralogy_features'])
        features.extend(['ton'])  # Always include tonnage
        features.extend(self.state_config['dynamic_features'])
        features.append('upl')  # Always include UPL
        return features
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Reset dynamic state
        self.current_day = 0
        self.total_mined_tonnage = 0
        self.total_revenue = 0
        self.total_costs = 0
        self.episode_npv = 0
        
        # Reset dynamic arrays
        self.data['mined_flag'] = np.zeros((self.nx, self.ny, self.nz), dtype=bool)
        self.data['extraction_day'] = np.zeros((self.nx, self.ny, self.nz), dtype=int)
        self.data['destination'] = np.zeros((self.nx, self.ny, self.nz), dtype=int)
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        # Convert action to coordinates
        x = action // self.ny
        y = action % self.ny
        
        # Check if action is valid (not masked)
        action_mask = self.action_mask()
        if not action_mask[action]:
            # Invalid action - return penalty
            reward = self.reward_config['shaping']['invalid_action_penalty']
            observation = self._get_observation()
            info = self._get_info()
            return observation, reward, False, False, info
        
        # Find the topmost unmined block in this column
        z = self._get_surface_block(x, y)
        
        if z is None:
            # No blocks available in this column
            reward = self.reward_config['shaping']['invalid_action_penalty']
            observation = self._get_observation()
            info = self._get_info()
            return observation, reward, False, False, info
        
        # Mine the block
        reward = self._mine_block(x, y, z)
        self.current_day += 1
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.current_day >= 1000  # Maximum episode length
        
        # Get new observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_surface_block(self, x: int, y: int) -> Optional[int]:
        """Get the topmost unmined block in column (x, y)."""
        for z in range(self.nz - 1, -1, -1):  # Start from top
            if not self.data['mined_flag'][x, y, z] and self.data['upl'][x, y, z]:
                # Check precedence constraints
                if self._check_precedence(x, y, z):
                    return z
        return None
    
    def _check_precedence(self, x: int, y: int, z: int) -> bool:
        """Check if block satisfies precedence constraints."""
        if not self.constraints_config['precedence']['enable_precedence']:
            return True
        
        # For simplicity, implement basic precedence:
        # Can only mine block if all blocks above it are mined or invalid
        for z_above in range(z + 1, self.nz):
            if self.data['upl'][x, y, z_above] and not self.data['mined_flag'][x, y, z_above]:
                return False
        
        # Cross-shape precedence (simplified)
        # Check adjacent blocks at same or higher levels
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Cross shape
        for dx, dy in offsets:
            nx_adj, ny_adj = x + dx, y + dy
            if 0 <= nx_adj < self.nx and 0 <= ny_adj < self.ny:
                for z_check in range(z, self.nz):
                    if (self.data['upl'][nx_adj, ny_adj, z_check] and 
                        not self.data['mined_flag'][nx_adj, ny_adj, z_check]):
                        # Adjacent block exists and is not mined - check if it should be mined first
                        if z_check > z:  # Only blocks above current level matter
                            return False
        
        return True
    
    def _mine_block(self, x: int, y: int, z: int) -> float:
        """Mine a block and calculate reward."""
        # Mark block as mined
        self.data['mined_flag'][x, y, z] = True
        self.data['extraction_day'][x, y, z] = self.current_day
        
        # Get block properties
        tonnage = self.data['ton'][x, y, z]
        cu_grade = self.data['cu'][x, y, z]
        mo_grade = self.data['mo'][x, y, z]
        rec = self.data['rec'][x, y, z] / 100.0  # Convert percentage to decimal
        bwi = self.data['bwi'][x, y, z]
        clays = self.data['clays'][x, y, z]
        
        # Calculate revenue
        revenue = self._calculate_revenue(tonnage, cu_grade, mo_grade, rec)
        
        # Calculate costs
        costs = self._calculate_costs(tonnage, bwi, clays)
        
        # Net reward for this block
        block_reward = revenue - costs
        
        # Apply discount factor (daily)
        if self.economics_config['discount']['convert_to_daily']:
            annual_rate = self.economics_config['discount']['annual_rate']
            daily_discount = (1 + annual_rate) ** (1/365)
            discount_factor = 1 / (daily_discount ** self.current_day)
            block_reward *= discount_factor
        
        # Update episode totals
        self.total_mined_tonnage += tonnage
        self.total_revenue += revenue
        self.total_costs += costs
        self.episode_npv += block_reward
        
        return float(block_reward)
    
    def _calculate_revenue(self, tonnage: float, cu_grade: float, mo_grade: float, recovery: float) -> float:
        """Calculate revenue from a mined block."""
        # Copper revenue
        cu_price = self.economics_config['prices']['cu_price_usd_per_lb']
        cu_content = tonnage * (cu_grade / 100.0) * recovery  # Tonnes of recoverable Cu
        cu_revenue = cu_content * cu_price * 2204.62  # Convert tonnes to pounds
        
        # Molybdenum revenue (if enabled)
        mo_revenue = 0
        if self.reward_config['revenue']['include_mo_revenue']:
            mo_price = self.economics_config['prices']['mo_price_usd_per_lb']
            mo_content = tonnage * (mo_grade / 100.0) * recovery  # Tonnes of recoverable Mo
            mo_revenue = mo_content * mo_price * 2204.62  # Convert tonnes to pounds
        
        return cu_revenue + mo_revenue
    
    def _calculate_costs(self, tonnage: float, bwi: float, clays: float) -> float:
        """Calculate costs for mining and processing a block."""
        costs = self.economics_config['costs']
        
        # Base costs
        mining_cost = tonnage * costs['mining_cost']
        processing_cost = tonnage * costs['processing_cost']
        fixed_cost = tonnage * costs['fixed_cost']
        
        # Processing modifiers
        modifier_cost = 0
        if self.reward_config['cost']['include_processing_modifiers']:
            # BWI penalty (higher BWI = harder grinding = higher cost)
            if bwi > 15:
                bwi_penalty = (bwi - 15) * self.economics_config['processing_modifiers']['bwi_factor']
                modifier_cost += tonnage * bwi_penalty
            
            # Clay penalty (higher clay = processing difficulties)
            if clays > 3:
                clay_penalty = (clays - 3) * self.economics_config['processing_modifiers']['clay_penalty']
                modifier_cost += tonnage * clay_penalty
        
        return mining_cost + processing_cost + fixed_cost + modifier_cost
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        # Terminate if no more mineable blocks
        if not np.any(self.action_mask()):
            return True
        
        # Terminate if daily capacity constraints violated consistently
        daily_capacity = self.constraints_config['daily_capacity']
        if (self.total_mined_tonnage < daily_capacity['min_tonnes'] * self.current_day * 0.8 and 
            self.current_day > 10):  # Allow some flexibility in early days
            return True
        
        return False
    
    def action_mask(self) -> np.ndarray:
        """Get mask of valid actions (required for MaskablePPO)."""
        mask = np.zeros(self.action_space.n, dtype=bool)
        
        for action in range(self.action_space.n):
            x = action // self.ny
            y = action % self.ny
            
            # Check if there's a mineable block in this column
            surface_z = self._get_surface_block(x, y)
            if surface_z is not None:
                # Check capacity constraints
                if self._check_daily_capacity():
                    mask[action] = True
        
        return mask
    
    def _check_daily_capacity(self) -> bool:
        """Check if mining more blocks would violate daily capacity."""
        daily_capacity = self.constraints_config['daily_capacity']
        
        # Estimate tonnes per block (use average)
        avg_tonnage = np.mean(self.data['ton'][self.data['ton'] > 0])
        
        # Check if adding another block would exceed capacity
        projected_tonnage = self.total_mined_tonnage + avg_tonnage
        max_allowed = daily_capacity['max_tonnes'] * (self.current_day + 1)
        
        return projected_tonnage <= max_allowed
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation state."""
        observation = np.zeros((self.n_features, self.nx, self.ny, self.nz), dtype=np.float32)
        
        for i, feature_name in enumerate(self.feature_names):
            if feature_name in self.data:
                feature_data = self.data[feature_name]
                if feature_data.dtype == bool:
                    observation[i] = feature_data.astype(np.float32)
                else:
                    observation[i] = feature_data.astype(np.float32)
        
        return observation
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information for logging."""
        return {
            'current_day': self.current_day,
            'total_mined_tonnage': self.total_mined_tonnage,
            'total_revenue': self.total_revenue,
            'total_costs': self.total_costs,
            'episode_npv': self.episode_npv,
            'n_valid_actions': np.sum(self.action_mask()),
            'avg_cu_grade': self._calculate_avg_grade('cu'),
            'avg_mo_grade': self._calculate_avg_grade('mo'),
            'waste_percentage': self._calculate_waste_percentage()
        }
    
    def _calculate_avg_grade(self, grade_type: str) -> float:
        """Calculate average grade of mined material."""
        mined_mask = self.data['mined_flag']
        if not np.any(mined_mask):
            return 0.0
        
        grades = self.data[grade_type][mined_mask]
        tonnages = self.data['ton'][mined_mask]
        
        # Weighted average by tonnage
        if np.sum(tonnages) > 0:
            return np.average(grades, weights=tonnages)
        else:
            return 0.0
    
    def _calculate_waste_percentage(self) -> float:
        """Calculate percentage of mined material that is waste."""
        mined_mask = self.data['mined_flag']
        if not np.any(mined_mask):
            return 0.0
        
        mined_tonnage = np.sum(self.data['ton'][mined_mask])
        if mined_tonnage == 0:
            return 0.0
        
        # Define waste as material with very low grade (< 0.1% Cu equivalent)
        cu_grades = self.data['cu'][mined_mask]
        mo_grades = self.data['mo'][mined_mask]
        
        # Simple Cu equivalent calculation
        cu_equiv = cu_grades + mo_grades * 5  # Mo is roughly 5x more valuable than Cu
        waste_mask = cu_equiv < 0.1
        
        waste_tonnage = np.sum(self.data['ton'][mined_mask][waste_mask])
        return (waste_tonnage / mined_tonnage) * 100.0


# Wrapper for sb3-contrib compatibility
from sb3_contrib.common.wrappers import ActionMasker

def make_mining_env(config_path: str, data_path: Optional[str] = None):
    """Create a mining environment with action masking for MaskablePPO."""
    env = MiningEnv(config_path, data_path)
    env = ActionMasker(env, lambda env: env.action_mask())
    return env


if __name__ == "__main__":
    # Test the environment
    import sys
    sys.path.append('..')
    
    config_path = "mine_rl_npv/configs/env.yaml"
    
    print("Creating mining environment...")
    env = make_mining_env(config_path)
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test reset
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    # Test a few steps
    for i in range(5):
        action_mask = env.action_mask()
        valid_actions = np.where(action_mask)[0]
        
        if len(valid_actions) > 0:
            action = np.random.choice(valid_actions)
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"Step {i+1}:")
            print(f"  Action: {action}, Reward: {reward:.2f}")
            print(f"  Valid actions: {len(valid_actions)}")
            print(f"  NPV so far: {info['episode_npv']:.2f}")
            
            if terminated or truncated:
                print("Episode finished!")
                break
        else:
            print("No valid actions available!")
            break