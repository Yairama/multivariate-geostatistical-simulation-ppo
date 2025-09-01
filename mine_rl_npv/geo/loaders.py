"""
Data loading and preprocessing utilities for mining block models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import yaml
from pathlib import Path


class BlockModelLoader:
    """Loads and preprocesses block model data for RL training."""
    
    def __init__(self, config_path: str):
        """Initialize loader with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.grid_config = self.config['environment']['grid_size']
        self.block_config = self.config['environment']['block_size']
        self.state_config = self.config['state']
        
    def load_csv(self, csv_path: str) -> pd.DataFrame:
        """Load block model from CSV file."""
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} blocks from {csv_path}")
        print(f"Columns: {list(df.columns)}")
        return df
        
    def preprocess_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert real-world coordinates to grid indices."""
        df = df.copy()
        
        # Get coordinate bounds
        x_min, x_max = df['x'].min(), df['x'].max()
        y_min, y_max = df['y'].min(), df['y'].max()
        z_min, z_max = df['z'].min(), df['z'].max()
        
        print(f"Coordinate bounds:")
        print(f"  X: {x_min} to {x_max}")
        print(f"  Y: {y_min} to {y_max}")
        print(f"  Z: {z_min} to {z_max}")
        
        # Convert to grid indices
        df['grid_x'] = ((df['x'] - x_min) / self.block_config['x']).astype(int)
        df['grid_y'] = ((df['y'] - y_min) / self.block_config['y']).astype(int)
        df['grid_z'] = ((df['z'] - z_min) / self.block_config['z']).astype(int)
        
        # Store bounds for later use
        self.coordinate_bounds = {
            'x_min': x_min, 'x_max': x_max,
            'y_min': y_min, 'y_max': y_max,
            'z_min': z_min, 'z_max': z_max
        }
        
        return df
    
    def create_3d_arrays(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Convert DataFrame to 3D arrays for RL environment."""
        # Get actual grid dimensions from data
        max_x = df['grid_x'].max() + 1
        max_y = df['grid_y'].max() + 1
        max_z = df['grid_z'].max() + 1
        
        print(f"Actual grid dimensions: {max_x} x {max_y} x {max_z}")
        
        # Initialize arrays
        arrays = {}
        
        # Geological features
        geo_features = self.state_config['geological_features']
        for feature in geo_features:
            if feature in df.columns:
                arrays[feature] = np.zeros((max_x, max_y, max_z))
                
        # Mineralogy features
        mineral_features = self.state_config['mineralogy_features']
        for feature in mineral_features:
            if feature in df.columns:
                arrays[feature] = np.zeros((max_x, max_y, max_z))
        
        # Tonnage (always included)
        arrays['ton'] = np.zeros((max_x, max_y, max_z))
        
        # Fill arrays with data
        for _, row in df.iterrows():
            x, y, z = int(row['grid_x']), int(row['grid_y']), int(row['grid_z'])
            
            for feature in arrays.keys():
                if feature in row:
                    arrays[feature][x, y, z] = row[feature]
        
        # Add dynamic state arrays (initialized to zero)
        arrays['mined_flag'] = np.zeros((max_x, max_y, max_z), dtype=bool)
        arrays['extraction_day'] = np.zeros((max_x, max_y, max_z), dtype=int)
        arrays['destination'] = np.zeros((max_x, max_y, max_z), dtype=int)
        
        # Calculate UPL if not present
        if 'upl' not in arrays:
            arrays['upl'] = self._calculate_upl(arrays)
            
        return arrays
    
    def _calculate_upl(self, arrays: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate Ultimate Pit Limit using simplified economic criteria."""
        print("Calculating UPL using economic criteria...")
        
        # Get economic parameters
        economics = self.config['economics']
        cu_price = economics['prices']['cu_price_usd_per_lb']
        mining_cost = economics['costs']['mining_cost']
        processing_cost = economics['costs']['processing_cost']
        
        # Simple NPV calculation per block
        cu_grade = arrays['cu']
        tonnage = arrays['ton']
        recovery = arrays.get('rec', np.full_like(cu_grade, 85.0)) / 100.0  # Convert percentage
        
        # Revenue per block (fixed to match environment calculation)
        # Environment: cu_content = tonnage * (cu_grade / 100.0) * recovery; cu_revenue = cu_content * cu_price * 2204.62
        cu_content = tonnage * (cu_grade / 100.0) * recovery  # Tonnes of recoverable Cu
        total_revenue = cu_content * cu_price * 2204.62  # Convert tonnes to pounds
        
        # Costs per block
        total_costs = (mining_cost + processing_cost) * tonnage
        
        # Net value per block
        net_value = total_revenue - total_costs
        
        # UPL = blocks with positive net value
        upl = (net_value > 0).astype(int)
        
        print(f"UPL calculation complete. Positive blocks: {upl.sum()}/{upl.size}")
        return upl
    
    def normalize_features(self, arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Normalize features for neural network input."""
        if self.state_config['normalization']['method'] == 'none':
            return arrays
            
        normalized = arrays.copy()
        
        # Features to normalize (exclude flags and indices)
        features_to_normalize = (
            self.state_config['geological_features'] + 
            self.state_config['mineralogy_features'] + 
            ['ton']
        )
        
        for feature in features_to_normalize:
            if feature in normalized:
                data = normalized[feature]
                
                if self.state_config['normalization']['method'] == 'standardize':
                    # Z-score normalization
                    mean = np.mean(data[data > 0])  # Only consider non-zero values
                    std = np.std(data[data > 0])
                    if std > 0:
                        normalized[feature] = (data - mean) / std
                        
                        # Clip outliers
                        if self.state_config['normalization']['clip_outliers']:
                            clip_val = self.state_config['normalization']['outlier_std']
                            normalized[feature] = np.clip(normalized[feature], -clip_val, clip_val)
                
                elif self.state_config['normalization']['method'] == 'minmax':
                    # Min-max normalization
                    min_val = np.min(data[data > 0])
                    max_val = np.max(data)
                    if max_val > min_val:
                        normalized[feature] = (data - min_val) / (max_val - min_val)
        
        return normalized
    
    def load_and_preprocess(self, csv_path: str) -> Dict[str, np.ndarray]:
        """Complete pipeline: load CSV and convert to 3D arrays."""
        print(f"Loading and preprocessing {csv_path}...")
        
        # Load CSV
        df = self.load_csv(csv_path)
        
        # Preprocess coordinates
        df = self.preprocess_coordinates(df)
        
        # Create 3D arrays
        arrays = self.create_3d_arrays(df)
        
        # Normalize features
        arrays = self.normalize_features(arrays)
        
        print("Preprocessing complete!")
        return arrays


def create_synthetic_data(config_path: str, nx: int = 50, ny: int = 50, nz: int = 25) -> Dict[str, np.ndarray]:
    """Create synthetic block model data for testing."""
    print(f"Creating synthetic data with dimensions {nx}x{ny}x{nz}")
    
    # Create coordinate grids
    x = np.arange(nx)
    y = np.arange(ny) 
    z = np.arange(nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Create synthetic geological patterns
    # Copper grade - decreases with depth, has spatial correlation
    cu_grade = 0.5 * np.exp(-Z/10) + 0.2 * np.sin(X/5) * np.cos(Y/5) + 0.1 * np.random.normal(0, 1, (nx, ny, nz))
    cu_grade = np.maximum(cu_grade, 0)
    
    # Molybdenum - different spatial pattern
    mo_grade = 0.1 * np.exp(-(X-nx/2)**2/100 - (Y-ny/2)**2/100) + 0.05 * np.random.normal(0, 1, (nx, ny, nz))
    mo_grade = np.maximum(mo_grade, 0)
    
    # Other features
    arrays = {
        'cu': cu_grade,
        'mo': mo_grade,
        'as': 0.01 * np.random.exponential(1, (nx, ny, nz)),
        'clays': 5.0 + 3.0 * np.random.normal(0, 1, (nx, ny, nz)),
        'bwi': 12.0 + 2.0 * np.random.normal(0, 1, (nx, ny, nz)),
        'rec': 85.0 + 10.0 * np.random.normal(0, 1, (nx, ny, nz)),
        'ton': np.full((nx, ny, nz), 1000.0),  # 1000 tonnes per block
        'chalcocite': 0.1 * np.random.exponential(1, (nx, ny, nz)),
        'bornite': 0.2 * np.random.exponential(1, (nx, ny, nz)),
        'chalcopyrite': 0.5 * np.random.exponential(1, (nx, ny, nz)),
        'tennantite': 0.05 * np.random.exponential(1, (nx, ny, nz)),
        'molibdenite': 0.1 * np.random.exponential(1, (nx, ny, nz)),
        'pyrite': 2.0 * np.random.exponential(1, (nx, ny, nz)),
        'mined_flag': np.zeros((nx, ny, nz), dtype=bool),
        'extraction_day': np.zeros((nx, ny, nz), dtype=int),
        'destination': np.zeros((nx, ny, nz), dtype=int),
        'upl': np.ones((nx, ny, nz), dtype=int)  # All blocks in UPL for testing
    }
    
    # Ensure realistic bounds
    arrays['clays'] = np.clip(arrays['clays'], 0, 20)
    arrays['bwi'] = np.clip(arrays['bwi'], 8, 20)
    arrays['rec'] = np.clip(arrays['rec'], 60, 95)
    
    print("Synthetic data created successfully!")
    return arrays


if __name__ == "__main__":
    # Test the loader
    config_path = "configs/env.yaml"
    loader = BlockModelLoader(config_path)
    
    # Test with real data
    try:
        arrays = loader.load_and_preprocess("data/sample_model.csv")
        print(f"Loaded arrays with shapes:")
        for key, array in arrays.items():
            print(f"  {key}: {array.shape}")
    except Exception as e:
        print(f"Error loading real data: {e}")
        print("Creating synthetic data instead...")
        arrays = create_synthetic_data(config_path)