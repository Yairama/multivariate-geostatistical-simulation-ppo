"""
3D Visualization for mining block models using PyVista.
"""

import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Optional, Tuple
import yaml
from pathlib import Path

import sys
sys.path.append('..')

from geo.loaders import BlockModelLoader


class MiningVisualizer:
    """3D visualizer for mining block models and RL results."""
    
    def __init__(self, config_path: str):
        """Initialize visualizer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.grid_config = self.config['environment']['grid_size']
        self.block_config = self.config['environment']['block_size']
        
        # Create PyVista plotter
        self.plotter = None
        
        # Color schemes
        self.grade_colormap = 'viridis'
        self.state_colors = {
            'unmined': 'lightgray',
            'mined': 'red',
            'oxide': 'orange',
            'sulfide': 'blue'
        }
    
    def load_data(self, data_arrays: Dict[str, np.ndarray]):
        """Load block model data for visualization."""
        self.data = data_arrays
        self.nx, self.ny, self.nz = data_arrays['cu'].shape
        
        # Create coordinate grids
        self.x_coords = np.arange(self.nx) * self.block_config['x']
        self.y_coords = np.arange(self.ny) * self.block_config['y']
        self.z_coords = np.arange(self.nz) * self.block_config['z']
        
        print(f"Loaded data with dimensions: {self.nx} x {self.ny} x {self.nz}")
    
    def create_block_mesh(self, values: np.ndarray, threshold: float = None) -> pv.UnstructuredGrid:
        """Create a mesh of blocks for 3D visualization."""
        # Filter blocks based on threshold
        if threshold is not None:
            mask = values > threshold
        else:
            mask = values > 0  # Only show non-zero blocks
        
        # Get coordinates of blocks to display
        x_indices, y_indices, z_indices = np.where(mask)
        
        if len(x_indices) == 0:
            # Return empty mesh
            return pv.UnstructuredGrid()
        
        # Create mesh
        cells = []
        cell_types = []
        points = []
        cell_values = []
        
        point_id = 0
        for i, (xi, yi, zi) in enumerate(zip(x_indices, y_indices, z_indices)):
            # Block corner coordinates
            x = self.x_coords[xi]
            y = self.y_coords[yi]
            z = self.z_coords[zi]
            
            dx = self.block_config['x']
            dy = self.block_config['y']
            dz = self.block_config['z']
            
            # 8 vertices of the block
            block_points = [
                [x, y, z],
                [x + dx, y, z],
                [x + dx, y + dy, z],
                [x, y + dy, z],
                [x, y, z + dz],
                [x + dx, y, z + dz],
                [x + dx, y + dy, z + dz],
                [x, y + dy, z + dz]
            ]
            
            points.extend(block_points)
            
            # Define hexahedral cell (8 points)
            cell = [8] + list(range(point_id, point_id + 8))
            cells.extend(cell)
            cell_types.append(pv.CellType.HEXAHEDRON)
            
            # Store value for this block
            cell_values.append(values[xi, yi, zi])
            
            point_id += 8
        
        # Create mesh
        mesh = pv.UnstructuredGrid(cells, cell_types, points)
        mesh.cell_data['values'] = np.array(cell_values)
        
        return mesh
    
    def visualize_grades(self, grade_type: str = 'cu', threshold: float = 0.1,
                        save_path: str = None, show_plot: bool = True):
        """Visualize grade distribution in 3D."""
        if grade_type not in self.data:
            raise ValueError(f"Grade type '{grade_type}' not found in data")
        
        grades = self.data[grade_type]
        
        # Create mesh
        mesh = self.create_block_mesh(grades, threshold)
        
        if mesh.n_cells == 0:
            print("No blocks to display with the given threshold")
            return
        
        # Setup plotter
        plotter = pv.Plotter(window_size=(1200, 800))
        
        # Add mesh with grade coloring
        plotter.add_mesh(
            mesh,
            scalars='values',
            cmap=self.grade_colormap,
            opacity=0.8,
            scalar_bar_args={'title': f'{grade_type.upper()} Grade (%)'}
        )
        
        # Add labels and title
        plotter.add_title(f'{grade_type.upper()} Grade Distribution (Threshold: {threshold}%)')
        plotter.show_axes()
        plotter.show_grid()
        
        # Set camera
        plotter.camera_position = 'isometric'
        
        if save_path:
            plotter.screenshot(save_path)
            print(f"Visualization saved to: {save_path}")
        
        if show_plot:
            plotter.show()
        
        return plotter
    
    def visualize_mining_state(self, mining_data: Dict[str, np.ndarray] = None,
                              save_path: str = None, show_plot: bool = True):
        """Visualize mining state (mined vs unmined blocks)."""
        if mining_data is None:
            mining_data = self.data
        
        mined_mask = mining_data['mined_flag']
        
        # Create separate meshes for mined and unmined blocks
        plotter = pv.Plotter(window_size=(1200, 800))
        
        # Unmined blocks (with UPL constraint)
        if 'upl' in mining_data:
            unmined_mask = (~mined_mask) & (mining_data['upl'] > 0)
        else:
            unmined_mask = ~mined_mask
        
        if np.any(unmined_mask):
            # Use tonnage for sizing unmined blocks
            tonnage_values = np.where(unmined_mask, mining_data['ton'], 0)
            unmined_mesh = self.create_block_mesh(tonnage_values, threshold=0)
            
            if unmined_mesh.n_cells > 0:
                plotter.add_mesh(
                    unmined_mesh,
                    color=self.state_colors['unmined'],
                    opacity=0.3,
                    label='Unmined'
                )
        
        # Mined blocks
        if np.any(mined_mask):
            # Color by extraction day
            extraction_days = np.where(mined_mask, mining_data['extraction_day'], 0)
            mined_mesh = self.create_block_mesh(extraction_days, threshold=0)
            
            if mined_mesh.n_cells > 0:
                plotter.add_mesh(
                    mined_mesh,
                    scalars='values',
                    cmap='plasma',
                    opacity=0.8,
                    scalar_bar_args={'title': 'Extraction Day'},
                    label='Mined'
                )
        
        plotter.add_title('Mining State Visualization')
        plotter.show_axes()
        plotter.show_grid()
        plotter.add_legend()
        plotter.camera_position = 'isometric'
        
        if save_path:
            plotter.screenshot(save_path)
            print(f"Mining state visualization saved to: {save_path}")
        
        if show_plot:
            plotter.show()
        
        return plotter
    
    def animate_mining_sequence(self, episode_data: Dict, save_path: str = None):
        """Create animation of mining sequence."""
        actions = episode_data['actions_taken']
        ny = self.ny
        
        # Setup plotter for animation
        plotter = pv.Plotter(window_size=(1200, 800))
        plotter.open_gif(save_path or "mining_animation.gif")
        
        # Initialize state
        mined_state = np.zeros((self.nx, self.ny, self.nz), dtype=bool)
        extraction_days = np.zeros((self.nx, self.ny, self.nz), dtype=int)
        
        # Add initial unmined blocks
        initial_tonnage = self.data['ton'].copy()
        initial_mesh = self.create_block_mesh(initial_tonnage, threshold=0)
        
        frames = []
        
        for day, action in enumerate(actions):
            # Convert action to coordinates
            x = action // ny
            y = action % ny
            
            # Find surface block
            for z in range(self.nz - 1, -1, -1):
                if not mined_state[x, y, z] and self.data['upl'][x, y, z]:
                    mined_state[x, y, z] = True
                    extraction_days[x, y, z] = day + 1
                    break
            
            # Create frame every few days
            if (day + 1) % 5 == 0 or day == len(actions) - 1:
                # Clear plotter
                plotter.clear()
                
                # Add unmined blocks
                unmined_tonnage = np.where(~mined_state, self.data['ton'], 0)
                unmined_mesh = self.create_block_mesh(unmined_tonnage, threshold=0)
                
                if unmined_mesh.n_cells > 0:
                    plotter.add_mesh(
                        unmined_mesh,
                        color=self.state_colors['unmined'],
                        opacity=0.3
                    )
                
                # Add mined blocks
                mined_days = np.where(mined_state, extraction_days, 0)
                mined_mesh = self.create_block_mesh(mined_days, threshold=0)
                
                if mined_mesh.n_cells > 0:
                    plotter.add_mesh(
                        mined_mesh,
                        scalars='values',
                        cmap='plasma',
                        opacity=0.8,
                        scalar_bar_args={'title': 'Extraction Day'}
                    )
                
                plotter.add_title(f'Mining Progress - Day {day + 1}')
                plotter.show_axes()
                plotter.camera_position = 'isometric'
                
                # Write frame
                plotter.write_frame()
        
        # Close animation
        plotter.close()
        
        if save_path:
            print(f"Animation saved to: {save_path}")
    
    def create_grade_cross_sections(self, grade_type: str = 'cu', save_dir: str = None):
        """Create cross-sectional views of grades."""
        grades = self.data[grade_type]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # XY plane (top view) - average over Z
        xy_data = np.mean(grades, axis=2)
        im1 = axes[0, 0].imshow(xy_data.T, origin='lower', cmap=self.grade_colormap)
        axes[0, 0].set_title(f'{grade_type.upper()} Grade - Top View (XY)')
        axes[0, 0].set_xlabel('X (blocks)')
        axes[0, 0].set_ylabel('Y (blocks)')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # XZ plane (side view) - average over Y
        xz_data = np.mean(grades, axis=1)
        im2 = axes[0, 1].imshow(xz_data.T, origin='lower', cmap=self.grade_colormap)
        axes[0, 1].set_title(f'{grade_type.upper()} Grade - Side View (XZ)')
        axes[0, 1].set_xlabel('X (blocks)')
        axes[0, 1].set_ylabel('Z (blocks)')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # YZ plane (front view) - average over X
        yz_data = np.mean(grades, axis=0)
        im3 = axes[1, 0].imshow(yz_data.T, origin='lower', cmap=self.grade_colormap)
        axes[1, 0].set_title(f'{grade_type.upper()} Grade - Front View (YZ)')
        axes[1, 0].set_xlabel('Y (blocks)')
        axes[1, 0].set_ylabel('Z (blocks)')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # Grade histogram
        axes[1, 1].hist(grades.flatten(), bins=50, alpha=0.7)
        axes[1, 1].set_title(f'{grade_type.upper()} Grade Distribution')
        axes[1, 1].set_xlabel(f'{grade_type.upper()} Grade (%)')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir) / f'{grade_type}_cross_sections.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cross sections saved to: {save_path}")
        
        plt.show()
        
        return fig
    
    def create_economic_visualization(self, save_dir: str = None):
        """Visualize economic potential of blocks."""
        # Calculate simple NPV per block
        cu_grade = self.data['cu']
        mo_grade = self.data['mo']
        tonnage = self.data['ton']
        recovery = self.data.get('rec', np.full_like(cu_grade, 85.0)) / 100.0
        
        # Economic parameters
        economics = self.config['economics']
        cu_price = economics['prices']['cu_price_usd_per_lb']
        mo_price = economics['prices']['mo_price_usd_per_lb']
        mining_cost = economics['costs']['mining_cost']
        processing_cost = economics['costs']['processing_cost']
        
        # Revenue per block
        cu_revenue = tonnage * (cu_grade / 100.0) * recovery * cu_price * 2204.62
        mo_revenue = tonnage * (mo_grade / 100.0) * recovery * mo_price * 2204.62
        total_revenue = cu_revenue + mo_revenue
        
        # Costs per block
        total_costs = tonnage * (mining_cost + processing_cost)
        
        # Net value
        net_value = total_revenue - total_costs
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Revenue map
        revenue_xy = np.mean(total_revenue, axis=2)
        im1 = axes[0, 0].imshow(revenue_xy.T, origin='lower', cmap='Greens')
        axes[0, 0].set_title('Total Revenue per Block (Top View)')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Cost map
        cost_xy = np.mean(total_costs, axis=2)
        im2 = axes[0, 1].imshow(cost_xy.T, origin='lower', cmap='Reds')
        axes[0, 1].set_title('Total Costs per Block (Top View)')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Net value map
        npv_xy = np.mean(net_value, axis=2)
        im3 = axes[1, 0].imshow(npv_xy.T, origin='lower', cmap='RdBu_r')
        axes[1, 0].set_title('Net Value per Block (Top View)')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # NPV distribution
        axes[1, 1].hist(net_value.flatten(), bins=50, alpha=0.7)
        axes[1, 1].axvline(0, color='red', linestyle='--', label='Break-even')
        axes[1, 1].set_title('Net Value Distribution')
        axes[1, 1].set_xlabel('Net Value ($)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir) / 'economic_visualization.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Economic visualization saved to: {save_path}")
        
        plt.show()
        
        return fig, {'revenue': total_revenue, 'costs': total_costs, 'net_value': net_value}


def main():
    """Test the visualizer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize mining block model")
    parser.add_argument("--config", type=str, default="configs/env.yaml",
                       help="Environment configuration file")
    parser.add_argument("--data", type=str, default="data/sample_model.csv",
                       help="Block model data file")
    parser.add_argument("--output", type=str, default="visualizations",
                       help="Output directory for visualizations")
    parser.add_argument("--grade", type=str, default="cu",
                       help="Grade type to visualize")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    loader = BlockModelLoader(args.config)
    data = loader.load_and_preprocess(args.data)
    
    # Create visualizer
    visualizer = MiningVisualizer(args.config)
    visualizer.load_data(data)
    
    # Generate visualizations
    print("Creating grade visualization...")
    visualizer.visualize_grades(
        grade_type=args.grade,
        save_path=str(output_dir / f'{args.grade}_3d.png'),
        show_plot=False
    )
    
    print("Creating cross sections...")
    visualizer.create_grade_cross_sections(
        grade_type=args.grade,
        save_dir=str(output_dir)
    )
    
    print("Creating economic visualization...")
    visualizer.create_economic_visualization(save_dir=str(output_dir))
    
    print("Creating mining state visualization...")
    visualizer.visualize_mining_state(
        save_path=str(output_dir / 'mining_state.png'),
        show_plot=False
    )
    
    print(f"Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()