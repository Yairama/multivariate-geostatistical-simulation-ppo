"""
Synthetic block model generator for testing and demonstration.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from typing import Tuple, Dict
import yaml


class SyntheticGenerator:
    """Generate synthetic porphyry-style deposit for testing."""
    
    def __init__(self, config_path: str):
        """Initialize generator with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def generate_porphyry_deposit(self, nx: int = 50, ny: int = 50, nz: int = 25, 
                                 seed: int = 42) -> Dict[str, np.ndarray]:
        """
        Generate a synthetic porphyry-style copper-molybdenum deposit.
        
        Args:
            nx, ny, nz: Grid dimensions
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary of block model arrays
        """
        np.random.seed(seed)
        
        print(f"Generating synthetic porphyry deposit: {nx}x{ny}x{nz}")
        
        # Create coordinate grids
        x = np.arange(nx)
        y = np.arange(ny)
        z = np.arange(nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Center coordinates
        cx, cy, cz = nx//2, ny//2, nz//2
        
        # Distance from center
        r_xy = np.sqrt((X - cx)**2 + (Y - cy)**2)
        r_3d = np.sqrt((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2)
        
        # Generate copper grades
        cu_grade = self._generate_copper_grades(X, Y, Z, cx, cy, cz, nx, ny, nz)
        
        # Generate molybdenum grades
        mo_grade = self._generate_molybdenum_grades(X, Y, Z, cx, cy, cz, nx, ny, nz)
        
        # Generate other elements
        as_grade = self._generate_arsenic_grades(cu_grade, mo_grade)
        clays = self._generate_clays(Z, nz)
        bwi = self._generate_bwi(cu_grade, clays)
        rec = self._generate_recovery(cu_grade, as_grade, clays)
        
        # Generate mineralogy
        mineralogy = self._generate_mineralogy(cu_grade, mo_grade, Z, nz)
        
        # Tonnage (constant for simplicity)
        tonnage = np.full((nx, ny, nz), 1000.0)
        
        # Calculate UPL using simplified criteria
        upl = self._calculate_upl(cu_grade, mo_grade, tonnage, rec)
        
        # Create data dictionary
        data = {
            'cu': cu_grade,
            'mo': mo_grade,
            'as': as_grade,
            'clays': clays,
            'bwi': bwi,
            'rec': rec,
            'ton': tonnage,
            'chalcocite': mineralogy['chalcocite'],
            'bornite': mineralogy['bornite'],
            'chalcopyrite': mineralogy['chalcopyrite'],
            'tennantite': mineralogy['tennantite'],
            'molibdenite': mineralogy['molibdenite'],
            'pyrite': mineralogy['pyrite'],
            'upl': upl,
            'mined_flag': np.zeros((nx, ny, nz), dtype=bool),
            'extraction_day': np.zeros((nx, ny, nz), dtype=int),
            'destination': np.zeros((nx, ny, nz), dtype=int)
        }
        
        print("Synthetic deposit generation complete!")
        self._print_deposit_stats(data)
        
        return data
    
    def _generate_copper_grades(self, X, Y, Z, cx, cy, cz, nx, ny, nz):
        """Generate realistic copper grade distribution."""
        # Main orebody - elliptical with depth decay
        cu_main = 1.5 * np.exp(-((X - cx)**2 / (15**2) + (Y - cy)**2 / (15**2)))
        cu_main *= np.exp(-Z / 20)  # Decay with depth
        
        # Secondary high-grade zones
        # Zone 1 - offset from center
        cu_zone1 = 0.8 * np.exp(-((X - cx + 10)**2 / (8**2) + (Y - cy + 5)**2 / (8**2)))
        cu_zone1 *= np.exp(-(Z - 10)**2 / (5**2))
        
        # Zone 2 - deeper zone
        cu_zone2 = 0.6 * np.exp(-((X - cx - 8)**2 / (6**2) + (Y - cy - 8)**2 / (6**2)))
        cu_zone2 *= np.exp(-(Z - 5)**2 / (8**2))
        
        # Combine zones
        cu_grade = cu_main + cu_zone1 + cu_zone2
        
        # Add correlated noise
        noise = 0.2 * np.random.normal(0, 1, (nx, ny, nz))
        noise = gaussian_filter(noise, sigma=1.5)  # Spatial correlation
        
        cu_grade += noise
        
        # Ensure non-negative and realistic bounds
        cu_grade = np.clip(cu_grade, 0, 3.0)
        
        return cu_grade
    
    def _generate_molybdenum_grades(self, X, Y, Z, cx, cy, cz, nx, ny, nz):
        """Generate molybdenum grades (typically deeper and more centralized)."""
        # Mo is typically deeper and more centralized than Cu
        mo_grade = 0.3 * np.exp(-((X - cx)**2 / (10**2) + (Y - cy)**2 / (10**2)))
        mo_grade *= np.exp(-(Z - 8)**2 / (12**2))  # Peak slightly deeper
        
        # Add uncorrelated noise
        noise = 0.05 * np.random.normal(0, 1, (nx, ny, nz))
        noise = gaussian_filter(noise, sigma=1.0)
        
        mo_grade += noise
        mo_grade = np.clip(mo_grade, 0, 0.5)
        
        return mo_grade
    
    def _generate_arsenic_grades(self, cu_grade, mo_grade):
        """Generate arsenic grades (correlated with Cu and Mo)."""
        # Arsenic is often associated with Cu and Mo mineralization
        as_grade = 0.01 * (cu_grade + 2 * mo_grade)
        as_grade += 0.005 * np.random.exponential(1, cu_grade.shape)
        as_grade = np.clip(as_grade, 0, 0.2)
        
        return as_grade
    
    def _generate_clays(self, Z, nz):
        """Generate clay content (typically higher near surface)."""
        # Clay content increases near surface (weathering)
        surface_effect = np.exp(-(Z - nz + 5) / 8)
        
        clays = 2.0 + 8.0 * surface_effect
        clays += 2.0 * np.random.normal(0, 1, Z.shape)
        clays = np.clip(clays, 0, 20)
        
        return clays
    
    def _generate_bwi(self, cu_grade, clays):
        """Generate Ball Work Index (related to rock hardness)."""
        # BWI typically correlates with alteration (higher Cu = softer rock)
        # But also higher clay = harder processing
        
        base_bwi = 15.0 - 2.0 * cu_grade + 0.2 * clays
        base_bwi += 1.5 * np.random.normal(0, 1, cu_grade.shape)
        bwi = np.clip(base_bwi, 8, 25)
        
        return bwi
    
    def _generate_recovery(self, cu_grade, as_grade, clays):
        """Generate metallurgical recovery."""
        # Recovery typically decreases with clay content and arsenic
        # But increases with grade (better liberation)
        
        base_recovery = 90 - 10 * clays / 20 - 50 * as_grade + 5 * cu_grade
        base_recovery += 5 * np.random.normal(0, 1, cu_grade.shape)
        recovery = np.clip(base_recovery, 50, 95)
        
        return recovery
    
    def _generate_mineralogy(self, cu_grade, mo_grade, Z, nz):
        """Generate mineral abundances."""
        # Different copper minerals dominate at different depths
        # Chalcocite (secondary enrichment) - near surface
        chalcocite = 0.5 * cu_grade * np.exp(-(Z - nz + 8) / 10)
        chalcocite += 0.1 * np.random.exponential(1, cu_grade.shape)
        
        # Bornite (intermediate depth)
        bornite = 0.3 * cu_grade * np.exp(-(Z - 15)**2 / (8**2))
        bornite += 0.05 * np.random.exponential(1, cu_grade.shape)
        
        # Chalcopyrite (primary mineral) - deeper
        chalcopyrite = 0.8 * cu_grade * (1 - np.exp(-Z / 10))
        chalcopyrite += 0.1 * np.random.exponential(1, cu_grade.shape)
        
        # Tennantite (As-bearing)
        tennantite = 0.2 * cu_grade * (Z / nz)  # Deeper
        tennantite += 0.02 * np.random.exponential(1, cu_grade.shape)
        
        # Molibdenite
        molibdenite = 0.6 * mo_grade
        molibdenite += 0.05 * np.random.exponential(1, cu_grade.shape)
        
        # Pyrite (widespread)
        pyrite = 1.0 + 2.0 * cu_grade + 0.5 * np.random.exponential(1, cu_grade.shape)
        
        # Ensure all are non-negative
        mineralogy = {
            'chalcocite': np.clip(chalcocite, 0, None),
            'bornite': np.clip(bornite, 0, None),
            'chalcopyrite': np.clip(chalcopyrite, 0, None),
            'tennantite': np.clip(tennantite, 0, None),
            'molibdenite': np.clip(molibdenite, 0, None),
            'pyrite': np.clip(pyrite, 0, None)
        }
        
        return mineralogy
    
    def _calculate_upl(self, cu_grade, mo_grade, tonnage, recovery):
        """Calculate Ultimate Pit Limit based on economic criteria."""
        # Economic parameters from config
        economics = self.config['economics']
        cu_price = economics['prices']['cu_price_usd_per_lb']
        mo_price = economics['prices']['mo_price_usd_per_lb']
        mining_cost = economics['costs']['mining_cost']
        processing_cost = economics['costs']['processing_cost']
        
        # Revenue calculation
        cu_revenue = tonnage * (cu_grade / 100) * (recovery / 100) * cu_price * 2204.62
        mo_revenue = tonnage * (mo_grade / 100) * (recovery / 100) * mo_price * 2204.62
        total_revenue = cu_revenue + mo_revenue
        
        # Cost calculation
        total_costs = tonnage * (mining_cost + processing_cost)
        
        # Net value
        net_value = total_revenue - total_costs
        
        # UPL = blocks with positive net value
        upl = (net_value > 0).astype(int)
        
        return upl
    
    def _print_deposit_stats(self, data):
        """Print statistics about the generated deposit."""
        print("\nDeposit Statistics:")
        print(f"Cu grade: {data['cu'].mean():.3f} ± {data['cu'].std():.3f}% (max: {data['cu'].max():.3f}%)")
        print(f"Mo grade: {data['mo'].mean():.3f} ± {data['mo'].std():.3f}% (max: {data['mo'].max():.3f}%)")
        print(f"As grade: {data['as'].mean():.4f} ± {data['as'].std():.4f}% (max: {data['as'].max():.4f}%)")
        print(f"Clay content: {data['clays'].mean():.1f} ± {data['clays'].std():.1f}%")
        print(f"BWI: {data['bwi'].mean():.1f} ± {data['bwi'].std():.1f}")
        print(f"Recovery: {data['rec'].mean():.1f} ± {data['rec'].std():.1f}%")
        print(f"UPL blocks: {data['upl'].sum():,} / {data['upl'].size:,} ({100*data['upl'].mean():.1f}%)")
    
    def save_as_csv(self, data: Dict[str, np.ndarray], output_path: str):
        """Save synthetic data as CSV file."""
        # Create coordinate arrays
        nx, ny, nz = data['cu'].shape
        coords = []
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    row = {
                        'x': i * self.config['environment']['block_size']['x'],
                        'y': j * self.config['environment']['block_size']['y'],
                        'z': k * self.config['environment']['block_size']['z']
                    }
                    
                    # Add all other data
                    for key, array in data.items():
                        if key not in ['mined_flag', 'extraction_day', 'destination']:
                            row[key] = array[i, j, k]
                    
                    coords.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(coords)
        df.to_csv(output_path, index=False)
        
        print(f"Synthetic data saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic mining data")
    parser.add_argument("--config", type=str, default="configs/env.yaml",
                       help="Environment configuration file")
    parser.add_argument("--output", type=str, default="data/synthetic_model.csv",
                       help="Output CSV file path")
    parser.add_argument("--nx", type=int, default=40,
                       help="Grid size X")
    parser.add_argument("--ny", type=int, default=40,
                       help="Grid size Y")
    parser.add_argument("--nz", type=int, default=20,
                       help="Grid size Z")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Generate synthetic data
    generator = SyntheticGenerator(args.config)
    data = generator.generate_porphyry_deposit(args.nx, args.ny, args.nz, args.seed)
    
    # Save as CSV
    generator.save_as_csv(data, args.output)