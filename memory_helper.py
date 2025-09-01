#!/usr/bin/env python3
"""
Memory Configuration Helper for MineRL-NPV Training

This script helps users choose the right configuration based on their memory constraints.
"""

import sys
import os
from pathlib import Path

# Add the mine_rl_npv package to the Python path
script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir / "mine_rl_npv"))

import yaml
import argparse


def calculate_memory_requirements(train_config, env_config):
    """Calculate memory requirements for a given configuration."""
    n_steps = train_config['training']['hyperparameters']['n_steps']
    n_envs = train_config['env_settings']['n_envs']
    
    # Count channels
    state_config = env_config['state']
    n_channels = (
        len(state_config.get('geological_features', [])) +
        len(state_config.get('mineralogy_features', [])) +
        len(state_config.get('dynamic_features', []))
    )
    
    # Grid dimensions
    grid_size = env_config['environment']['grid_size']
    nx, ny, nz = grid_size['nx'], grid_size['ny'], grid_size['nz']
    
    # Calculate memory
    total_elements = n_steps * n_envs * n_channels * nx * ny * nz
    memory_gb = total_elements * 4 / (1024**3)  # 4 bytes per float32
    
    return {
        'n_steps': n_steps,
        'n_envs': n_envs,
        'grid_size': (nx, ny, nz),
        'channels': n_channels,
        'total_elements': total_elements,
        'memory_gb': memory_gb
    }


def load_config_info():
    """Load information about all available configurations."""
    configs = {
        'full': {
            'train': 'mine_rl_npv/configs/train.yaml',
            'env': 'mine_rl_npv/configs/env.yaml',
            'description': 'Full-scale training with maximum model capacity',
            'recommended_for': 'High-end GPUs with 32GB+ VRAM'
        },
        'memory_optimized': {
            'train': 'mine_rl_npv/configs/train_memory_optimized.yaml',
            'env': 'mine_rl_npv/configs/env_memory_optimized.yaml',
            'description': 'Balanced configuration with moderate memory usage',
            'recommended_for': 'Mid-range GPUs with 8-16GB VRAM'
        },
        'ultra_light': {
            'train': 'mine_rl_npv/configs/train_ultra_light.yaml',
            'env': 'mine_rl_npv/configs/env_ultra_light.yaml',
            'description': 'Minimal memory usage for CPU or low-end GPUs',
            'recommended_for': 'CPU training or GPUs with <8GB VRAM'
        },
        'small': {
            'train': 'examples/train_small.yaml',
            'env': 'examples/env_small.yaml',
            'description': 'Testing configuration with small grid',
            'recommended_for': 'Quick testing and development'
        }
    }
    
    return configs


def print_config_comparison():
    """Print a comparison of all configurations."""
    configs = load_config_info()
    
    print("=" * 80)
    print("MEMORY CONFIGURATION COMPARISON")
    print("=" * 80)
    print()
    
    results = []
    
    for name, config_info in configs.items():
        try:
            # Load configurations
            with open(config_info['train'], 'r') as f:
                train_config = yaml.safe_load(f)
            with open(config_info['env'], 'r') as f:
                env_config = yaml.safe_load(f)
            
            # Calculate memory
            memory_info = calculate_memory_requirements(train_config, env_config)
            memory_info['name'] = name
            memory_info['config_info'] = config_info
            results.append(memory_info)
            
        except FileNotFoundError as e:
            print(f"⚠️  Config '{name}' not found: {e}")
            continue
    
    # Sort by memory usage
    results.sort(key=lambda x: x['memory_gb'])
    
    # Print comparison table
    print(f"{'Config':<20} {'Memory (GB)':<12} {'Grid Size':<12} {'Steps':<6} {'Envs':<5} {'Description'}")
    print("-" * 80)
    
    for result in results:
        name = result['name']
        memory_gb = result['memory_gb']
        grid_size = f"{result['grid_size'][0]}×{result['grid_size'][1]}×{result['grid_size'][2]}"
        n_steps = result['n_steps']
        n_envs = result['n_envs']
        description = result['config_info']['description'][:40] + "..." if len(result['config_info']['description']) > 40 else result['config_info']['description']
        
        print(f"{name:<20} {memory_gb:<12.2f} {grid_size:<12} {n_steps:<6} {n_envs:<5} {description}")
    
    print()
    print("DETAILED RECOMMENDATIONS:")
    print("-" * 40)
    
    for result in results:
        name = result['name']
        config_info = result['config_info']
        memory_gb = result['memory_gb']
        
        print(f"\n🔧 {name.upper()}:")
        print(f"   Memory: {memory_gb:.2f} GB")
        print(f"   {config_info['description']}")
        print(f"   💡 {config_info['recommended_for']}")
        
        # Training command
        train_path = config_info['train']
        print(f"   📝 Command:")
        print(f"      python3 train_model.py --config {train_path} --data mine_rl_npv/data/sample_model.csv --visualization")


def recommend_config(available_memory_gb):
    """Recommend a configuration based on available memory."""
    configs = load_config_info()
    
    print(f"\n🔍 RECOMMENDATION FOR {available_memory_gb} GB MEMORY:")
    print("-" * 50)
    
    recommendations = []
    
    for name, config_info in configs.items():
        try:
            with open(config_info['train'], 'r') as f:
                train_config = yaml.safe_load(f)
            with open(config_info['env'], 'r') as f:
                env_config = yaml.safe_load(f)
            
            memory_info = calculate_memory_requirements(train_config, env_config)
            
            if memory_info['memory_gb'] <= available_memory_gb:
                recommendations.append((name, config_info, memory_info))
                
        except FileNotFoundError:
            continue
    
    if not recommendations:
        print("❌ No configurations fit within your memory constraint.")
        print("   Consider using the ultra_light configuration or reducing batch sizes manually.")
        return
    
    # Sort by memory usage (descending) to recommend the most capable config that fits
    recommendations.sort(key=lambda x: x[2]['memory_gb'], reverse=True)
    
    best_name, best_config, best_memory = recommendations[0]
    
    print(f"✅ RECOMMENDED: {best_name}")
    print(f"   Memory usage: {best_memory['memory_gb']:.2f} GB (fits in {available_memory_gb} GB)")
    print(f"   {best_config['description']}")
    print(f"   Grid size: {best_memory['grid_size'][0]}×{best_memory['grid_size'][1]}×{best_memory['grid_size'][2]}")
    print(f"   Steps: {best_memory['n_steps']}, Envs: {best_memory['n_envs']}")
    print()
    print(f"📝 COMMAND TO RUN:")
    print(f"   python3 train_model.py --config {best_config['train']} --data mine_rl_npv/data/sample_model.csv --visualization")
    
    if len(recommendations) > 1:
        print(f"\n📋 OTHER COMPATIBLE OPTIONS:")
        for name, config_info, memory_info in recommendations[1:]:
            print(f"   • {name}: {memory_info['memory_gb']:.2f} GB")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Memory Configuration Helper for MineRL-NPV Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show all configurations
  python memory_helper.py --compare
  
  # Get recommendation for 8GB memory
  python memory_helper.py --memory 8
  
  # Get recommendation for 32GB memory  
  python memory_helper.py --memory 32
        """
    )
    
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all available configurations"
    )
    
    parser.add_argument(
        "--memory",
        type=float,
        help="Available memory in GB to get recommendations"
    )
    
    args = parser.parse_args()
    
    if not args.compare and args.memory is None:
        # Default: show comparison
        print_config_comparison()
    elif args.compare:
        print_config_comparison()
    elif args.memory is not None:
        recommend_config(args.memory)


if __name__ == "__main__":
    main()