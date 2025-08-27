#!/usr/bin/env python3
"""
Standalone evaluation script for trained MineRL-NPV models.

This script provides both headless and visualization modes for evaluation.
- Headless mode: Evaluation without 3D visualization (console logs only)
- Visualization mode: Evaluation with 3D visualization capabilities

Usage:
    # Headless mode (default)
    python evaluate_model.py --model experiments/runs/run_20231201_120000/models/best_model.zip --data data/sample

    # Visualization mode
    python evaluate_model.py --model experiments/runs/run_20231201_120000/models/best_model.zip --data data/sample --visualization

    # Custom evaluation
    python evaluate_model.py --model best_model.zip --data data/sample --episodes 50 --output results/
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
import numpy as np

# Add the mine_rl_npv package to the Python path
script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir / "mine_rl_npv"))

# Set environment for headless rendering if needed
def setup_headless_environment():
    """Setup environment variables for headless operation."""
    os.environ['PYVISTA_OFF_SCREEN'] = 'true'
    os.environ['DISPLAY'] = ''
    # Suppress Qt and visualization warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='pyvista')
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='vtk')


def setup_visualization_environment():
    """Setup environment for visualization mode."""
    # Ensure we can use interactive plotting
    import matplotlib
    if 'DISPLAY' not in os.environ or not os.environ['DISPLAY']:
        matplotlib.use('Agg')  # Use non-interactive backend as fallback
    else:
        matplotlib.use('TkAgg')  # Use interactive backend when possible


def run_visualization_demo(evaluator, results, output_dir):
    """Run visualization demonstrations in visualization mode."""
    try:
        from viz.viewer import MiningVisualizer
        
        print("\n🎨 Generating 3D visualizations...")
        
        # Create visualizer
        env_config_path = str(Path(evaluator.config_path).parent / "env.yaml")
        visualizer = MiningVisualizer(env_config_path)
        
        # Load data
        visualizer.load_data(evaluator.env.unwrapped.data)
        
        # Generate various visualizations
        vis_dir = Path(output_dir) / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Grade visualization
        print("  📊 Creating grade visualizations...")
        visualizer.visualize_grades(
            grade_type='cu',
            save_path=str(vis_dir / 'copper_grades_3d.png'),
            show_plot=False
        )
        
        # 2. Mining state visualization
        if 'episodes' in results and results['episodes']:
            print("  ⛏️  Creating mining state visualization...")
            # Use data from first episode as example
            episode_data = results['episodes'][0]
            
            # Create mining state visualization
            visualizer.visualize_mining_state(
                save_path=str(vis_dir / 'mining_state.png'),
                show_plot=False
            )
        
        # 3. Economic visualization
        print("  💰 Creating economic visualization...")
        visualizer.create_economic_visualization(
            save_dir=str(vis_dir)
        )
        
        print(f"  ✅ Visualizations saved to: {vis_dir}")
        
    except Exception as e:
        print(f"  ⚠️  Warning: Could not generate 3D visualizations: {e}")


def main():
    """Main entry point for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained MineRL-NPV model with headless/visualization modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation (headless mode)
  python evaluate_model.py --model best_model.zip --data data/sample

  # Evaluation with visualization
  python evaluate_model.py --model best_model.zip --data data/sample --visualization

  # Custom evaluation with comparison
  python evaluate_model.py --model best_model.zip --data data/sample --episodes 100 --compare --plot
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        help="Path to trained model file (.zip)"
    )
    
    parser.add_argument(
        "--data", 
        type=str, 
        required=True,
        help="Path to block model data directory or file"
    )
    
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--headless", 
        action="store_true", 
        default=True,
        help="Run in headless mode (no 3D visualization, console logs only) - DEFAULT"
    )
    mode_group.add_argument(
        "--visualization", 
        action="store_true", 
        default=False,
        help="Run with visualization mode (enable 3D visualization capabilities)"
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=10,
        help="Number of episodes to evaluate (default: 10)"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="evaluation_results",
        help="Output directory for results (default: evaluation_results)"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Path to training configuration (auto-detected if not provided)"
    )
    
    # Analysis options
    parser.add_argument(
        "--compare", 
        action="store_true",
        help="Compare with random policy baseline"
    )
    
    parser.add_argument(
        "--plot", 
        action="store_true",
        help="Generate evaluation plots"
    )
    
    parser.add_argument(
        "--deterministic", 
        action="store_true", 
        default=True,
        help="Use deterministic policy evaluation (default: True)"
    )
    
    parser.add_argument(
        "--verbose", 
        type=int, 
        default=1,
        choices=[0, 1, 2],
        help="Verbosity level (0: silent, 1: info, 2: debug)"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Determine mode
    if args.visualization:
        headless_mode = False
    else:
        headless_mode = True
    
    # Setup environment based on mode
    if headless_mode:
        print("🤖 Running in HEADLESS mode (no 3D visualization)")
        setup_headless_environment()
    else:
        print("👁️  Running in VISUALIZATION mode (3D visualization enabled)")
        setup_visualization_environment()
    
    # Validate paths
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Error: Model file not found: {model_path}")
        sys.exit(1)
    
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ Error: Data path not found: {data_path}")
        sys.exit(1)
    
    # Auto-detect config if not provided
    config_path = args.config
    if config_path is None:
        # Try to find config in model directory
        model_dir = model_path.parent.parent
        potential_config = model_dir / "config.yaml"
        if potential_config.exists():
            config_path = str(potential_config)
        else:
            # Use default config
            config_path = "mine_rl_npv/configs/train.yaml"
    
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"❌ Error: Configuration file not found: {config_path}")
        print("Please specify --config or ensure the config exists in the model directory")
        sys.exit(1)
    
    # Import evaluation modules (after environment setup)
    try:
        from rl.evaluate import MiningEvaluator
        import torch
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"❌ Error importing required modules: {e}")
        print("Make sure you're running from the repository root and all dependencies are installed.")
        sys.exit(1)
    
    print(f"🏆 Model path: {model_path}")
    print(f"📊 Data path: {data_path}")
    print(f"⚙️  Config path: {config_path}")
    print(f"📁 Output directory: {args.output}")
    
    # Set random seed
    print(f"🎲 Setting random seed: {args.seed}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    try:
        # Create evaluator
        print("🚀 Initializing evaluator...")
        evaluator = MiningEvaluator(str(model_path), str(config_path), str(data_path))
        
        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run evaluation
        print(f"🎯 Evaluating model over {args.episodes} episodes...")
        print("=" * 60)
        
        results = evaluator.evaluate_episodes(args.episodes, deterministic=args.deterministic)
        
        print("=" * 60)
        print("✅ Evaluation completed successfully!")
        
        # Generate report
        print("\n📄 Generating evaluation report...")
        report = evaluator.create_evaluation_report(
            results, 
            save_path=output_dir / "evaluation_report.txt"
        )
        print(report)
        
        # Compare with random policy if requested
        if args.compare:
            print("\n🔄 Comparing with random policy...")
            try:
                comparison = evaluator.compare_policies(args.episodes)
                
                print("\n" + "="*50)
                print("COMPARISON WITH RANDOM POLICY")
                print("="*50)
                
                for metric, improvement in comparison['improvement'].items():
                    print(f"{metric}: {improvement:+.1f}% improvement")
                    
            except Exception as e:
                print(f"⚠️  Warning: Could not run policy comparison: {e}")
        
        # Generate plots if requested
        if args.plot:
            print("\n📊 Generating evaluation plots...")
            try:
                evaluator.plot_results(results, save_dir=output_dir)
                print(f"  ✅ Plots saved to: {output_dir}")
            except Exception as e:
                print(f"  ⚠️  Warning: Could not generate plots: {e}")
        
        # Generate 3D visualizations in visualization mode
        if not headless_mode:
            run_visualization_demo(evaluator, results, output_dir)
        
        print(f"\n📁 All results saved to: {output_dir}")
        
        # Summary statistics
        if 'episodes' in results and results['episodes']:
            npvs = [ep['total_npv'] for ep in results['episodes']]
            print(f"\n📊 Summary Statistics:")
            print(f"  Average NPV: ${np.mean(npvs):,.2f}")
            print(f"  Std NPV: ${np.std(npvs):,.2f}")
            print(f"  Best NPV: ${np.max(npvs):,.2f}")
            print(f"  Worst NPV: ${np.min(npvs):,.2f}")
        
        # Next steps information
        if not headless_mode:
            print("\n🎨 Visualization files created:")
            print(f"  - 3D visualizations: {output_dir}/visualizations/")
            print(f"  - Evaluation plots: {output_dir}/evaluation_plots.png")
        else:
            print("\n💡 To see visualizations, run with --visualization flag")
        
        return results
        
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Evaluation failed with error: {e}")
        if args.verbose >= 2:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()