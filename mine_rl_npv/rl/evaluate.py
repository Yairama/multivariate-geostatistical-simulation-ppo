"""
Evaluation script for trained MineRL-NPV models.
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

import sys
sys.path.append('..')

from envs.mining_env import make_mining_env


class MiningEvaluator:
    """Evaluator for trained mining RL models."""
    
    def __init__(self, model_path: str, config_path: str, data_path: str = None):
        """Initialize evaluator."""
        self.model_path = model_path
        self.config_path = config_path
        self.data_path = data_path
        
        # Load configuration
        env_config_path = str(Path(config_path).parent / "env.yaml")
        with open(env_config_path, 'r') as f:
            self.env_config = yaml.safe_load(f)
        
        # Load model
        print(f"Loading model from {model_path}")
        self.model = MaskablePPO.load(model_path)
        
        # Create environment
        self.env = make_mining_env(env_config_path, data_path)
        
        print("Evaluator initialized successfully!")
    
    def evaluate_episodes(self, n_episodes: int = 10, deterministic: bool = True) -> Dict:
        """Evaluate model over multiple episodes."""
        print(f"Evaluating model over {n_episodes} episodes...")
        
        episode_results = []
        
        for episode in range(n_episodes):
            result = self.run_single_episode(deterministic=deterministic)
            result['episode'] = episode
            episode_results.append(result)
            
            print(f"Episode {episode + 1}/{n_episodes}: NPV = {result['total_npv']:.2f}, "
                  f"Tonnage = {result['total_tonnage']:.0f}, Steps = {result['steps']}")
        
        # Aggregate results
        aggregated = self._aggregate_results(episode_results)
        
        return {
            'episodes': episode_results,
            'aggregated': aggregated
        }
    
    def run_single_episode(self, deterministic: bool = True, render: bool = False) -> Dict:
        """Run a single episode and collect detailed metrics."""
        obs, info = self.env.reset()
        
        episode_data = {
            'steps': 0,
            'total_npv': 0.0,
            'total_revenue': 0.0,
            'total_costs': 0.0,
            'total_tonnage': 0.0,
            'actions_taken': [],
            'rewards': [],
            'daily_tonnage': [],
            'daily_cu_grade': [],
            'daily_mo_grade': [],
            'daily_npv': [],
            'n_valid_actions': []
        }
        
        done = False
        while not done:
            # Get valid actions
            action_mask = self.env.action_mask()
            valid_actions = np.where(action_mask)[0]
            
            if len(valid_actions) == 0:
                break
            
            # Predict action
            action, _ = self.model.predict(obs, action_masks=action_mask, deterministic=deterministic)
            
            # Take step
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Record data
            episode_data['steps'] += 1
            episode_data['actions_taken'].append(int(action))
            episode_data['rewards'].append(float(reward))
            episode_data['n_valid_actions'].append(len(valid_actions))
            
            # Record daily metrics from info
            if info:
                episode_data['daily_tonnage'].append(info.get('total_mined_tonnage', 0))
                episode_data['daily_cu_grade'].append(info.get('avg_cu_grade', 0))
                episode_data['daily_mo_grade'].append(info.get('avg_mo_grade', 0))
                episode_data['daily_npv'].append(info.get('episode_npv', 0))
        
        # Final metrics
        if info:
            episode_data['total_npv'] = info.get('episode_npv', 0)
            episode_data['total_revenue'] = info.get('total_revenue', 0)
            episode_data['total_costs'] = info.get('total_costs', 0)
            episode_data['total_tonnage'] = info.get('total_mined_tonnage', 0)
            episode_data['avg_cu_grade'] = info.get('avg_cu_grade', 0)
            episode_data['avg_mo_grade'] = info.get('avg_mo_grade', 0)
            episode_data['waste_percentage'] = info.get('waste_percentage', 0)
        
        return episode_data
    
    def _aggregate_results(self, episode_results: List[Dict]) -> Dict:
        """Aggregate results across episodes."""
        metrics = ['total_npv', 'total_revenue', 'total_costs', 'total_tonnage', 
                  'avg_cu_grade', 'avg_mo_grade', 'waste_percentage', 'steps']
        
        aggregated = {}
        
        for metric in metrics:
            values = [ep.get(metric, 0) for ep in episode_results]
            aggregated[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
        
        return aggregated
    
    def benchmark_random_policy(self, n_episodes: int = 10) -> Dict:
        """Benchmark against random policy."""
        print(f"Benchmarking random policy over {n_episodes} episodes...")
        
        episode_results = []
        
        for episode in range(n_episodes):
            obs, info = self.env.reset()
            
            episode_data = {
                'total_npv': 0.0,
                'total_tonnage': 0.0,
                'steps': 0
            }
            
            done = False
            while not done:
                # Get valid actions
                action_mask = self.env.action_mask()
                valid_actions = np.where(action_mask)[0]
                
                if len(valid_actions) == 0:
                    break
                
                # Random action
                action = np.random.choice(valid_actions)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                episode_data['steps'] += 1
            
            # Final metrics
            if info:
                episode_data['total_npv'] = info.get('episode_npv', 0)
                episode_data['total_tonnage'] = info.get('total_mined_tonnage', 0)
            
            episode_results.append(episode_data)
            
            print(f"Random Episode {episode + 1}/{n_episodes}: NPV = {episode_data['total_npv']:.2f}")
        
        return self._aggregate_results(episode_results)
    
    def compare_policies(self, n_episodes: int = 10) -> Dict:
        """Compare RL policy with random policy."""
        print("Comparing RL policy vs Random policy...")
        
        rl_results = self.evaluate_episodes(n_episodes, deterministic=True)
        random_results = self.benchmark_random_policy(n_episodes)
        
        comparison = {
            'rl_policy': rl_results['aggregated'],
            'random_policy': random_results,
            'improvement': {}
        }
        
        # Calculate improvements
        for metric in ['total_npv', 'total_tonnage', 'avg_cu_grade']:
            rl_mean = rl_results['aggregated'][metric]['mean']
            random_mean = random_results[metric]['mean']
            
            if random_mean != 0:
                improvement = ((rl_mean - random_mean) / abs(random_mean)) * 100
            else:
                improvement = 0
            
            comparison['improvement'][metric] = improvement
        
        return comparison
    
    def analyze_mining_sequence(self, episode_data: Dict) -> Dict:
        """Analyze the mining sequence from an episode."""
        analysis = {
            'total_blocks_mined': len(episode_data['actions_taken']),
            'avg_reward_per_block': np.mean(episode_data['rewards']),
            'reward_trend': np.polyfit(range(len(episode_data['rewards'])), episode_data['rewards'], 1)[0],
            'tonnage_per_day': np.mean(np.diff([0] + episode_data['daily_tonnage'])),
            'grade_stability': {
                'cu_std': np.std(episode_data['daily_cu_grade']),
                'mo_std': np.std(episode_data['daily_mo_grade'])
            }
        }
        
        return analysis
    
    def create_evaluation_report(self, results: Dict, save_path: str = None) -> str:
        """Create a comprehensive evaluation report."""
        report = []
        report.append("# MineRL-NPV Model Evaluation Report")
        report.append("=" * 50)
        report.append("")
        
        # Model info
        report.append("## Model Information")
        report.append(f"Model Path: {self.model_path}")
        report.append(f"Configuration: {self.config_path}")
        report.append("")
        
        # Aggregated results
        agg = results['aggregated']
        report.append("## Performance Summary")
        report.append(f"Episodes Evaluated: {len(results['episodes'])}")
        report.append("")
        
        metrics = [
            ('Total NPV', 'total_npv', '$'),
            ('Total Tonnage', 'total_tonnage', 't'),
            ('Avg Cu Grade', 'avg_cu_grade', '%'),
            ('Avg Mo Grade', 'avg_mo_grade', '%'),
            ('Waste Percentage', 'waste_percentage', '%'),
            ('Episode Length', 'steps', 'steps')
        ]
        
        for name, key, unit in metrics:
            values = agg[key]
            report.append(f"{name:20}: {values['mean']:8.2f} ± {values['std']:6.2f} {unit}")
        
        report.append("")
        
        # Episode-by-episode breakdown
        report.append("## Episode Breakdown")
        report.append("Episode | NPV ($) | Tonnage (t) | Cu Grade (%) | Steps")
        report.append("-" * 60)
        
        for ep in results['episodes']:
            report.append(f"{ep['episode']:7} | {ep['total_npv']:7.0f} | "
                         f"{ep['total_tonnage']:11.0f} | {ep.get('avg_cu_grade', 0):12.2f} | "
                         f"{ep['steps']:5}")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to: {save_path}")
        
        return report_text
    
    def plot_results(self, results: Dict, save_dir: str = None):
        """Create visualization plots."""
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract episode data
        episodes = results['episodes']
        
        # 1. NPV distribution
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 3, 1)
        npvs = [ep['total_npv'] for ep in episodes]
        plt.hist(npvs, bins=10, alpha=0.7)
        plt.xlabel('NPV ($)')
        plt.ylabel('Frequency')
        plt.title('NPV Distribution')
        
        # 2. NPV vs Episode Length
        plt.subplot(2, 3, 2)
        steps = [ep['steps'] for ep in episodes]
        plt.scatter(steps, npvs)
        plt.xlabel('Episode Length (steps)')
        plt.ylabel('NPV ($)')
        plt.title('NPV vs Episode Length')
        
        # 3. Tonnage vs NPV
        plt.subplot(2, 3, 3)
        tonnages = [ep['total_tonnage'] for ep in episodes]
        plt.scatter(tonnages, npvs)
        plt.xlabel('Total Tonnage (t)')
        plt.ylabel('NPV ($)')
        plt.title('NPV vs Tonnage')
        
        # 4. Grade distribution
        plt.subplot(2, 3, 4)
        cu_grades = [ep.get('avg_cu_grade', 0) for ep in episodes]
        mo_grades = [ep.get('avg_mo_grade', 0) for ep in episodes]
        plt.scatter(cu_grades, mo_grades)
        plt.xlabel('Avg Cu Grade (%)')
        plt.ylabel('Avg Mo Grade (%)')
        plt.title('Grade Relationship')
        
        # 5. Learning curve (if multiple episodes)
        plt.subplot(2, 3, 5)
        episode_nums = range(len(episodes))
        plt.plot(episode_nums, npvs, 'o-')
        plt.xlabel('Episode')
        plt.ylabel('NPV ($)')
        plt.title('NPV Over Episodes')
        
        # 6. Waste percentage
        plt.subplot(2, 3, 6)
        waste_pcts = [ep.get('waste_percentage', 0) for ep in episodes]
        plt.hist(waste_pcts, bins=10, alpha=0.7)
        plt.xlabel('Waste Percentage (%)')
        plt.ylabel('Frequency')
        plt.title('Waste Distribution')
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(save_dir / 'evaluation_plots.png', dpi=300, bbox_inches='tight')
            print(f"Plots saved to: {save_dir / 'evaluation_plots.png'}")
        
        plt.show()


def main():
    """Main entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate MineRL-NPV model")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model (.zip file)")
    parser.add_argument("--config", type=str, default="configs/train.yaml",
                       help="Path to training configuration")
    parser.add_argument("--data", type=str, default=None,
                       help="Path to block model data")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of episodes to evaluate")
    parser.add_argument("--output", type=str, default="evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--compare", action="store_true",
                       help="Compare with random policy")
    parser.add_argument("--plot", action="store_true",
                       help="Generate plots")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = MiningEvaluator(args.model, args.config, args.data)
    
    # Run evaluation
    results = evaluator.evaluate_episodes(args.episodes)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate report
    report = evaluator.create_evaluation_report(
        results, 
        save_path=output_dir / "evaluation_report.txt"
    )
    print(report)
    
    # Compare with random policy
    if args.compare:
        comparison = evaluator.compare_policies(args.episodes)
        
        print("\n" + "="*50)
        print("COMPARISON WITH RANDOM POLICY")
        print("="*50)
        
        for metric, improvement in comparison['improvement'].items():
            print(f"{metric}: {improvement:+.1f}% improvement")
    
    # Generate plots
    if args.plot:
        evaluator.plot_results(results, save_dir=output_dir)
    
    print(f"\nEvaluation complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()