#!/usr/bin/env python3
"""
Test result comparison tool for regression detection.

This tool saves and compares test results across runs to detect
performance regressions or accuracy changes.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

sys.path.insert(0, '/home/user/Documents/poker_knightNG/src')

from poker_knight_ng import solve_poker_hand


class TestResultComparator:
    """Compare test results across runs to detect regressions."""
    
    def __init__(self, results_dir: str = "tests/results"):
        """Initialize comparator with results directory."""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.baseline_file = self.results_dir / "baseline.json"
        self.history_file = self.results_dir / "history.json"
    
    def run_standard_tests(self) -> Dict[str, Any]:
        """Run standard test scenarios and collect results."""
        timestamp = datetime.now().isoformat()
        results = {
            'timestamp': timestamp,
            'scenarios': {},
            'performance': {},
            'accuracy': {}
        }
        
        # Define standard test scenarios
        scenarios = [
            {
                'name': 'AA_vs_1_preflop',
                'hand': ['A♠', 'A♥'],
                'opponents': 1,
                'board': None,
                'mode': 'default'
            },
            {
                'name': 'KK_vs_2_flop',
                'hand': ['K♥', 'K♦'],
                'opponents': 2,
                'board': ['Q♣', '7♦', '2♠'],
                'mode': 'default'
            },
            {
                'name': 'flush_draw',
                'hand': ['A♦', 'K♦'],
                'opponents': 1,
                'board': ['Q♦', '7♦', '2♣'],
                'mode': 'default'
            },
            {
                'name': 'multiway_6',
                'hand': ['T♥', 'T♣'],
                'opponents': 5,
                'board': None,
                'mode': 'default'
            },
            {
                'name': 'icm_bubble',
                'hand': ['A♠', 'K♠'],
                'opponents': 3,
                'board': None,
                'mode': 'default',
                'tournament_context': {
                    'payouts': [0.5, 0.3, 0.2],
                    'players_remaining': 4,
                    'average_stack': 5000
                },
                'stack_sizes': [5000, 5000, 5000, 5000]
            },
            {
                'name': 'board_texture_wet',
                'hand': ['Q♥', 'Q♦'],
                'opponents': 2,
                'board': ['J♠', 'T♠', '9♠'],
                'mode': 'default'
            },
            {
                'name': 'position_button',
                'hand': ['K♣', 'Q♣'],
                'opponents': 3,
                'board': ['A♦', 'K♦', '7♣'],
                'mode': 'default',
                'hero_position': 'button',
                'players_to_act': 0
            },
            {
                'name': 'spr_calculation',
                'hand': ['J♥', 'J♦'],
                'opponents': 1,
                'board': ['T♣', '8♦', '2♥'],
                'mode': 'default',
                'stack_sizes': [1000, 1200],
                'pot_size': 200,
                'bet_size': 0.5
            }
        ]
        
        # Warm-up run to initialize GPU
        print("Warming up GPU...")
        _ = solve_poker_hand(['2♣', '3♦'], 1, None, 'fast')
        
        print("Running standard test scenarios...")
        for scenario in scenarios:
            print(f"  Testing {scenario['name']}...", end='', flush=True)
            
            # Run multiple times for timing
            times = []
            win_probs = []
            
            for _ in range(3):
                start = time.time()
                # Build kwargs with optional parameters
                kwargs = {
                    'hero_hand': scenario['hand'],
                    'num_opponents': scenario['opponents'],
                    'board_cards': scenario.get('board'),
                    'simulation_mode': scenario['mode']
                }
                
                # Add optional parameters if present
                if 'tournament_context' in scenario:
                    kwargs['tournament_context'] = scenario['tournament_context']
                if 'stack_sizes' in scenario:
                    kwargs['stack_sizes'] = scenario['stack_sizes']
                if 'hero_position' in scenario:
                    kwargs['hero_position'] = scenario['hero_position']
                if 'players_to_act' in scenario:
                    kwargs['players_to_act'] = scenario['players_to_act']
                if 'pot_size' in scenario:
                    kwargs['pot_size'] = scenario['pot_size']
                if 'bet_size' in scenario:
                    kwargs['bet_size'] = scenario['bet_size']
                
                result = solve_poker_hand(**kwargs)
                elapsed = (time.time() - start) * 1000
                
                times.append(elapsed)
                win_probs.append(result.win_probability)
            
            # Store results
            avg_time = sum(times) / len(times)
            avg_win = sum(win_probs) / len(win_probs)
            
            # Store results with additional metrics
            scenario_results = {
                'avg_time_ms': avg_time,
                'avg_win_probability': avg_win,
                'times': times,
                'win_probabilities': win_probs,
                'confidence_interval': result.confidence_interval,
                'actual_simulations': result.actual_simulations
            }
            
            # Add new feature results if available
            if result.icm_equity is not None:
                scenario_results['icm_equity'] = result.icm_equity
            if result.board_texture_score is not None:
                scenario_results['board_texture'] = result.board_texture_score
            if result.positional_advantage_score is not None:
                scenario_results['positional_advantage'] = result.positional_advantage_score
            if result.spr is not None:
                scenario_results['spr'] = result.spr
            if result.pot_odds is not None:
                scenario_results['pot_odds'] = result.pot_odds
            
            results['scenarios'][scenario['name']] = scenario_results
            
            print(f" {avg_time:.1f}ms, win={avg_win:.3f}")
        
        # Performance summary
        results['performance']['avg_time_ms'] = sum(
            s['avg_time_ms'] for s in results['scenarios'].values()
        ) / len(results['scenarios'])
        
        # Run modes comparison
        print("\nComparing simulation modes...")
        mode_results = {}
        for mode in ['fast', 'default', 'precision']:
            start = time.time()
            result = solve_poker_hand(
                hero_hand=['Q♠', 'Q♥'],
                num_opponents=2,
                simulation_mode=mode
            )
            elapsed = (time.time() - start) * 1000
            
            mode_results[mode] = {
                'time_ms': elapsed,
                'simulations': result.actual_simulations,
                'win_probability': result.win_probability
            }
            print(f"  {mode}: {elapsed:.1f}ms for {result.actual_simulations} sims")
        
        results['performance']['modes'] = mode_results
        
        return results
    
    def save_baseline(self, results: Dict[str, Any]) -> None:
        """Save results as new baseline."""
        with open(self.baseline_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nBaseline saved to {self.baseline_file}")
    
    def compare_to_baseline(self, current: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current results to baseline."""
        if not self.baseline_file.exists():
            print("No baseline found. Current results will become the baseline.")
            return {'status': 'no_baseline'}
        
        with open(self.baseline_file, 'r') as f:
            baseline = json.load(f)
        
        comparison = {
            'status': 'compared',
            'baseline_timestamp': baseline['timestamp'],
            'current_timestamp': current['timestamp'],
            'regressions': [],
            'improvements': [],
            'details': {}
        }
        
        # Compare each scenario
        for name, curr_data in current['scenarios'].items():
            if name not in baseline['scenarios']:
                continue
            
            base_data = baseline['scenarios'][name]
            
            # Performance comparison
            time_diff = curr_data['avg_time_ms'] - base_data['avg_time_ms']
            time_pct = (time_diff / base_data['avg_time_ms']) * 100
            
            # Accuracy comparison
            win_diff = abs(curr_data['avg_win_probability'] - base_data['avg_win_probability'])
            
            details = {
                'time_diff_ms': time_diff,
                'time_diff_pct': time_pct,
                'win_prob_diff': win_diff,
                'baseline_time': base_data['avg_time_ms'],
                'current_time': curr_data['avg_time_ms'],
                'baseline_win': base_data['avg_win_probability'],
                'current_win': curr_data['avg_win_probability']
            }
            
            comparison['details'][name] = details
            
            # Flag regressions (>10% slower or >2% accuracy change)
            if time_pct > 10:
                comparison['regressions'].append({
                    'scenario': name,
                    'type': 'performance',
                    'message': f"Performance regression: {time_pct:.1f}% slower"
                })
            
            if win_diff > 0.02:
                comparison['regressions'].append({
                    'scenario': name,
                    'type': 'accuracy',
                    'message': f"Accuracy change: {win_diff:.3f} difference"
                })
            
            # Flag improvements (>10% faster)
            if time_pct < -10:
                comparison['improvements'].append({
                    'scenario': name,
                    'type': 'performance',
                    'message': f"Performance improvement: {abs(time_pct):.1f}% faster"
                })
        
        return comparison
    
    def append_to_history(self, results: Dict[str, Any]) -> None:
        """Append results to history file."""
        history = []
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                history = json.load(f)
        
        # Extract summary for history
        summary = {
            'timestamp': results['timestamp'],
            'avg_time_ms': results['performance']['avg_time_ms'],
            'scenarios': {
                name: {
                    'time_ms': data['avg_time_ms'],
                    'win_prob': data['avg_win_probability'],
                    'icm_equity': data.get('icm_equity'),
                    'board_texture': data.get('board_texture'),
                    'spr': data.get('spr')
                }
                for name, data in results['scenarios'].items()
            }
        }
        
        history.append(summary)
        
        # Keep last 100 entries
        if len(history) > 100:
            history = history[-100:]
        
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def print_comparison_report(self, comparison: Dict[str, Any]) -> None:
        """Print a formatted comparison report."""
        if comparison['status'] == 'no_baseline':
            print("\nNo baseline found. Current results saved as baseline.")
            return
        
        print("\n" + "="*60)
        print("COMPARISON REPORT")
        print("="*60)
        print(f"Baseline: {comparison['baseline_timestamp']}")
        print(f"Current:  {comparison['current_timestamp']}")
        print()
        
        # Summary
        num_regressions = len(comparison['regressions'])
        num_improvements = len(comparison['improvements'])
        
        if num_regressions == 0 and num_improvements == 0:
            print("✅ No significant changes detected")
        else:
            if num_regressions > 0:
                print(f"⚠️  {num_regressions} regression(s) detected:")
                for reg in comparison['regressions']:
                    print(f"   - {reg['scenario']}: {reg['message']}")
            
            if num_improvements > 0:
                print(f"✨ {num_improvements} improvement(s) detected:")
                for imp in comparison['improvements']:
                    print(f"   - {imp['scenario']}: {imp['message']}")
        
        # Detailed results
        print("\nDetailed Results:")
        print("-" * 60)
        print(f"{'Scenario':<20} {'Time (ms)':<15} {'Win Prob':<15} {'Status'}")
        print("-" * 60)
        
        for name, details in comparison['details'].items():
            time_str = f"{details['current_time']:.1f} ({details['time_diff_pct']:+.1f}%)"
            win_str = f"{details['current_win']:.3f} ({details['win_prob_diff']:+.3f})"
            
            # Status icon
            if details['time_diff_pct'] > 10:
                status = "🔴"  # Regression
            elif details['time_diff_pct'] < -10:
                status = "🟢"  # Improvement
            else:
                status = "🟡"  # No significant change
            
            print(f"{name:<20} {time_str:<15} {win_str:<15} {status}")
        
        print("="*60)


def main():
    """Main entry point for comparison tool."""
    comparator = TestResultComparator()
    
    # Run tests
    print("Running poker solver tests...")
    results = comparator.run_standard_tests()
    
    # Compare to baseline
    comparison = comparator.compare_to_baseline(results)
    
    # Save to history
    comparator.append_to_history(results)
    
    # Print report
    comparator.print_comparison_report(comparison)
    
    # Update baseline if requested
    if len(sys.argv) > 1 and sys.argv[1] == '--update-baseline':
        comparator.save_baseline(results)
        print("\nBaseline updated!")
    elif comparison['status'] == 'no_baseline':
        comparator.save_baseline(results)
    
    # Exit with error if regressions found
    if comparison.get('regressions'):
        sys.exit(1)


if __name__ == "__main__":
    main()