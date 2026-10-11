import os
import glob
import subprocess
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from visualization.utils import (plot_geometric_batches, plot_exponential_arrivals, 
                                 plot_availability_weibull, plot_stair, set_plot_style, 
                                 plot_overall_comparison)
from SchedulerArtifacts.utils import generate_synthetic_trace, preprocess_trace

class ScaleGANNBenchmarker:
    def __init__(self, source_trace="SchedulerArtifacts/GPU_Traces/p3-trace.csv"):
        self.source_trace = source_trace
        self.base_results_dir = "SchedulerArtifacts/benchmark_results"
        self.policy_results = "SchedulerArtifacts/results/policy"
        self.random_results = "SchedulerArtifacts/results/random"
        self.comparison_data = []
        os.makedirs(self.base_results_dir, exist_ok=True)
        os.makedirs(self.policy_results, exist_ok=True)
        os.makedirs(self.random_results, exist_ok=True)
        set_plot_style()

    def analyze_and_run(self, trace_df, label, trace_path, num_sim_runs=5):
        label_slug = label.lower().replace(' ', '_')
        save_dir = os.path.join(self.base_results_dir, f"{label_slug}_analysis")
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Analyzing {label} patterns...")
        plot_geometric_batches(trace_df, label, save_dir)
        plot_exponential_arrivals(trace_df, label, save_dir)
        plot_availability_weibull(trace_df, label, save_dir)

        print(f"Running simulations for {label}...")
        for f in glob.glob(f"{self.policy_results}/*.csv") + glob.glob(f"{self.random_results}/*.csv"):
            if os.path.exists(f): os.remove(f)
        subprocess.run(["./build/testSimulation", str(num_sim_runs), trace_path], check=True)
        
        policy_avg = self._aggregate(f"{self.policy_results}/sim_events_*.csv")
        random_avg = self._aggregate(f"{self.random_results}/sim_events_*.csv")

        plt.figure(figsize=(16, 9))
        plot_stair(policy_avg, 'Retry-Prioritized', 'blue', 'task_id', 'cumulative_avg')
        plot_stair(random_avg, 'Random Baseline', 'red', 'task_id', 'cumulative_avg')
        plt.xlabel('Shards Computed')
        plt.ylabel('Cumulative Duration (s)')
        plt.title(f'Performance Comparison: {label}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'{label_slug}_comparison.png'), bbox_inches='tight')
        plt.show()
        
        # Collect overall comparison metrics
        self.comparison_data.append({
            'Scenario': label,
            'Policy': 'Retry-Prioritized',
            'Duration': policy_avg['cumulative_avg'].iloc[-1]
        })
        self.comparison_data.append({
            'Scenario': label,
            'Policy': 'Random Baseline',
            'Duration': random_avg['cumulative_avg'].iloc[-1]
        })

    def _aggregate(self, pattern):
        all_runs = []
        for f in glob.glob(pattern):
            df = pd.read_csv(f)
            finishes = df[df['event_type'] == 'FINISH'].groupby('task_id')['timestamp'].max()
            stats = df.groupby('task_id', as_index=False).agg(start=('timestamp', 'min'))
            stats['finish'] = stats['task_id'].map(finishes)
            stats = stats.dropna()
            stats['dur'] = (stats['finish'] - stats['start']) / 1000.0
            all_runs.append(stats)
        avg = pd.concat(all_runs).groupby('task_id', as_index=False)['dur'].mean()
        avg.sort_values('task_id', inplace=True)
        avg['cumulative_avg'] = avg['dur'].cumsum()
        return avg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ScaleGANN Benchmarking Pipeline")
    parser.add_argument('--plot-only', action='store_true', help='Only generate the final overall comparison plot from cached CSV data')
    args = parser.parse_args()

    bench = ScaleGANNBenchmarker()
    csv_path = os.path.join(bench.base_results_dir, "comparison_data.csv")

    if args.plot_only:
        if os.path.exists(csv_path):
            print(f"Generating overall comparison plot from cached data at {csv_path}...")
            comp_df = pd.read_csv(csv_path)
            plot_overall_comparison(comp_df, bench.base_results_dir)
            print("Successfully updated overall comparison plot.")
        else:
            print(f"Error: Cached data not found at {csv_path}. Please run the full pipeline once first.")
    else:
        # 1. Original Trace
        orig_df = preprocess_trace(bench.source_trace)
        bench.analyze_and_run(orig_df, "Original Trace", bench.source_trace)
        
        # 2. Variations (Genuine Scenario Mutations)
        variations = [
            # High Volatility: GPUs die 5x faster, arrive 5x faster. Tests interrupt handling.
            {"label": "Scenario A High Volatility", "events": 200, "seed": 42, 
             "scale_mod": 0.2, "lambda_mod": 5.0, "p_batch_mod": 1.0},

            # Burst Arrivals: GPUs arrive in much larger groups, less frequently. Tests queue mapping.
            {"label": "Scenario B Burst Arrivals", "events": 200, "seed": 101, 
             "scale_mod": 1.0, "lambda_mod": 0.5, "p_batch_mod": 0.2},

            # Sustained Capacity: GPUs live 3x longer, arrive slower. Tests baseline throughput.
            {"label": "Scenario C Sustained Capacity", "events": 200, "seed": 202, 
             "scale_mod": 3.0, "lambda_mod": 0.5, "p_batch_mod": 1.0}
        ]

        for v in variations:
            v_path = f"SchedulerArtifacts/GPU_Traces/{v['label'].lower().replace(' ', '_')}.csv"
            v_df = generate_synthetic_trace(
                orig_df, 
                v['events'], 
                v['seed'],
                scale_modifier=v['scale_mod'],
                lambda_modifier=v['lambda_mod'],
                p_batch_modifier=v['p_batch_mod']
            )
            v_df.to_csv(v_path, index=False)
            bench.analyze_and_run(v_df, v['label'], v_path)
            
        # Generate and cache overall comparison plot
        if bench.comparison_data:
            comp_df = pd.DataFrame(bench.comparison_data)
            comp_df.to_csv(csv_path, index=False)
            plot_overall_comparison(comp_df, bench.base_results_dir)
            print("Successfully saved comparison data and generated overall comparison plot.")
