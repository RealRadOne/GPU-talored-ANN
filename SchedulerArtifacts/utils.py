import pandas as pd
import numpy as np
import glob
from scipy.stats import weibull_min
import os


def get_stats(filename):
    df = pd.read_csv(filename)
    finish_map = df[df['event_type'] == 'FINISH'].groupby('task_id')['timestamp'].max()
    df_ret = df.groupby('task_id', as_index=False).agg(
                npts=('npts', 'max'),
                start_time=('timestamp', 'min'),
                interruptions=('event_type', lambda x: (x == 'INTERRUPT').sum())
            )
    df_ret['finish_time'] = df_ret['task_id'].map(finish_map)
    df_ret = df_ret.dropna()
    df_ret['duration_s'] = (df_ret['finish_time'] - df_ret['start_time']) / 1000.0
    return df_ret[['task_id', 'npts', 'start_time', 'interruptions', 'finish_time', 'duration_s']]

def get_aggregated_policy_stats(pattern, limit):
    files = sorted(glob.glob(pattern))
    files = files[:limit]
    all_runs = []
    print(f"Aggregating efficiency for {len(files)} policy runs...")
    for run_idx, f in enumerate(files):
        try:
            df_local = get_stats(f)
            df_selected = df_local[['task_id','npts', 'start_time','finish_time','duration_s', 'interruptions']].copy()
            df_selected['run_id'] = run_idx
            all_runs.append(df_selected)
        except Exception as e:
            continue    
    if not all_runs: return pd.DataFrame()
    return pd.concat(all_runs, ignore_index=True)

def preprocess_trace(source_trace_path):
    """Loads a raw trace file and calculates availability durations for each node."""
    df = pd.read_csv(source_trace_path)
    max_ts = df['timestamp'].max()
    avail_map = {}
    
    for gpu_id in df['gpu_id'].unique():
        gpu_events = df[df['gpu_id'] == gpu_id]
        add_t = gpu_events[gpu_events['event_type'] == 'add']['timestamp'].iloc[0]
        rem_events = gpu_events[gpu_events['event_type'] == 'remove']
        rem_t = rem_events['timestamp'].iloc[0] if not rem_events.empty else max_ts
        avail_map[gpu_id] = (rem_t - add_t) / 1000.0
    
    df['availability_duration'] = df['gpu_id'].map(avail_map)
    return df

def generate_synthetic_trace(source_df, num_events, seed=42, 
                             scale_modifier=1.0, lambda_modifier=1.0, p_batch_modifier=1.0):
    np.random.seed(seed)
    durations = source_df.groupby('gpu_id')['availability_duration'].first().dropna()
    shape, loc, scale = weibull_min.fit(durations, floc=0)
    
    # Apply modifier: < 1.0 means shorter lifespans
    scale *= scale_modifier
    
    add_events = source_df[source_df['event_type'] == 'add'].sort_values('timestamp')
    batch_sizes = add_events.groupby('timestamp').size()
    
    initial_batch_size = batch_sizes.iloc[0] if not batch_sizes.empty else 1
    
    num_waves = len(batch_sizes)
    duration_ms = add_events['timestamp'].max() - add_events['timestamp'].min()
    total_duration_seconds = max(duration_ms, 1.0) / 1000.0
    
    if num_waves > 1:
        lambda_arrival = (num_waves - 1) / total_duration_seconds
    else:
        lambda_arrival = 1.0 / 600.0 
        
    # Apply modifier: > 1.0 means faster arrivals
    lambda_arrival *= lambda_modifier
        
    p_batch = 1.0 / max(batch_sizes.mean(), 1.0)
    # Apply modifier: < 1.0 means larger batches
    p_batch *= p_batch_modifier
    p_batch = min(max(p_batch, 0.01), 1.0)

    node_pool = list(range(1, 178))
    active_nodes = {} 
    
    synthetic_events = []
    current_time_s = 0

    for b in range(num_events):
        if b > 0:
            inter_arrival = np.random.exponential(scale=1.0/lambda_arrival)
            inter_arrival = round(inter_arrival / 60.0) * 60.0
            if inter_arrival < 60.0: inter_arrival = 60.0 
            current_time_s += inter_arrival
    
        for nid in list(active_nodes.keys()):
            if active_nodes[nid] <= current_time_s:
                del active_nodes[nid]

        batch_size = initial_batch_size if b == 0 else np.random.geometric(p_batch)
            
        for _ in range(batch_size):
            available_nodes = [n for n in node_pool if n not in active_nodes]
            if not available_nodes:
                node_id = max(node_pool) + 1
                node_pool.append(node_id)
            else:
                node_id = min(available_nodes)
            
            duration = np.random.weibull(shape) * scale
            duration = max(duration, 60.0) 
            removal_time_s = current_time_s + duration
            active_nodes[node_id] = removal_time_s
            
            gpu_id = f"node{node_id}"
            # Match simulator expectation: timestamp, event_type, gpu_id
            synthetic_events.append({'timestamp': int(current_time_s*1000), 'event_type': 'add', 'gpu_id': gpu_id})
            synthetic_events.append({'timestamp': int(removal_time_s*1000), 'event_type': 'remove', 'gpu_id': gpu_id})
            
    synthetic_df = pd.DataFrame(synthetic_events)
    synthetic_df = synthetic_df[['timestamp', 'event_type', 'gpu_id']]
    synthetic_df.sort_values(['timestamp', 'event_type'], ascending=[True, True], inplace=True)
    
    # Calculate availability duration for the synthetic trace
    max_ts = synthetic_df['timestamp'].max()
    avail_map = {}
    for gpu_id in synthetic_df['gpu_id'].unique():
        gpu_events = synthetic_df[synthetic_df['gpu_id'] == gpu_id]
        add_t = gpu_events[gpu_events['event_type'] == 'add']['timestamp'].iloc[0]
        rem_events = gpu_events[gpu_events['event_type'] == 'remove']
        rem_t = rem_events['timestamp'].iloc[0] if not rem_events.empty else max_ts
        avail_map[gpu_id] = (rem_t - add_t) / 1000.0
    
    synthetic_df['availability_duration'] = synthetic_df['gpu_id'].map(avail_map)
    return synthetic_df     
