import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.stats import weibull_min, expon, geom
import numpy as np
import os

def set_plot_style():
    mpl.rcParams.update({
        'font.size': 14,            
        'axes.labelsize': 16,       
        'xtick.labelsize': 14,      
        'ytick.labelsize': 14,
        'axes.titlesize': 18,
        'legend.fontsize': 14,
        'figure.autolayout': True, 
        'text.usetex': False        
    })

def plot_geometric_batches(gpu_trace, label_prefix, save_dir='.'):
    """Batched arrival histogram (Geometric distribution)"""
    add_events = gpu_trace[gpu_trace['event_type'] == 'add'].copy()
    add_events['ts_s'] = add_events['timestamp'] / 1000.0
    batch_sizes = add_events.groupby('ts_s').size()
    
    if batch_sizes.empty: return
    
    mean_batch = max(batch_sizes.mean(), 1.0)
    p_batch = 1.0 / mean_batch
    
    plt.figure(figsize=(16, 9))
    max_batch = batch_sizes.max()
    bins = np.arange(1, max_batch + 2) - 0.5
    
    sns.histplot(batch_sizes, bins=bins, kde=False, stat='density', color='purple', label=f'{label_prefix} Batch Sizes', alpha=0.6)
    
    # Overlay Geometric PMF
    x = np.arange(1, max_batch + 1)
    pmf = geom.pmf(x, p_batch)
    plt.plot(x, pmf, 'ko-', label=f'Geometric Fit (p={p_batch:.2f})', linewidth=2, markersize=8)
    
    plt.xlabel('Batch Size (GPUs)')
    plt.ylabel('Density')
    plt.title(f'{label_prefix} Batched Arrival Histogram')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, f'{label_prefix}_batch_geometric.png'), bbox_inches='tight')
    plt.close()

def plot_exponential_arrivals(gpu_trace, label_prefix, save_dir='.'):
    """Inter-arrival time histogram (Exponential distribution)"""
    add_events = gpu_trace[gpu_trace['event_type'] == 'add'].sort_values('timestamp')
    unique_timestamps = add_events['timestamp'].unique() / 1000.0
    
    if len(unique_timestamps) < 2: return
    
    inter_arrivals = np.diff(unique_timestamps)
    # Filter out 0 gaps just in case
    inter_arrivals = inter_arrivals[inter_arrivals > 0]
    
    if len(inter_arrivals) < 2: return
    
    loc, scale = expon.fit(inter_arrivals, floc=0)
    
    plt.figure(figsize=(16, 9))
    sns.histplot(inter_arrivals, bins=50, kde=False, stat='density', color='orange', label=f'{label_prefix} Inter-Arrivals', alpha=0.6)
    
    x = np.linspace(0, inter_arrivals.max(), 1000)
    pdf = expon.pdf(x, loc, scale)
    plt.plot(x, pdf, 'k-', label=f'Exponential Fit (scale={scale:.2f}s)', linewidth=2)
    
    plt.xlabel('Inter-Arrival Time (s)')
    plt.ylabel('Density')
    plt.title(f'{label_prefix} Inter-Arrival Time Histogram')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, f'{label_prefix}_interarrival_exponential.png'), bbox_inches='tight')
    plt.close()

def plot_availability_weibull(gpu_trace, label_prefix, save_dir='.'):
    """GPU availability histogram (Weibull distribution)"""
    if 'availability_duration' not in gpu_trace.columns:
        # Fallback if preprocessing wasn't perfect, though it should be.
        max_ts = gpu_trace['timestamp'].max()
        avail_map = {}
        for gpu_id in gpu_trace['gpu_id'].unique():
            gpu_events = gpu_trace[gpu_trace['gpu_id'] == gpu_id]
            add_events = gpu_events[gpu_events['event_type'] == 'add']
            rem_events = gpu_events[gpu_events['event_type'] == 'remove']
            
            if add_events.empty: continue
            add_t = add_events['timestamp'].iloc[0]
            rem_t = rem_events['timestamp'].iloc[0] if not rem_events.empty else max_ts
            avail_map[gpu_id] = (rem_t - add_t) / 1000.0
        
        # Deduplicate to get one lifespan per GPU
        durations = np.array(list(avail_map.values()))
    else:
        durations = gpu_trace.groupby('gpu_id')['availability_duration'].first().dropna().values
        
    durations = durations[durations > 0]
    if len(durations) < 2: return
    
    shape, loc, scale = weibull_min.fit(durations, floc=0)
    
    plt.figure(figsize=(16, 9))
    sns.histplot(durations, bins=50, kde=False, stat="density", color='green', label=f'{label_prefix} Lifespans', alpha=0.6)
    
    x = np.linspace(0, durations.max(), 1000)
    pdf = weibull_min.pdf(x, shape, loc, scale)
    plt.plot(x, pdf, 'k-', label=f'Weibull Fit (shape={shape:.2f}, scale={scale:.2f})', linewidth=2)
    
    plt.xlabel('Availability Duration (s)')
    plt.ylabel('Density')
    plt.title(f'{label_prefix} Availability Histogram')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, f'{label_prefix}_availability_weibull.png'), bbox_inches='tight')
    plt.close()

def plot_stair(df, label, color, x_col, y_col):
    plt.step(df[x_col], df[y_col], where='post', label=label, color=color, linewidth=2)
    px, py = df[x_col].iloc[-1], df[y_col].iloc[-1]
    plt.scatter(px, py, color='black', marker='*', s=100, zorder=5)
    plt.text(px + 2, py, f'{label}: {py:,.0f}s', fontsize=14, fontweight='bold', color=color, ha='left', va='center')

def plot_overall_comparison(df, save_dir='.'):
    """
    Generates a grouped bar chart comparing Retry-Prioritized vs. Random Baseline
    across all scenarios.
    df columns expected: ['Scenario', 'Policy', 'Duration']
    """
    plt.style.use('default')
    set_plot_style()
    fig, ax = plt.subplots(figsize=(14, 8))
    
    scenarios = df['Scenario'].unique()
    x = np.arange(len(scenarios))
    width = 0.35
    
    # Extract values for each policy
    durations_policy = []
    durations_random = []
    for s in scenarios:
        val_p = df[(df['Scenario'] == s) & (df['Policy'] == 'Retry-Prioritized')]['Duration'].values
        val_r = df[(df['Scenario'] == s) & (df['Policy'] == 'Random Baseline')]['Duration'].values
        durations_policy.append(val_p[0] if len(val_p) > 0 else 0)
        durations_random.append(val_r[0] if len(val_r) > 0 else 0)
    
    # Plotting using the exact colors and alpha/edge properties of the other plots
    rects1 = ax.bar(x - width/2, durations_policy, width, label='Retry-Prioritized', color='yellow', alpha=0.7, edgecolor='black', linewidth=1.5)
    rects2 = ax.bar(x + width/2, durations_random, width, label='Random Baseline', color='green', alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Simulation Scenario', labelpad=15)
    ax.set_ylabel('Total Duration (s)', labelpad=15)
    ax.set_title('Overall Policy Performance Comparison Across Scenarios', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend(title='Policy', frameon=True, facecolor='white', edgecolor='none')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Label bars with their height values
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if not np.isnan(height) and height > 0:
                ax.annotate(f'{height:,.0f}s',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 5),  # 5 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=12,
                            fontweight='bold')
                            
    autolabel(rects1)
    autolabel(rects2)
    
    plt.savefig(os.path.join(save_dir, 'overall_comparison.png'), bbox_inches='tight')
    plt.close()

