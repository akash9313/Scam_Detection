"""
Generate publication-quality IEEE figures for SpamAdapt Condensed Paper.
Figure 1: Grouped bar chart comparing Accuracy and Macro F1 across UCI Test (N=774) and External Benchmark (N=220).
Figure 2: Dual-panel figure comparing Storage Footprint (MB) and CPU Inference Latency (ms).
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Set publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Times'],
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9.5,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8,
    'figure.titlesize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.03,
})

out_dir = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# FIGURE 1: Performance Comparison (UCI Test N=774 vs External Benchmark N=220)
# ==============================================================================
def generate_figure1():
    models = ['Full FT v2\n(\\tau=0.10)', 'LoRA v2\n(\\tau=0.18)', 'DoRA v2\n(\\tau=0.18)', 'lora_v3\n(\\tau=0.50)']
    
    # Accuracy values (%)
    uci_acc = [98.58, 98.06, 98.06, 97.03]
    ext_acc = [87.73, 76.82, 77.73, 83.18]
    
    # Macro F1 values (%)
    uci_f1 = [96.74, 95.44, 95.44, 93.19]
    ext_f1 = [87.72, 76.81, 77.72, 83.10]
    
    x = np.arange(len(models))
    width = 0.20
    
    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    
    rects1 = ax.bar(x - 1.5*width, uci_acc, width, label='UCI Acc (N=774)', color='#1f77b4', edgecolor='black', linewidth=0.6)
    rects2 = ax.bar(x - 0.5*width, uci_f1, width, label='UCI Macro F1', color='#aec7e8', edgecolor='black', linewidth=0.6)
    rects3 = ax.bar(x + 0.5*width, ext_acc, width, label='Ext Acc (N=220)', color='#d62728', edgecolor='black', linewidth=0.6)
    rects4 = ax.bar(x + 1.5*width, ext_f1, width, label='Ext Macro F1', color='#ff9896', edgecolor='black', linewidth=0.6)
    
    ax.set_ylabel('Score (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(['Full FT v2', 'LoRA v2', 'DoRA v2', 'lora_v3'], fontweight='normal')
    ax.set_ylim(65, 103)
    ax.set_yticks([70, 80, 90, 100])
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(loc='lower left', ncol=2, framealpha=0.9, handlelength=1.2, handletextpad=0.4, columnspacing=0.8)
    
    # Add value annotations on top of bars
    for rects, fmt in [(rects1, '%.1f'), (rects3, '%.1f')]:
        for r in rects:
            h = r.get_height()
            ax.annotate(fmt % h,
                        xy=(r.get_x() + r.get_width() / 2, h),
                        xytext=(0, 2),  # 2 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=6.5, rotation=0)
            
    fig.tight_layout()
    pdf_path = os.path.join(out_dir, 'fig_performance_comparison.pdf')
    png_path = os.path.join(out_dir, 'fig_performance_comparison.png')
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    print(f"Figure 1 saved to {pdf_path} and {png_path}")

# ==============================================================================
# FIGURE 2: Computational Efficiency & Storage Footprint
# Dual-panel: Panel (a) Storage Footprint (MB), Panel (b) Mean CPU Latency (ms)
# ==============================================================================
def generate_figure2():
    models = ['Full FT v2', 'LoRA v2', 'DoRA v2']
    storage = [418.35, 1.82, 1.90]  # Total serialized directory footprint (MB)
    latency = [99.48, 92.05, 110.00] # Mean per-message CPU inference latency (ms)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.45, 1.85))
    
    # Panel (a): Storage Footprint (Logarithmic scale to display 418 MB vs 1.8 MB clearly)
    bars1 = ax1.bar(models, storage, color=['#2ca02c', '#1f77b4', '#ff7f0e'], edgecolor='black', linewidth=0.6, width=0.55)
    ax1.set_ylabel('Storage (MB, log scale)')
    ax1.set_yscale('log')
    ax1.set_ylim(0.5, 1200)
    ax1.set_yticks([1, 10, 100, 1000])
    ax1.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    ax1.set_title('(a) Checkpoint Size', fontsize=8.5)
    ax1.tick_params(axis='x', rotation=25)
    
    for b, val in zip(bars1, storage):
        h = b.get_height()
        ax1.annotate(f'{val:.2f}M' if val < 10 else f'{val:.1f}M',
                     xy=(b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 2), textcoords="offset points",
                     ha='center', va='bottom', fontsize=6.5)
        
    # Panel (b): CPU Latency (ms)
    bars2 = ax2.bar(models, latency, color=['#2ca02c', '#1f77b4', '#ff7f0e'], edgecolor='black', linewidth=0.6, width=0.55)
    ax2.set_ylabel('Mean Latency (ms)')
    ax2.set_ylim(0, 135)
    ax2.set_yticks([0, 30, 60, 90, 120])
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    ax2.set_title('(b) CPU Inference Time', fontsize=8.5)
    ax2.tick_params(axis='x', rotation=25)
    
    for b, val in zip(bars2, latency):
        h = b.get_height()
        ax2.annotate(f'{val:.1f}',
                     xy=(b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 2), textcoords="offset points",
                     ha='center', va='bottom', fontsize=6.5)
        
    fig.tight_layout()
    pdf_path = os.path.join(out_dir, 'fig_efficiency_latency.pdf')
    png_path = os.path.join(out_dir, 'fig_efficiency_latency.png')
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    print(f"Figure 2 saved to {pdf_path} and {png_path}")

if __name__ == '__main__':
    generate_figure1()
    generate_figure2()
