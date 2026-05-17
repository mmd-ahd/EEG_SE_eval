import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

derivatives_dir = r"ds003505\derivatives"
metrics_dir = os.path.join(derivatives_dir, 'resolution_metrics')
ga_dir = os.path.join(metrics_dir, 'grand_averages')
stats_dir = os.path.join(metrics_dir, 'statistics')

os.makedirs(stats_dir, exist_ok=True)

SPACING = 'ico4' 
methods = ['MNE', 'dSPM', 'sLORETA', 'eLORETA']
metrics = ['PSF_PLE', 'CTF_PLE', 'PSF_SD', 'CTF_SD']

sns.set_theme(style="white", rc={"axes.grid": True, "grid.linestyle": ":", "grid.color": ".8"})
palette = sns.color_palette("mako", n_colors=len(methods))

fig, axes = plt.subplots(nrows=len(methods), ncols=len(metrics), figsize=(18, 10), sharex='col')

for row, method in enumerate(methods):
    method_color = palette[row]
    
    for col, metric in enumerate(metrics):
        ax = axes[row, col]
        ga_fname = os.path.join(ga_dir, f"GA_{SPACING}_{method}_{metric}-lh.stc")
        
        try:
            ga_stc = mne.read_source_estimate(ga_fname)
            data = ga_stc.data.flatten()
            
            mean_val = np.mean(data)
            std_val = np.std(data)
            median_val = np.median(data)
            
            bins = np.linspace(0, 10.0, 40)
            xlabel = 'Localization Error (cm)' if 'PLE' in metric else 'Spatial Deviation (cm)'
            
            use_kde = True if std_val > 0.01 else False
            
            sns.histplot(
                data, bins=bins, kde=use_kde, color=method_color, 
                edgecolor='white', linewidth=0.5, alpha=0.7, ax=ax,
                line_kws={'linewidth': 2} if use_kde else None
            )
            
            ax.axvline(mean_val, color='#003049', linestyle='--', linewidth=2, alpha=0.9) 
            
            stat_text = f"$\mu$: {mean_val:.1f} cm\n$\sigma$: {std_val:.1f} cm\n$M$: {median_val:.1f} cm"
            ax.text(0.95, 0.85, stat_text, transform=ax.transAxes, fontsize=11,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor='white', edgecolor='lightgray', alpha=0.85))
            
            ax.set_ylabel('') 
            ax.set_yticks([]) 
            
            if row == 0:
                ax.set_title(metric.replace('_', ' '), fontsize=16, fontweight='bold', pad=15)
            if col == 0:
                ax.set_ylabel(method, fontsize=16, fontweight='bold', labelpad=15)
            if row == len(methods) - 1:
                ax.set_xlabel(xlabel, fontsize=14, fontweight='500', labelpad=10)
            else:
                ax.set_xlabel('')
                
            sns.despine(ax=ax, left=True, top=True, right=True)
            
        except Exception:
            ax.text(0.5, 0.5, "Data Missing", ha='center', va='center', color='gray')
            ax.axis('off')

plt.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.08, wspace=0.1, hspace=0.2)

out_fig = os.path.join(stats_dir, f'Histograms_Modern_{SPACING}.png')
plt.savefig(out_fig, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()